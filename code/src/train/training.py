# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contributor(s): 2019 Paula Czarnowska <pjc211@cam.ac.uk>

import collections
import random
import sys
import time
from itertools import groupby

import numpy as np
from src.train.matching import Matching, Alignment
from src.utils.cupy_utils import *
from src.train.options import ModelType

#################################
#       VALIDATION FUNCTIONS    #
#################################


def test(test_dict, test_coverage, xw, zw):
    # we skip length normalization here
    print('Evaluating translation...')
    src = test_dict.keys()
    BATCH_SIZE = 500

    # Find translations
    translation = collections.defaultdict(int)

    # we just use nearest neighbour for retrieval
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        nn = similarities.argmax(axis=1).tolist()
        for k in range(j - i):
            translation[src[i + k]] = nn[k]

    # Compute accuracy
    accuracy = np.mean(
        [1 if translation[i] in test_dict[i] else 0 for i in src])
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(
        test_coverage, accuracy))


#################################
#       TRAINING CODE           #
#################################


def get_mapped_vectors(x, z, src_indices, trg_indices, opts):
    """
    Learns the matrix transformation(s) and uses it/them to transform the
    given embeddings. It returns two embedding matrices resulting from
    applying the transformation(s).
    :param x:
    :param z:
    :param src_indices:
    :param trg_indices:
    :return:
    """
    if opts.orthogonal:  # orthogonal mapping solving Procrustes problem
        u, s, vt = opts.xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
        w = vt.T.dot(u.T)
        xw = x.dot(w)  # the projected source embeddings
        zw = z
    elif opts.unconstrained:  # unconstrained mapping
        x_pseudoinv = opts.xp.linalg.inv(
            x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
        w = x_pseudoinv.dot(z[trg_indices])
        xw = x.dot(w)
        zw = z
    else:  # advanced mapping
        xw = x
        zw = z
        # STEP 1: Whitening
        def whitening_transformation(m):
            u, s, vt = opts.xp.linalg.svd(m, full_matrices=False)
            return vt.T.dot(opts.xp.diag(1/s)).dot(vt)
        if opts.whiten:
            wx1 = whitening_transformation(xw[src_indices])
            wz1 = whitening_transformation(zw[trg_indices])
            xw = xw.dot(wx1)
            zw = zw.dot(wz1)

        # STEP 2: Orthogonal mapping
        wx2, s, wz2_t = opts.xp.linalg.svd(
            xw[src_indices].T.dot(zw[trg_indices]))
        wz2 = wz2_t.T
        xw = xw.dot(wx2)
        zw = zw.dot(wz2)

        # STEP 3: Re-weighting
        xw *= s**opts.src_reweight
        zw *= s**opts.trg_reweight

        # STEP 4: De-whitening
        if opts.src_dewhiten == 'src':
            xw = xw.dot(wx2.T.dot(opts.xp.linalg.inv(wx1)).dot(wx2))
        elif opts.src_dewhiten == 'trg':
            xw = xw.dot(wz2.T.dot(opts.xp.linalg.inv(wz1)).dot(wz2))
        if opts.trg_dewhiten == 'src':
            zw = zw.dot(wx2.T.dot(opts.xp.linalg.inv(wx1)).dot(wx2))
        elif opts.trg_dewhiten == 'trg':
            zw = zw.dot(wz2.T.dot(opts.xp.linalg.inv(wz1)).dot(wz2))

        # STEP 5: Dimensionality reduction
        if opts.dim_reduction > 0:
            xw = xw[:, :opts.dim_reduction]
            zw = zw[:, :opts.dim_reduction]
    return xw, zw, w


########################################
#         LAT-MORPH MODEL              #
########################################

def get_matrix_transformations(
        x, z, src_indices, trg_indices, tag_indices, tag2xw, opts):
    """
    Uses solution to the orthogonal Procrustes problem to get a transformation
    matrix for each tag. It returns an array holding transformed embeddings -
    one transformed embedding matrix for each tag.
    :param x:
    :param z:
    :param src_indices:
    :param trg_indices:
    :param tag_indices:
    :param tag2xw:
    :return:
    """
    # group the pairs by the morph tag
    keyfunc = lambda x: x[0]
    sorted_indices = sorted(zip(
        tag_indices, src_indices, trg_indices), key=keyfunc)
    morph_grouped = groupby(sorted_indices, keyfunc)

    tags_covered = 0
    # for each tag: get the mapping and transform all embeddings
    # (save the result in tag2xw)
    for tag_ind, group in morph_grouped:
        tags_covered += 1
        group = opts.xp.asarray(list(group), dtype=opts.dtype)
        src_inds, trg_inds = group[:, 1], group[:, 2]
        proc = z[trg_inds].T.dot(x[src_inds])  # , out=procrutes_tmp)
        u, s, vt = opts.xp.linalg.svd(proc)
        mtrans = vt.T.dot(u.T)  # , out=mtrans_tmp)
        x.dot(mtrans, out=tag2xw[tag_ind])

    if tags_covered < len(tag2xw):
        print(f"WARNING: Transformation learned for only {tags_covered} " +
              f"out of {len(tag2xw)} tags.")
    return tag2xw


def get_xw(xw, x_tags, src_indices, tag_indices, tag2xw):
    """
    Fills in the given xw array with the mapped embeddings. Each embedding
    is mapped according to the tag of the corresponding source word - if the
    tag is unknown and has not been determiend during the matching it is
    sampled from the distribution p(tag|s).
    :param xw:
    :param x_tags:
    :param src_indices:
    :param tag_indices:
    :param tag2xw:
    :return:
    """
    # assuming uniform for now
    matched_indices = set(src_indices)
    rand_guesses = 0

    for i in range(len(xw)):
        if i in matched_indices:
            index = cms.xp.where(src_indices == i)[0][0]
            tag = tag_indices[index]
        elif x_tags[i]:
            tag = x_tags[i][0]  # Take the first available tag
        else:
            tag = random.randint(0, len(tag2xw) - 1)  # uniform distribution
            rand_guesses += 1
        xw[i] = tag2xw[tag][i]
    print(f"Randomly guessed tags: {rand_guesses} out of {len(xw)}")
    return xw


########################################
#         GENERATING TAG MASKS         #
########################################


def create_tag_masks(opts, n_rows):
    """
    Generates a number of 2D lists - each mapping a specific tag id to the
    indices of words associated with that tag. If the tag for a word is unknown
    its index is added to the indices' lists of *all* tags.

    A 2D list is created for each chunk (determined by the cms.args.chunk_size
    parameter) since the masks are used in retrieve_ids function, which scans
    through the list of source embedding one chunk at the time.
    :param x_tags:
    :param tag_count:
    :return:
    """
    tag_count = len(opts.tag2ind)

    masks_dict = {}  # a set of tag masks for every chunk
    # is and js are exactly as those used in mathing
    for i in range(0, n_rows, opts.chunk_size):
        j = min(n_rows, i + opts.chunk_size)
        masks = [[] for _ in range(tag_count)]
        for z in range(i, j):
            tag_list = opts.src_data.tags_ids[z]
            if not tag_list:
                for mask in masks:
                    # z-i because it's an index to the *chunk* of the original x
                    mask.append(z - i)
            else:
                for tag in tag_list:
                    masks[tag].append(z - i)
        masks_dict[(i, j)] = opts.xp.asarray(
            [opts.xp.array(m) for m in masks], dtype=opts.dtype)
    return masks_dict


def create_full_tag_mask(y_tags, tag_count, n_rows):
    masks = [[] for _ in range(tag_count)]
    for z in range(n_rows):
        tag_list = y_tags[z]
        if not tag_list:
            for mask in masks:
                mask.append(z)
        else:
            for tag in tag_list:
                masks[tag].append(z)
    return masks


def create_split_tag_mask(y_tags, tag_count, max_dimz):
    mask_dict = {}
    # there is a tag for every word in the target matrix (or UNK tag)
    zwlen = len(y_tags)

    for k in range(0, zwlen, max_dimz):
        l = min(zwlen, k + max_dimz)
        masks = [[] for _ in range(tag_count)]
        for z in range(k, l):
            tag_list = y_tags[z]
            if not tag_list:
                for mask in masks:
                    mask.append(z-k)  # unknown tag, append to all masks
            else:
                for tag in tag_list:
                    masks[tag].append(z-k)
        mask_dict[(k, l)] = masks
    # a dicitonary mapping indexes to an list of masks - one per tag
    return mask_dict


def get_good_inds(opts, n_rows=None):
    """
    Used if matching constraints are imposed - the function goes through
    all the words in the source vocabulary and for each determines
    the ids of the words in target vocabulary that share the same tag.
    The returned is a dictionary mapping each chunk of the source matrix
    (determined by args.chunk_size) to a tuple of 2 lists: source ids and target ids,
    each of the same length (= word id in ith element of source list has a
    matching tag with the word id in ith element of the target list)

    The 'chunking' enables easy use of the 'good inds' during training.
    :param opts:
    :param n_rows:
    :return:
    """
    inds_dict = {}  # a set of tag masks for every chunk
    src, trg = opts.src_data, opts.trg_data
    tag_count = len(opts.tag2ind)

    if opts.lat_model:
        target_masks = create_full_tag_mask(trg.tag_ids, tag_count, n_rows)
        # is and js are exactly as those used in mathing
        for i in range(0, n_rows, opts.chunk_size):
            j = min(n_rows, i + opts.chunk_size)
            xs, ys = [], []
            for z in range(i, j):
                tag_list = src.tag_ids[z]
                if tag_list:
                    for tag in tag_list:
                        yss = target_masks[tag]
                        xss = [z-i] * len(yss)
                        xs += xss
                        ys += yss
            inds_dict[(i, j)] = (xs, ys)
    else:
        xwlen = len(src.tag_ids)
        zwlen = len(trg.tag_ids)

        target_masks = create_split_tag_mask(
            trg.tag_ids, tag_count, opts.max_dimz)
        for i in range(0, xwlen, opts.max_dimx):
            j = min(xwlen, i + opts.max_dimx)
            for k in range(0, zwlen, opts.max_dimz):
                l = min(zwlen, k + opts.max_dimz)

                tmasks = target_masks[(k, l)]
                xs, ys = [], []
                for z in range(i, j):
                    tag_list = src.tag_ids[z]
                    if tag_list:
                        for tag in tag_list:
                            yss = tmasks[tag]
                            xss = [z-i] * len(yss)
                            xs += xss
                            ys += yss
                inds_dict[(i, j, k, l)] = (xs, ys)

    return inds_dict


########################################
#              TRAINING                #
########################################

def setup_for_training(opts, x, z):
    """
    Fill in the given Options object with fields necessary for training
    :param opts:
    :param x:
    :param z:
    :return:
    """
    tag_count = 0 if not opts.tag2ind else len(opts.tag2ind)
    tag2xw, matching = None, None

    if opts.lat_model:
        rc = opts.rank_constr
        n_rows = rc if rc and (rc < x.shape[0]) else x.shape[0]
        if n_rows > z.shape[0]:
            n_rows = z.shape[0]
        if opts.verbose:
            print(f"Considering only {n_rows} top words from each language")
        matching = Matching(n_rows, tag_count, opts)
    else:
        n_rows = None
        opts.max_dimx = opts.max_dimx if opts.max_dimx < x.shape[0] \
            else x.shape[0]
        opts.max_dimz = opts.max_dimz if opts.max_dimz < z.shape[0] \
            else z.shape[0]
        print(opts.max_dimx, opts.max_dimz)
        matching = Alignment(opts.max_dimx, opts.max_dimz, opts)

    if opts.model_type == ModelType.LAT_MORPH:
        tag2xw = opts.xp.asarray(
            [x.copy() for _ in range(tag_count)], dtype=opts.dtype)
        opts.src_tag_masks = create_tag_masks(opts, n_rows)

    if opts.match_constr:
        opts.good_inds = get_good_inds(opts, n_rows)
    return x, z, tag2xw, matching


def print_dict_sample(opts, src_indices, trg_indices, n=15):
    """
    Print random n pairs from the newly created dictionary. If morphology is
    used then the printed pairs also display tag ids corresponding to the
    words' tags.
    :param opts:
    :param src_indices:
    :param trg_indices:
    :param n:
    :return:
    """
    src, trg = opts.src_data, opts.trg_data

    random_inds = [random.randint(0, len(src_indices)-1) for _ in range(n)]
    if opts.match_constr or opts.model_type == ModelType.LAT_MORPH:
        for i in random_inds:
            print(src.words[src_indices[i]], " (",
                  ' '.join([str(a) for a in src.tag_ids[src_indices[i]]]),
                  " ", ' '.join([str(a) for a in trg.tag_ids[trg_indices[i]]]), ") ",
                  trg.words[trg_indices[i]])
    else:
        for i in random_inds:
            print(src.words[src_indices[i]], " ", trg.words[trg_indices[i]])


def train(
        x, z, src_indices, trg_indices, tag_indices, opts,
        validate_fun=None, test_fun=None, args=None, sgd_iter=False):
    """
    Main training function.
    :param x:               source embedding matrix
    :param z:               target embedding matrix
    :param src_indices:
    :param trg_indices:
    :param tag_indices:
    :param opts:
    :param test_fun:
    :return:                transformed matrices x and z
    """
    if opts.log:
        log = open(
            opts.log, mode='w', encoding=opts.encoding,
            errors='surrogateescape')

    if opts.model_type == ModelType.NOT_SELF_LEARN:
        return get_mapped_vectors(x, z, src_indices, trg_indices, opts)

    print(opts.xp)

    # tag2xw is a list which holds x embeddings
    # tranformed differently depending on the tag - used in lat_morph
    x, z, tag2xw, matching = setup_for_training(opts, x, z)

    prev_objective = objective = -100
    it = 1
    t = time.time()
    W = None  # transformation matrix for x  -- x.dot(W)

    if opts.verbose:
        print("Entering the training loop:")
    while it == 1 or objective - prev_objective >= opts.threshold:
        # 1. Update the embedding mapping
        if opts.model_type == ModelType.LAT_MORPH:
            tag2xw = get_matrix_transformations(
                x, z, src_indices, trg_indices, tag_indices, tag2xw, opts)
        else:
            xw, zw, W = get_mapped_vectors(
                x, z, src_indices, trg_indices, opts)

        if opts.verbose:
            print_dict_sample(opts, src_indices, trg_indices)

        # 2. Learn the new dictionary
        if opts.model_type == ModelType.LAT_MORPH:
            # use a different mapping for each tag
            # this model uses x transformations in tag2xw so doesn't need xw
            new_objective, src_indices, trg_indices, tag_indices = \
                matching.get_new_dictionary(None, z, tag2xw)
        else:
            new_objective, src_indices, trg_indices, _ =\
                matching.get_new_dictionary(xw, zw)
        prev_objective = objective
        objective = new_objective

        val_res_str, short_val_res_str = '', ''
        if validate_fun is not None:
            val_res_str, short_val_res_str = validate_fun(xw, zw)

        # Logging
        duration = time.time() - t
        if opts.verbose:
            print(file=sys.stderr)
            print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
            print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
            print(val_res_str, file=sys.stderr)
            sys.stderr.flush()
        if opts.log is not None:
            print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(
                it, 100 * objective, short_val_res_str, duration), file=log)
            log.flush()

        t = time.time()
        it += 1

    if opts.model_type == ModelType.LAT_MORPH:
        # we have multiple xw matrices - one for each tag, the following function
        # picks the optimal transformation for each word
        xw = get_xw(xw, opts.src_data.tags_ids, src_indices, tag_indices, tag2xw)
    if test_fun is not None:
        test_fun(xw, zw)
    if opts.verbose:
        print("Exiting the training loop...")
    return xw, zw, W
