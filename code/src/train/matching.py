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

import math
import sys
import time

import numpy as np
from lap import lapmod
from src.train.options import ModelType
from src.utils.embeddings import length_normalize


class Matching:

    def __init__(self, n_rows, tag_count, opts):
        self.opts = opts

        #  all LAPMOD data has to be on cpu
        self.src_indices = np.concatenate(
            [np.arange(n_rows)] * opts.n_repeats, 0)

        ii = np.empty((n_rows * opts.n_repeats + 1,), dtype=int)
        ii[0] = 0
        # if each src id should be matched to trg id,
        # then we need to double the source indices
        for i in range(1, n_rows * opts.n_repeats + 1):
            ii[i] = ii[i - 1] + opts.n_similar

        self.ii = ii
        # cc holds the elements of the assignement cost matrix
        self.cc = np.empty(n_rows * opts.n_similar)
        # kk holds the column indices. Must be sorted within one row.
        self.kk = np.empty(n_rows * opts.n_similar, dtype=int)

        if opts.model_type == ModelType.LAT_MORPH:
            # the similarities array is reused in different matching
            self.sims = [
                opts.xp.empty((opts.chunk_size, n_rows),
                              dtype=opts.dtype) for _ in range(tag_count)]
        self.n_rows = n_rows

    def retrieve_ids(self, xw, zw, tag2xw):
        """
        Fills the kk list with target indices that should be considered during
        matching and the cc list with the cost values.
        :param xw:
        :param zw:
        :param tag_masks:
        :param tag2xw:
        :return:
        """
        start_time = time.time()
        n_rows = self.n_rows
        opts = self.opts

        for i in range(0, n_rows, opts.chunk_size):
            j = min(n_rows, i + opts.chunk_size)

            if opts.model_type == ModelType.LAT_MORPH:
                for z in range(len(self.sims)):
                    # -2 just to be lower than the mininum cosine value
                    self.sims[z].fill(-2)
                ijmasks = opts.src_tag_masks[(i, j)]
                for tag, txw in enumerate(tag2xw):
                    tag_mask = ijmasks[tag]
                    if len(tag_mask) == 0:
                        continue
                    res = txw[i:j][tag_mask].dot(zw[:n_rows].T)
                    # only compute the scores for the relevant tags
                    self.sims[tag][tag_mask] = res
                # for each source-target pair get the maximum possible result
                sim = opts.xp.maximum.reduce(self.sims)
                # this needs to be done in case the last
                # chunk has a different size
                sim = sim[:(j - i)]
            else:
                sim = xw[i:j].dot(zw[:n_rows].T)

                not_norm = False
                for o in range(len(sim)):
                    for p in range(len(sim[o])):
                        if sim[o, p] < -1:
                            not_norm = True
                            break
                if not_norm:
                    print("NOT NORMALISED XW")
                    tmp_xw = length_normalize(xw)
                    sim = tmp_xw[i:j].dot(zw[:n_rows].T)

            if opts.match_constr:
                ginds = opts.good_inds[(i,j)]
                mask = opts.xp.full((j-i, n_rows), -2, dtype=opts.dtype)
                xs, ys = ginds
                mask[xs, ys] = 0
                sim = sim + mask

            # get indices of n largest elements
            trg_indices = opts.xp.argpartition(sim, -opts.n_similar)[:,-opts.n_similar:]
            if opts.xp != np:
                # all indices must be on cpu for lapmod
                trg_indices = opts.xp.asnumpy(trg_indices)
            trg_indices.sort()
            trg_indices = trg_indices.flatten()
            row_indices = np.asarray([[z] * opts.n_similar
                                    for z in range(j - i)], dtype=int).flatten()

            sim_scores = sim[row_indices, trg_indices]
            if opts.xp != np:
                # moving simscores to cpu for lapmod
                sim_scores = opts.xp.asnumpy(sim_scores)

            self.cc[i * opts.n_similar:j * opts.n_similar] =\
                np.subtract(1, sim_scores)
            self.kk[i * opts.n_similar:j * opts.n_similar] = trg_indices
            if i % 10000 == 0 and i > 0:
                print(f'Processed {i} rows.')
        print(f'Retrieval of ids took {time.time() - start_time}s.')

    def run_lapmod(self):
        """
        Runs the lapmod algorithm and returns a new dictionary
        :return:
        """
        n_rows = self.n_rows
        cc = self.cc
        kk = self.kk
        opts = self.opts

        if opts.n_repeats > 1:
            # duplicate costs and target indices
            for i in range(1, opts.n_repeats):
                new_cc = np.concatenate([new_cc, self.cc], axis=0)
                if opts.asym == '1:2':
                    # for 1:2, we don't duplicate the target indices
                    new_kk = np.concatenate([new_kk, self.kk], axis=0)
                else:
                    # update target indices so that they refer to new columns
                    new_kk = np.concatenate([new_kk, self.kk + n_rows * i],
                                            axis=0)
            cc = new_cc
            kk = new_kk

        unique_targets = set(kk)
        print(f"Target words in matching: {len(unique_targets)}")

        cost, trg_indices, _ = lapmod(n_rows * opts.n_repeats, cc, self.ii,
                                      kk)
        if opts.no_random_matching:
            wrong_inds = []
            for i, trgind in enumerate(trg_indices):
                krow = self.ii[i]
                ks = kk[krow:krow + opts.n_similar]
                if trgind not in ks:
                    wrong_inds.append(i)

            print(f"Removing {len(wrong_inds)} incorrectly matched words...")
            trg_indices = np.delete(trg_indices, wrong_inds)
            new_src_indices = np.delete(self.src_indices, wrong_inds)
        else:
            new_src_indices = self.src_indices

        print(f"Items in matching: {len(new_src_indices)}")
        return new_src_indices, trg_indices

    def get_new_objective(self, xw, zw, src_indices, trg_indices, tag2xw):
        opts = self.opts
        best_sim_forward = np.full(xw.shape[0], -100, dtype=opts.dtype)
        a = 0

        tag_indices = np.empty_like(src_indices)
        for i in range(len(src_indices)):
            src_idx = src_indices[i]
            trg_idx = trg_indices[i]

            # we do this if opts.n_repeats > 0 to assign the target
            # indices in the cost matrix to the correct idx
            while trg_idx >= self.n_rows:
                # if we repeat, we have indices that are > rows_x
                trg_idx -= self.n_rows
                trg_indices[i] = trg_idx

            if opts.model_type == ModelType.LAT_MORPH:
                # retrieve the suitable tag masks to identify
                # the tags selected at matching
                while src_idx >= a + opts.chunk_size:
                    a += opts.chunk_size
                j = min(self.n_rows, a + opts.chunk_size)
                # +a because for each chunk the indices start from 0
                current_masks = opts.src_tag_masks[(a, j)] + a

                # apply masks, and then select top
                src_vecs = tag2xw[:, src_idx]
                dots = src_vecs.dot(zw[trg_idx].T)
                for tag, txw in enumerate(tag2xw):
                    tag_mask = current_masks[tag]
                    if src_idx not in tag_mask:
                        dots[tag] = -math.inf

                # tranform suitably, according to the tag chosen during matching
                best_sim = dots.max()
                tag_indices[i] = dots.argmax()
                best_sim_forward[src_idx] =\
                    max(best_sim_forward[src_idx], best_sim)
            else:
                best_sim = xw[src_idx].dot(zw[trg_idx].T)
                best_sim_forward[src_idx] =\
                    max(best_sim_forward[src_idx], best_sim)
        ok_inds = np.where(best_sim_forward != -100)
        best_sim_forward = np.take(best_sim_forward, ok_inds)
        return np.mean(best_sim_forward).tolist(), tag_indices

    def get_new_dictionary(self, xw, zw, tag2xw=None):
        """"
        Gets the most likely matching between two sets of embeddings.
        :param x:
        :param z:
        :return:
        """
        start = time.time()
        self.retrieve_ids(xw, zw, tag2xw)
        new_src_indices, new_trg_indices = self.run_lapmod()

        new_objective, new_tag_indices =\
            self.get_new_objective(
                xw, zw, new_src_indices, new_trg_indices, tag2xw)

        print(f'Whole matching took {time.time() - start}s.')
        return new_objective, new_src_indices, new_trg_indices, new_tag_indices


class Alignment:
    # NOTE: this might not work on a GPU

    def __init__(self, max_dimx, max_dimz, opts):
        self.maxx = max_dimx
        self.maxz = max_dimz
        self.opts = opts

    def get_new_dictionary(self, xw, zw):
        """
        Inducing a new training dictionary as described in Artexte et al. (2017)
        :param x:
        :param z:
        :return:
        """
        opts = self.opts
        best_sim_forward = np.full(xw.shape[0], -100, dtype=opts.dtype)
        src_indices_forward = np.arange(xw.shape[0])
        trg_indices_forward = np.zeros(xw.shape[0], dtype=int)
        best_sim_backward = np.full(zw.shape[0], -100, dtype=opts.dtype)
        src_indices_backward = np.zeros(zw.shape[0], dtype=int)
        trg_indices_backward = np.arange(zw.shape[0])

        # for efficiency and due to space reasons, look at sub-matrices of
        # size (MAX_DIM_X x MAX_DIM_Z)
        for i in range(0, xw.shape[0], self.maxx):
            j = min(xw.shape[0], i + self.maxx)
            if opts.verbose:
                print(f'src ids: {i}-{j}', file=sys.stderr)
            for k in range(0, zw.shape[0], self.maxz):
                l = min(zw.shape[0], k + self.maxz)
                sim = xw[i:j].dot(zw[k:l].T)

                if opts.match_constr:
                    ginds = opts.good_inds[(i, j, k, l)]
                    mask = opts.xp.full((j - i, l - k), -2, dtype=opts.dtype)
                    xs, ys = ginds
                    mask[xs, ys] = 0
                    sim = sim + mask

                if opts.direction in ('forward', 'union'):
                    ind = sim.argmax(
                        axis=1)  # trg indices with max sim for each src id (MAX_DIM_X)
                    val = sim[opts.xp.arange(sim.shape[0]), ind]  # the max sim value for each src id (MAX_DIM_X)
                    ind += k  # add the current position to get the global trg indices
                    mask = (val > best_sim_forward[i:j])  #  mask the values if the current value < best sim seen so far for the current src ids
                    best_sim_forward[i:j][mask] = val[
                        mask]  #  update the best sim values
                    trg_indices_forward[i:j][mask] = ind[
                        mask]  #  update the matched trg indices for the src ids
                if opts.direction in ('backward', 'union'):
                    ind = sim.argmax(axis=0)
                    val = sim[ind, opts.xp.arange(sim.shape[1])]
                    ind += i
                    mask = (val > best_sim_backward[k:l])
                    best_sim_backward[k:l][mask] = val[mask]
                    src_indices_backward[k:l][mask] = ind[mask]
        if opts.direction == 'forward':
            src_indices = src_indices_forward
            trg_indices = trg_indices_forward
        elif opts.direction == 'backward':
            src_indices = src_indices_backward
            trg_indices = trg_indices_backward
        elif opts.direction == 'union':
            src_indices = np.concatenate(
                (src_indices_forward, src_indices_backward))
            trg_indices = np.concatenate(
                (trg_indices_forward, trg_indices_backward))

        # Objective function evaluation
        if opts.direction == 'forward':
            new_objective = np.mean(best_sim_forward).tolist()
        elif opts.direction == 'backward':
            new_objective = np.mean(best_sim_backward).tolist()
        elif opts.direction == 'union':
            new_objective = (np.mean(best_sim_forward) + np.mean(
                best_sim_backward)).tolist() / 2
        return new_objective, src_indices, trg_indices, None
