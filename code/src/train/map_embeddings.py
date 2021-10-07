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

import argparse

import src.utils.embeddings as embeddings
from src.train.training import *
from src.train.options import *
from src.utils.dictionary_utils import *
from src.utils.commons import *
from src.utils.tag_utils import *
from src.data_prep.oov_embs import get_oov_vecs
from src.train.grad_training import grad_train

xp = None


def main():
    global xp

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map the source embeddings into the target embedding space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
  #  parser.add_argument('src_output', help='the output source embeddings')
  #  parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('matrix_output', help='the output of the transformation matrix')
    parser.add_argument('--model_out', default='test/model', help='the output of the trained model (sgd)')


    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp64', help='the floating-point precision (defaults to fp64)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--num_words', default=200000, type=int, help='whether to use only the top n most frequent words for learning embeddings')

    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments (EMNLP 2016)')
    mapping_group.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the training dictionary file (defaults to stdin)')
    mapping_group.add_argument('--test_dict', help='the test dictionary file')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=['unit', 'center'], help='the normalization actions to perform in order')

    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('self-learning arguments', 'Optional arguments for self-learning (ACL 2017)')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='forward', help='the direction for dictionary induction (defaults to forward)')
    self_learning_group.add_argument('--numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    self_learning_group.add_argument('--identical', action='store_true', help='use identical words as training dictionary')

    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    self_learning_group.add_argument('--lat_var', action='store_true', help='use the latent-variable model')

    self_learning_group.add_argument('--match_constr', action='store_true', help='Impose additional, tag related constraints on the matching (can be used with --lat-var setting).')
    self_learning_group.add_argument('--lat_morph', action='store_true', help='use the morphologically aware latent-variable model')
    self_learning_group.add_argument('--morph_feats', default=None, help='which morphological features should be considered')
    self_learning_group.add_argument('--umsource', help='')
    self_learning_group.add_argument('--morph_vocab', action='store_true', help='reduce the vocabulary to words for which the morphological tag is known ')
    self_learning_group.add_argument('--single_tag', action='store_true', help='')
    self_learning_group.add_argument('--no_random_matching', default=True, help='')
    self_learning_group.add_argument('--src_lang', help='')
    self_learning_group.add_argument('--trg_lang', help='')

    sgd_group = parser.add_argument_group('sgd arguments', '')
    sgd_group.add_argument('--sgd', action='store_true')
    sgd_group.add_argument('--sgd_iter', action='store_true')


    sgd_group.add_argument('--bs', default=24, type=int, help='training batch size')
    sgd_group.add_argument('--epochs', default=25, type=int, help='maximum training epochs')
    sgd_group.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adadelta', 'Adam'])
    sgd_group.add_argument('--lr', default=0.05, type=float, help='learning rate')
    sgd_group.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    sgd_group.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
    sgd_group.add_argument('--estop', default=1e-8, type=float, help='early stopping criterion')
    sgd_group.add_argument('--cooldown', default=0, type=int, help='cooldown of `ReduceLROnPlateau`')
    sgd_group.add_argument('--reg_alpha', default=15, type=float, help='the weight of the regularization in the loss')

    #parser.add_argument('--max_norm', default=0, type=float, help='gradient clipping max norm')

    self_learning_group.add_argument('--filter_dict', default=[], type=lambda x: [i for i in x.split("|")])
    self_learning_group.add_argument('--balance_pos', action='store_true')
    self_learning_group.add_argument('--filter_train_embs', default=[], type=lambda x: [i for i in x.split("|")]) # this is just for unimorph training - when we know the tag

  #  self_learning_group.add_argument('--tag_underspec', action='store_true')
    # note that if the oov vecs are added normalize argument should probably also be given
   # self_learning_group.add_argument('--src_oov_vecs', default=None, help='if provided the src oov embeddings will be read from that file')
   # self_learning_group.add_argument('--trg_oov_vecs', default=None, help='if provided the trg oov embeddings will be read from that file')

    mapping_group.add_argument('--reverse_dict', action='store_true')

    self_learning_group.add_argument('--n_similar', type=int, default=3, help='# of most similar trg indices used for sparsifying in latent-variable models')
    self_learning_group.add_argument('--chunk_size', default=1000, type=int, help='default size of matrix chunks for latent-variable model')
    self_learning_group.add_argument('--n_repeats', default=1, type=int, help='repeats embeddings to get 2:2, 3:3, etc. alignment in latent-variable model')
    self_learning_group.add_argument('--asym', default='1:1', help='specify 1:2 or 2:1 for assymmetric matching in latent-variable model')
    self_learning_group.add_argument('--rank_constr', type=int, help='match only the top n most frequent words during alignment in latent-variable model')
    advanced_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments (AAAI 2018)')
    advanced_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    advanced_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    advanced_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    advanced_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    advanced_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    advanced_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    args = parser.parse_args()

    validation_fun, test_fun = None, None

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)
    if args.verbose:
        print("Info: arguments\n\t" + "\n\t".join(
            ["{}: {}".format(a, v) for a, v in vars(args).items()]),
              file=sys.stderr)

    # global settings
    dtype = set_precision(args.precision)
    xp = select_matrix_library(args.cuda)
    opts = Options(args, xp, dtype)
    src, trg = opts.src_data, opts.trg_data

    # Read input embeddings
    src.words, x = embeddings.read(args.src_input, args.encoding, dtype) #, threshold=100000)
    trg.words, z = embeddings.read(args.trg_input, args.encoding, dtype) #, threshold=100000)

    print(f"Retrieved embeddings, Size of source vocabulary:"
          f"{len(src.words)}, of target vocabulary: {len(trg.words)}")

    x = embeddings.normalize(x, args.normalize)
    z = embeddings.normalize(z, args.normalize)
    x, z = xp.asarray(x), xp.asarray(z)

    # Build tag dictionary, retrieve tags for words
    if args.lat_morph or args.morph_vocab or args.match_constr:
        x, z = retrieve_tags(args, opts, x, z)

    # fix the shape, restrict to top num_words (if this arg is given),
    # filter them is the filter_train_embs argument is given
    x, z = process_embeddings(args, opts, x, z)

    # word processing, tags retrieval is completed
    src.retrieve_word2ind()
    trg.retrieve_word2ind()

    print(f"Processed embeddings, Size of source vocabulary:"
          f"{len(src.words)}, of target vocabulary: {len(trg.words)}")

    # Building dictionaries
    src_indices, trg_indices, tag_indices = build_train_dict(args, opts)
    print(len(src_indices))

    if len(src_indices) == 0:
        print("The training dictionary is of size 0.")
        return
    else:
        print("Dictionary size: ", len(trg_indices))

    if args.validation:
        validation_dict, validation_coverage = build_val_dict(args, opts)
        validation_fun = lambda x, z: validate(
            validation_dict, validation_coverage, x, z)
    if args.test_dict:
        test_dict, test_coverage = build_test_dict(args, opts)
        test_fun = lambda x, z: test(test_dict, test_coverage, x, z)
    print("Retrieved training and test dicitonaries")

    if args.sgd_iter:
        train_fun = partial(train, args=args, sgd_iter=True)
    elif args.sgd:
        train_fun = partial(grad_train, args=args)
    else:
        train_fun = train
    xw, zw, W = train_fun(
        x, z, src_indices, trg_indices, tag_indices,
        opts, validation_fun, test_fun)

    # # Save mapped embeddings
    # with open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape') as srcfile:
    #     embeddings.write(src.words, xw, srcfile)
    #
    # if args.trg_output:
    #     with open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape') as trgfile:
    #         embeddings.write(trg.words, zw, trgfile)

    with open(
            args.matrix_output, mode='w', encoding=args.encoding,
            errors='surrogateescape') as matfile:
        for r in range(0, len(W)):
            matfile.write(' '.join([str(x) for x in W[r]]))
            matfile.write("\n")


#############################
#       SETUP FUNCTIONS     #
#############################

def add_oov(
        src_words, trg_words, src_oov_vecs,
        trg_oov_vecs, x, z, encoding, dtype):
    try:
        src_oov_words, oov_x = embeddings.read(src_oov_vecs, encoding, dtype)
        trg_oov_words, oov_z = embeddings.read(trg_oov_vecs, encoding, dtype)
    except:
        src_oov_words = oov_x = trg_oov_words = oov_z = None

    if oov_x is not None and oov_z is not None:
        src_words += src_oov_words
        x = np.concatenate((x, oov_x), axis=0)
        trg_words += trg_oov_words
        z = np.concatenate((z, oov_z), axis=0)
    return src_words, trg_words, src_oov_words, trg_oov_words, x, z


def tags_to_ids(tags, tag2id):
    tag_ids = []
    for tag_list in tags:
        id_list = [tag2id[tag] if tag in tag2id else -1 for tag in tag_list]
        tag_ids.append(id_list)
    return tag_ids


def process_embeddings(args, opts, x, z):
    src, trg = opts.src_data, opts.trg_data
    nwords = args.num_words

    if nwords and nwords != 0:
        assert nwords > 0
        print(
            f'Restricting source and target words to top {nwords} '
            f'words...', file=sys.stderr)
        src.words = src.words[:nwords]
        trg.words = trg.words[:nwords]
        x = x[:nwords]
        z = z[:nwords]
        if src.tags and trg.tags:
            src.tags = src.tags[:nwords]
            trg.tags = trg.tags[:nwords]

    if args.morph_vocab and args.filter_train_embs:
        x = filter_embs_based_on_feats(
            x, InputType.SRC, args.filter_train_embs, opts)
        z = filter_embs_based_on_feats(
            z, InputType.TRG, args.filter_train_embs, opts)

    return x, z


def get_tags_wrapper(words, input_type, opts):
    ldata = opts.get_data(input_type)
    return get_tags(words, ldata.um, opts.morph_feats,
                    opts.tag2ind, opts.single_tag, ldata.lang)


def retrieve_tags(args, opts, x, z):
    """
    Retrieves tags for the words and if args.morph_vocab==True filters all
    words in both languages based on whether they appear in unimorph. The tags
    are saved in the opts object and so is the tag2ind dictionary creates
    in the process. The vocabulary (words) for each language stored in opts
    is updated if args.morph_vocab==True.
    :param args:
    :param opts:
    :param x:
    :param z:
    :return: x and z (unchanged if args.morph_vocab==False)
    """
    src, trg = opts.src_data, opts.trg_data

    _, src.um, _ = read_um(get_um_path(
        args.umsource, args.src_lang), lang=args.src_lang)
    _, trg.um, _ = read_um(get_um_path(
        args.umsource, args.trg_lang), lang=args.trg_lang)
    src.tags = get_tags_wrapper(src.words, InputType.SRC, opts)
    trg.tags = get_tags_wrapper(trg.words, InputType.TRG, opts)

    opts.tag2ind = get_tag2ind_from_vocabulary(opts)
    print("Retrieved index mapping for tags.")

    if args.morph_vocab:  # filter the embeddings - keep only words in unimorph
        x = filter_unknown_morph(x, InputType.SRC, opts)
        z = filter_unknown_morph(z, InputType.TRG, opts)
        print(f"Filtered embeddings. Size of source vocabulary:"
              f"{len(src.words)}, of target vocabulary: {len(trg.words)}")

    src.tag_ids = tags_to_ids(src.tags, opts.tag2ind)
    trg.tag_ids = tags_to_ids(trg.tags, opts.tag2ind)
    print(f"Words with unknown tags (src): {src.tag_ids.count(-1)} out of {len(src.words)}")
    print(f"Words with unknown tags (trg): {trg.tag_ids.count(-1)} out of {len(trg.words)}")
    print(f"Tag vocabulary: {opts.tag2ind}")
    return x, z


def filter_unknown_morph(embs, input_type, opts):
    tag_cond = lambda tag: tag in opts.tag2ind
    return filter_embs(embs, input_type, tag_cond, opts)


def filter_embs_based_on_feats(embs, input_type, feats, opts):
    tag_cond = lambda tag: any([all([y.lower() in tag.lower() 
                                for y in x.split("+")]) for x in feats])
    return filter_embs(embs, input_type, tag_cond, opts)


def filter_embs(embs, input_type, tag_condition, opts):
    ldata = opts.get_data(input_type)
    tags = ldata.tags

    new_words, new_embs, new_tags = [], [], []
    for i, tag_list in enumerate(tags):
        new_tag_list = []
        for tag in tag_list:
            if tag_condition(tag):
                new_tag_list.append(tag)

        if new_tag_list:
            new_words.append(ldata.words[i])
            new_embs.append(embs[i])
            new_tags.append(new_tag_list)
    ldata.words = new_words
    ldata.tags = new_tags
    ldata.tag_ids = tags_to_ids(ldata.tags, opts.tag2ind)
    return xp.array(new_embs)


if __name__ == '__main__':
    main()
