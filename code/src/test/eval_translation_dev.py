# Copyright (C) 2017-2018  Mikel Artetxe <artetxem@gmail.com>
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
# Contributor(s):
# 2019 Paula Czanowska <pjc211@cam.ac.uk>

import argparse
from src.utils.cupy_utils import *
from src.utils.dictionary_utils import *
from functools import partial
import pickle
from src.test.options import Options

from src.test.helpers import *
from src.test.trans_constraints import *
from src.test.trans_engine import *
import sys
import logging

BATCH_SIZE = 1000
xp = dtype = None
THRESHOLD = 0
BIG_CACHE = {}


def main():
    global xp, dtype

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('--src_embeddings', help='the source language embeddings')
    parser.add_argument('--trg_embeddings', help='the target language embeddings')
    parser.add_argument('--original_trg_vecs', default=None)  # this is needed to determine what transformations were used
    parser.add_argument('--original_src_vecs', default=None)  # this is needed to determine what transformations were used
    parser.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[])  # should be the same a used in training
    parser.add_argument('--matrix_source', default=None)
    parser.add_argument('--full_eval', action='store_true')
    parser.add_argument('--map_eval', action='store_true')


    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp64', help='the floating-point precision (defaults to fp64)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')

    parser.add_argument('--umsource', help='')
    parser.add_argument('--src_lang')
    parser.add_argument('--trg_lang')
    parser.add_argument('--test_info')
    parser.add_argument('--out_dir')
    parser.add_argument('--filter_dict', default=[], type=lambda x: [i for i in x.split("|")])
    parser.add_argument('--reverse_dict', action='store_true')

    # These are necessary to add OOV words into the evaluation
    parser.add_argument('--trg_oov_vecs', default=None)
    parser.add_argument('--src_oov_vecs', default=None)

    parser.add_argument('--read_pickle', action='store_true')
    parser.add_argument('--reinflection_models_dir', default=None)
    parser.add_argument('--analysis_models_dir', default=None)
    parser.add_argument('--min_distance', default=False)
    parser.add_argument('--lemma_filter', default=None)
    parser.add_argument('--oov_lemma_filter', default=None)
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--hybrid2', action='store_true')
    parser.add_argument('--hybrid3', action='store_true')

    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    args.reinflection_model = get_best_model_file(args.reinflection_models_dir)
    args.analysis_model = get_best_model_file(
        args.analysis_models_dir) if args.analysis_models_dir else None
    print(f"\nReinflection model: {args.reinflection_model}")
    print(f"Analysis model: {args.analysis_model}")

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            logging.info('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()

    else:
        xp = np
    xp.random.seed(args.seed)

    logging.info(f"\nEmbedding threshold:{THRESHOLD}, Full eval: {args.full_eval}")
    print(f"\nEmbedding threshold:{THRESHOLD}, Full eval: {args.full_eval}")

    logging.info("Retrieving embeddings...")
    if args.src_embeddings and args.trg_embeddings:
        src_words, trg_words, src_oov_words, trg_oov_words, x, z, oov_x, oov_z =\
            process_trained_embeddings(args, THRESHOLD, dtype)

        logging.info("In vocab src:", len(src_words))
        logging.info("OOV src:", len(src_oov_words))
        logging.info("read embs ok")
    else:
        matrix = read_matrix(args.matrix_source, args.encoding)
        if not args.read_pickle:
            src_words, trg_words, src_oov_words, trg_oov_words, x, z, oov_x, oov_z =\
                get_x_z_with_oovs(
                    args.original_src_vecs, args.original_trg_vecs, matrix,
                    args.src_oov_vecs, args.trg_oov_vecs,
                    args.normalize, args.encoding, dtype, THRESHOLD)
        else:
            src_words, trg_words, src_oov_words, trg_oov_words, x, z, oov_x, oov_z =\
                [], [], [], [], None, None, None, None
    x = xp.asarray(x)
    z = xp.asarray(z)

    # Length normalize embeddings so their dot product effectively computes
    # the cosine similarity
    if not args.src_embeddings and not args.dot and \
            "unit" not in args.normalize:
        x = embeddings.length_normalize(x)
        z = embeddings.length_normalize(z)
        if oov_x is not None and oov_z is not None:
            oov_x = embeddings.length_normalize(oov_x)
            oov_z = embeddings.length_normalize(oov_z)
    logging.info("Got embeddings.")

    options = Options(args, xp, dtype)
    srcd, trgd = options.src_data, options.trg_data

    trgd.initialize(trg_words, trg_oov_words, z, oov_z)
    srcd.initialize(src_words, src_oov_words, x, oov_x)

    # Read dictionary and compute coverage (this also alters the vocabulary in
    # src to only include words in the dictionary)
    big_dict = get_fine_grained_eval_dict(
        args, args.dictionary, options)

    # fills the options object with tags/tagids
    retrieve_tags(options, big_dict)

    get_all_translations(args, options, big_dict)
    logging.info("got all trans.")

    if args.full_eval and not args.map_eval:
        logging.info("Complete eval...")
        perform_total_eval(args, big_dict, options)
    else:
        logging.info("Simple evaluation....")
        full_eval(args, big_dict, options, no_morph_splits=True)
    logging.info("Evaluation completed")


def perform_total_eval(args, big_dict, options):
    def picklefun(fname, res):
        pickle_file = open(
            f"{args.out_dir}/{fname}_{args.test_info}.txt", "wb")
        pickle.dump(res, pickle_file)
        pickle_file.close()

    if not big_dict.morph_dict:
        name_add = "old_dir_"
    else:
        name_add = ""

    size2morph2res = eval_based_on_lemma_steps(args, big_dict, options)
    picklefun(f"pickled_{name_add}morph_steps", size2morph2res)

    # print("\n---------------------- FREQ STEPS")
    # freq_results = eval_based_on_freq_steps(args, big_dict, options)
    # picklefun(f"pickled_{name_add}freq_step_res", freq_results)

    # print("\n---------------------- FREQ BINS")
    # freq_results_bins = eval_based_on_freq_bins(args, big_dict, options)
    # picklefun(f"pickled_{name_add}freq_bins_res", freq_results_bins)
    # #
    print("\n---------------------- MORPH")
    morph2results = full_eval(args, big_dict, options)
    picklefun(f"pickled_{name_add}morph_res", morph2results)


def get_all_translations(args, opts, big_dict):
    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix
    dict_tests = [
            (
                "In vocab", 0,
                partial(big_dict.get_dictionary, dtype=DictTypes.IN_VOCAB),
                x, z
            ),
            (
                "All", 0,
                partial(big_dict.get_dictionary, dtype=DictTypes.ALL),
                full_x, full_z)
        ]

    # get full results to reuse later...
    perform_eval(
        args, big_dict, dict_tests, opts, verbose=False,
        only_normal=not args.full_eval, latex=False, retrieve_full=True)


def eval_based_on_freq_steps(args, big_dict, opts):
    d = big_dict
    dict_tests = []

    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix

    sizes_k = [0, 10, 50, 100, 200, 300, 400, 500, 600]
    prev_total_len = 0

    threshold = 0
    for i in range(1, len(sizes_k)):
        k = sizes_k[i]
        prev_k = sizes_k[i-1]

        threshold = k * 1000
        prev_threshold = prev_k * 1000

        if threshold > len(z):
            threshold = prev_threshold
            break

        fun = partial(
            d.get_dictionary,
            dtype=DictTypes.IN_VOCAB,
            chunks_info=[prev_threshold, threshold, 0, len(z)])
        fun2 = partial(
            d.get_dictionary,
            dtype=DictTypes.IN_VOCAB,
            chunks_info=[prev_threshold, threshold, prev_threshold, threshold],
            allow_one_more_freq=True)
        src_nb, src2trg_nb, coverage_nb = fun()
        if len(src_nb) == prev_total_len:
            continue
        else:
            prev_total_len = len(src_nb)

        dict_tests.append(
            (f"Top {k}k (src filter) (full search)", 0, fun, x, z))
        dict_tests.append((f"Top {k}k (binned) (full search)", 0, fun2, x, z))

    fun2 = partial(
        d.get_dictionary,
        dtype=DictTypes.IN_VOCAB,
        chunks_info=[threshold, len(x), threshold, len(z)],
        allow_one_more_freq=True)

    dict_tests += [
        ("All (src filter) (full search)", 0,
            partial(d.get_dictionary, dtype=DictTypes.IN_VOCAB,
                    chunks_info=[threshold, len(x), 0, len(z)]),
            x, z),
        ("All (binned) (full search)", 0,
            fun2, x, z),
        ("All+OOVs (src filter) (full search)", 0,
            partial(d.get_dictionary, dtype=DictTypes.SRC_OOVS),
            full_x, full_z ),
        ("All+OOVs (binned) (full search)", 0,
            partial(d.get_dictionary, dtype=DictTypes.EITHER_OOVS),
            full_x, full_z),
    ]

    return perform_eval(args, big_dict, dict_tests, opts, only_normal=False)


def get_the_nums_in_freq_bins(args, big_dict, opts):
    d = big_dict

    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix

    freq_info = {}
    threshold = 0
    sizes_k = [0, 10, 50, 100, 200, 300, 400, 500, 600]
    for i in range(1, len(sizes_k)):
        k = sizes_k[i]
        prev_k = sizes_k[i-1]

        threshold = k * 1000
        prev_threshold = prev_k * 1000
        if threshold > len(z):
            threshold = prev_threshold
            break

        src_nb, src2trg_nb, coverage_nb = d.get_dictionary(
            dtype=DictTypes.IN_VOCAB,
            chunks_info=[0, threshold, 0, threshold])
        src_bin, src2trg_bin, coverage_bin = d.get_dictionary(
            dtype=DictTypes.IN_VOCAB,
            chunks_info=[prev_threshold, threshold, prev_threshold, threshold],
            allow_one_more_freq=True)
        freq_info[f"Top {k}k"] = len(src_nb)
        freq_info[f"{prev_k}k--{k}k"] = len(src_bin)


    src_all_bin, src2trg_all_bin, coverage_all_bin = d.get_dictionary(
        dtype=DictTypes.IN_VOCAB,
        chunks_info=[threshold, len(x), threshold, len(z)],
        allow_one_more_freq=True)
    src_all_nb, src2trg_all_nb, coverage_all_nb = d.get_dictionary(
        dtype=DictTypes.IN_VOCAB)
    freq_info[f"All"] = len(src_all_nb)
    freq_info[f"{threshold/1000}k--"] = len(src_all_bin)
    return freq_info


def eval_based_on_freq_bins(
        args, big_dict, opts, allow_one_more_freq=False,
        enforce_one_more_freq=False):
    """
    If the bin is too small (lower than a threshold) then merge it with
    neighbouring bins until its size reaches the threshold.
    :param args:
    :param big_dict:
    :param opts:
    :return:
    """
    d = big_dict
    dict_tests = []

    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix

    threshold=50

    bin_size = 50000
    smaller_embs = min(len(x), len(z))

    completed = set()
    for i in range(0, smaller_embs, bin_size):
        if i in completed:
            continue

        num = 1
        while num == 1 or (len(src) < threshold and j < smaller_embs):
            j = i + num * bin_size
            chunks_info = [i, j, i, j]
            fun = partial(d.get_dictionary, dtype=DictTypes.IN_VOCAB,
                          chunks_info=chunks_info,
                          allow_one_more_freq=allow_one_more_freq,
                          enforce_one_more_freq=enforce_one_more_freq)
            src, _, _ = fun()
            completed.add(i + (num - 1) * bin_size)
            num += 1
        if j > smaller_embs:
            name = f"{i//1000}k--"
        else:
            name = f"{i//1000}k--{j//1000}k" if i != 0 else f"0--{j//1000}k"
        dict_tests.append((f"{name} (full search)", 0, fun, x, z))

    if enforce_one_more_freq:
        dict_tests += [
            ("OOVs (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.ONE_SIDE_OOV),
                full_x, full_z),
            ("Nonce (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.ONE_SIDE_NONCE),
                full_x, full_z)
        ]
    elif allow_one_more_freq:
        dict_tests += [
            ("OOVs (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.EITHER_OOVS), 
                full_x, full_z),
            ("Nonce (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.EITHER_NONCES),
                full_x, full_z)
        ]
    else:
        dict_tests += [
            ("OOVs (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.BOTH_OOVS),
                full_x, full_z),
            ("Nonce (full search)", 0,
                partial(d.get_dictionary, dtype=DictTypes.BOTH_NONCES),
                full_x, full_z)
        ]

    return perform_eval(args, big_dict, dict_tests, opts, only_normal=False)


def eval_lemma_morph(args, big_dict, opts, prev_threshold, threshold):
    morph2results = {}

    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix

    for morph, d in big_dict.morph2dict.items():
        fun = partial(d.get_lemma_filtered_dictionary,
                       dtype=DictTypes.ALL,
                       chunks_info=[prev_threshold, threshold, 0,
                                    len(trgd.lemma2words)], opts=opts)

        fun2 = partial(d.get_lemma_filtered_dictionary,
                       dtype=DictTypes.IN_VOCAB,
                       chunks_info=[prev_threshold, threshold, 0,
                                    len(trgd.lemma2words)], opts=opts)

        dict_tests = [
            ("In vocab", 0, fun2, x, z),
            ("All", 0, fun, full_x, full_z)
        ]

        results = perform_eval(args, big_dict, dict_tests, opts,
                               verbose=False, only_normal=False)
        morph2results[morph] = results
    return morph2results


def find_best_splits(src_data):
    target_rank1 = 20000
    target_rank2 = 60000

    split1, split2 = None, None
    for i in range(len(src_data.ranked_lemmas)):
        ind = i * 1000
        ind = min(ind, len(src_data.ranks) - 1)
        if src_data.ranks[ind] > target_rank1:
            split1 = [0, i]
            print("low rank ", src_data.ranks[ind])
            break

    for j in range(split1[1] + 1, len(src_data.ranked_lemmas)):
        ind = j * 1000
        ind = min(ind, len(src_data.ranks) - 1)

        if src_data.ranks[ind] > target_rank2:
            split2 = [j, None]
            print("high rank ", src_data.ranks[ind])
            break

    return [split1, split2]


def eval_based_on_lemma_steps(args, big_dict, opts):
    srcd, trgd = opts.src_data, opts.trg_data
    sizes_k = find_best_splits(srcd)

    size2morph2res = {}
    for i in range(0, len(sizes_k)):
        prev_k, k = sizes_k[i]

        if k is None or k * 1000 > len(srcd.lemma2words):
            morph2res = eval_lemma_morph(
                args, big_dict, opts, prev_k * 1000, len(srcd.lemma2words))
            size2morph2res[f"{prev_k}k-- (binned) (full search)"] = morph2res
        else:
            threshold = k * 1000
            prev_threshold = prev_k * 1000
            morph2res = eval_lemma_morph(
                args, big_dict, opts, prev_threshold, threshold)
            size2morph2res[f"{prev_k}k--{k}k (binned) (full search)"] =\
                morph2res

    return size2morph2res


def full_eval(args, big_dict, opts, no_morph_splits=False):
    srcd, trgd = opts.src_data, opts.trg_data
    x, full_x = srcd.matrix, srcd.full_matrix
    z, full_z = trgd.matrix, trgd.full_matrix

    morph2results = {}

    dlist = [("ALL", big_dict)]
    if not no_morph_splits:
        dlist += big_dict.morph2dict.items()
    for morph, d in dlist:
        dict_tests = [
            ("In vocab", 0,
             partial(d.get_dictionary, dtype=DictTypes.IN_VOCAB), x, z),
            ("All", 0,
             partial(d.get_dictionary, dtype=DictTypes.ALL), full_x, full_z)
        ]

        results = perform_eval(
            args, big_dict, dict_tests, opts, verbose=True,
            only_normal=no_morph_splits, latex=False)  # (morph == "ALL"))

        morph2results[morph] = results
    return morph2results


def perform_eval(
        args, big_dict, dict_tests, opts, verbose=True, latex=False,
        only_normal=False, retrieve_full=False):
    out_file = f"{args.out_dir}/out_for_analysis"

    normal_res = evaluate_on_different_dict_views(
        args, big_dict, dict_tests, opts,
        out_file=out_file + "-norm.txt", retrieve_full=retrieve_full)
    if verbose:
        if normal_res:
            print("NORMAL:")
            print_res(normal_res, latex=latex)

    if big_dict.morph_dict and not only_normal and \
            args.reinflection_models_dir is None:
        um_normal_res = None

        lex_res = evaluate_on_different_dict_views(
            args, big_dict, dict_tests, opts, enforce_lex=True,
            out_file=out_file + "-lexeme.txt", retrieve_full=retrieve_full)
        if verbose:
            print("\nCONTROL FOR LEXEME:")
            print_res(lex_res, latex=latex)

        # morph_res = evaluate_on_different_dict_views(
        #     args, big_dict, dict_tests, opts,
        #     enforce_morph=True, out_file=out_file  + "-morph.txt",
        #     retrieve_full=retrieve_full)
        # if verbose:
        #    print("\nCONTROL FOR MORPHOLOGY:")
        #      print_res(morph_res, latex=latex)
        return {"NORMAL": normal_res, "UM_NORMAL": um_normal_res,
                    "LEX": lex_res} #, "MORPH": morph_res}
    else:
        return {"NORMAL": normal_res}


def evaluate_on_different_dict_views(
        args, big_dict, dict_tests, opts, enforce_morph=False,
        enforce_lex=False, only_unimorph=False, out_file=None,
        retrieve_full=False):
    N_SAMPLES = 5
    result = {}
    srcd, trgd = opts.src_data, opts.trg_data

    if only_unimorph:
        key = "UM_NORMAL"
    elif enforce_lex:
        key = "LEX"
    elif enforce_morph:
        key = "MORPH"
    else:
        key = "NORMAL"

    for test in dict_tests:
        name, sample_size, get_dict_fun, src_matrix, trg_matrix = test

        if src_matrix is None or trg_matrix is None:
            continue

        samples = N_SAMPLES if sample_size > 0 else 1
        accuracies = []
        total_coverage = 0
        srctrg2info = big_dict.srctrg2info
        src_tags, trg_tags = srcd.tag_ids, trgd.tag_ids

        empty_dictionary = False
        # if we get a random sample from a dictionary -
        # we do it multiple times and average
        for i in range(0, samples):
            src, _, coverage = get_dict_fun()
            _, src2trg, _ = big_dict.get_dictionary(dtype=DictTypes.ALL)
            src_to_trans = src
            if len(src) == 0:
                empty_dictionary = True
                continue

            # enforce lex works on unimorph only anyway
            if only_unimorph and not enforce_lex:
                src_new, trg_new = get_only_unimorph_data(
                    opts, src_matrix, trg_matrix)
                _, src_matrix, src_tags, src_old2new_map = src_new
                _, trg_matrix, trg_tags, trg_old2new_map = trg_new
                src_to_trans = [src_old2new_map[s] for s in src]

            if enforce_morph:
                good_inds = retrieve_good_inds(
                    src_to_trans, len(trg_matrix),
                    src_tags, trg_tags, BATCH_SIZE, opts)
            elif enforce_lex:
                good_inds = get_good_inds_lemma_based(
                    src_to_trans, src2trg, srctrg2info,
                    len(trg_matrix), BATCH_SIZE, opts)
            else:
                good_inds = None

            if (key, str(trg_matrix)) not in big_dict.full_translation:

                translations = get_translations(
                    args, opts, src_matrix, trg_matrix, src_to_trans,
                    src2trg, good_inds, srctrg2info, BATCH_SIZE, xp, dtype)
                if only_unimorph and not enforce_lex:
                    src_new2old_map = {v: k for k, v in src_old2new_map.items()}
                    trg_new2old_map = {v: k for k, v in trg_old2new_map.items()}
                    trans_tmp = {}
                    for s, t in translations.items():
                        trans_tmp[src_new2old_map[s]] = trg_new2old_map[t]
                    translations = trans_tmp

                # map evaluation only for normal, no reinfl models
                if key == "NORMAL" and args.map_eval and \
                        args.reinflection_models_dir is None:
                    get_map(args, opts, src_matrix, trg_matrix, src,
                            translations, src2trg, srctrg2info, BATCH_SIZE)
            else:
                translations = big_dict.full_translation[(key, str(trg_matrix))]

            accuracies.append(rate_translation(
                src, src2trg, translations, opts,
                unimorph_only=(only_unimorph or enforce_morph or enforce_lex),
                out_file=out_file))
            total_coverage += coverage

        if empty_dictionary:
            continue

        # get mean and std for each acc, macc, lacc
        accuracies = process_accuracies(accuracies)
        coverage = total_coverage / samples

        # note that morphcounts are really truly representative
        # if sample size is 0 (no sampling)
        morph_counts = big_dict.get_morph_counts(
            src, src2trg, opts.src_data.words, opts.trg_data.words)

        if (key, str(trg_matrix)) not in big_dict.full_translation: # and retrieve_full:
            big_dict.full_translation[(key, str(trg_matrix))] = translations
        result[(name, sample_size)] =\
            (*accuracies, len(src), coverage, morph_counts)
    return result


def get_translations(
        args, opts, x, z, src, src2trg, good_inds,
        srctrg2info, chunk_size, xp, dtype):
    if args.reinflection_model:
        if args.lemma_filter is None:
            print("WARNING. Reinflection model applied to not " +
                  "filtered embeddings!")

        if srctrg2info is not None:
            better_inds = get_good_inds_only_lemma(
                src, len(z), BATCH_SIZE, opts) if args.lemma_filter else None

            # if better inds is None then good_inds will be returned
            better_inds = merge_good_inds(better_inds, good_inds)

            return get_translations_via_lemma(
                args, opts, x, z, src, src2trg, better_inds,
                srctrg2info, chunk_size, xp, dtype)
        else:
            print("ERROR: srctrg2info is None:\n"
                  "unable to determine word's lemma and tag for"
                  "reinflection and no analyzer is provided.")
    else:
        return get_normal_translations(
            args, x, z, src, good_inds, chunk_size, xp, dtype)


if __name__ == '__main__':
    main()
