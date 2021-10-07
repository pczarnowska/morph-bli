import re
import sys
import collections
from src.utils.tag_utils import crop_tag, get_pos
import random
from functools import partial
import enum
from collections import Counter

########################
## EVALUATION       ####
########################


class DictTypes(enum.IntEnum):
    IN_VOCAB = 1
    SRC_OOVS = 2
    TRG_OOVS = 3
    BOTH_OOVS = 4
    EITHER_OOVS = 5
    ALL = 6
    SRC_NONCES = 7
    TRG_NONCES = 8
    BOTH_NONCES = 9
    EITHER_NONCES = 10
    ONE_SIDE_NONCE = 11
    ONE_SIDE_OOV = 12


class TDict:
    def __init__(self):
        self.dictionary_map = dict()
        for dt in DictTypes:
            self.dictionary_map[dt] = collections.defaultdict(set)
        self.srctrg2info = collections.defaultdict(list)

    def get_coverage(self, d):
        total_lines = 0
        for src, trg_list in d.items():
            total_lines += len(trg_list)
        return total_lines/len(self.srctrg2info)

    def check_ind(self, start, end, ind):
        return ind >= start and ind < end

    def check_lemma_id(
            self, start, end, data, word_ind, less_than_start=False):
        word = data.words[word_ind]
        lemmas = data.word2lemma[word]

        if not less_than_start:
            verdict = True in [
                self.check_ind(start, end, data.lemma2ranked_ind[lemma]) for
                lemma in lemmas]
        else:
            verdict = True in [
                data.lemma2lanked_ind[lemma] < start for
                lemma in lemmas]
        return verdict

    def get_morph_counts(self, srcs, src2trg, src_words, trg_words):
        morph_counts = collections.defaultdict(int)
        for src in srcs:
            srcw = src_words[src]
            morph_counts["all"] += 1

            for trg in src2trg[src]:
                trgw = trg_words[trg]
                for info in self.srctrg2info[(srcw, trgw)]:
                    morph, _, _ = info
                    morph_counts[morph] += 1
        return morph_counts

    def filter_dict(
            self, dictionary, chunks_info,
            allow_one_more_freq=False, enforce_one_more_freq=False):
        src_start, src_end, trg_start, trg_end = chunks_info
        new_map = collections.defaultdict(set)

        src_more_freq_count = 0
        trg_more_freq_count = 0

        for src_ind, trg_inds in dictionary.items():
            if self.check_ind(src_start, src_end, src_ind):
                for trg_ind in trg_inds:
                    if self.check_ind(trg_start, trg_end, trg_ind) and \
                            (not enforce_one_more_freq or trg_start == 0):
                        new_map[src_ind].add(trg_ind)
                    elif (allow_one_more_freq or enforce_one_more_freq) and \
                            trg_ind < trg_start:
                        trg_more_freq_count += 1
                        new_map[src_ind].add(trg_ind)

            elif (allow_one_more_freq or enforce_one_more_freq) and \
                    src_ind < src_start:
                for trg_ind in trg_inds:
                    if self.check_ind(trg_start, trg_end, trg_ind):
                        src_more_freq_count += 1
                        new_map[src_ind].add(trg_ind)
        return new_map

    def filter_dict_based_on_lemma(
            self, dictionary, chunks_info, opts,
            allow_one_more_freq=False, enforce_one_more_freq=False):
        src_start, src_end, trg_start, trg_end = chunks_info
        new_map = collections.defaultdict(set)

        srcd, trgd = opts.src_data, opts.trg_data

        for src_ind, trg_inds in dictionary.items():
            if self.check_lemma_id(src_start, src_end, srcd, src_ind):
                for trg_ind in trg_inds:

                    if self.check_lemma_id(trg_start, trg_end, trgd, trg_ind) \
                            and (not enforce_one_more_freq or trg_start == 0):
                        new_map[src_ind].add(trg_ind)
                    elif (allow_one_more_freq or enforce_one_more_freq)\
                            and self.check_lemma_id(
                                trg_start, trg_end, trgd,
                                trg_ind, less_than_start=True):
                        new_map[src_ind].add(trg_ind)

            elif (allow_one_more_freq or enforce_one_more_freq) and\
                    self.check_lemma_id(
                        src_start, src_end, srcd, src_ind,
                        less_than_start=True):
                for trg_ind in trg_inds:
                    if self.check_lemma_id(trg_start, trg_end, trgd, trg_ind):
                        new_map[src_ind].add(trg_ind)
        return new_map

    def get_dtype_view(self, dtype):
        return self.dictionary_map[dtype]

    def _get_trimmed(self, new_size, basefun):
        srckeys, base_map, coverage = basefun()

        if new_size > len(base_map):
            return srckeys, base_map, coverage

        tmp_src = [*srckeys]
        random.shuffle(tmp_src)
        selected_src = tmp_src[:new_size]
        new_map = dict()
        for src_ind, trg_inds in base_map.items():
            if src_ind in selected_src:
                new_map[src_ind] = trg_inds
        return sorted(new_map.keys()), new_map, self.get_coverage(new_map)

    def get_dictionary(
            self, dtype, chunks_info=None, allow_one_more_freq=False,
            enforce_one_more_freq=False):
        # ^ the last two params are for filtering
        selected_d = self.dictionary_map[dtype]
        if chunks_info:
            selected_d = self.filter_dict(
                selected_d, chunks_info, allow_one_more_freq,
                enforce_one_more_freq)
        return sorted(selected_d.keys()), selected_d, \
            self.get_coverage(selected_d)

    def get_lemma_filtered_dictionary(
            self, dtype, chunks_info, opts,
            allow_one_more_freq=False, enforce_one_more_freq=False):
        selected_d = self.dictionary_map[dtype]
        selected_d = self.filter_dict_based_on_lemma(
            selected_d, chunks_info, opts, allow_one_more_freq,
            enforce_one_more_freq)
        return sorted(selected_d.keys()), selected_d, \
            self.get_coverage(selected_d)

    def get_trimmed(
            self, new_size, dtype, chunks_info=None,
            allow_one_more_freq=False, enforce_one_more_freq=False):
        selected_d = self.dictionary_map[dtype]
        if chunks_info:
            selected_d = self.filter_dict(
                selected_d, chunks_info, allow_one_more_freq,
                enforce_one_more_freq)
        return self._get_trimmed(new_size, selected_d)


class BigDict(TDict):
    def __init__(self, morph2dict, vocab):
        super().__init__()

        self.morph2dict = morph2dict
        self.src_vocab = vocab
        self.full_translation = {}

        for morph, d in morph2dict.items():
            for dt in DictTypes:
                current_d = self.dictionary_map[dt]
                sub_d = d.dictionary_map[dt]
                for src, trg in sub_d.items():
                    current_d[src].update(trg)

            for (src, trg), info_list in d.srctrg2info.items():
                self.srctrg2info[(src, trg)] += info_list

        if len(self.morph2dict) == 1:
            self.morph_dict = False
        else:
            self.morph_dict = True
        self.morph2dict["ALL"] = self


def get_vocab_from_dict(args):
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    vocab = set()

    for line in f.readlines():
        splits = line.split()
        if len(splits) == 5:  # morphological dictionary
            src, trg, _, _, _ = splits
        elif len(splits) == 6:  # morphological dictionary
            src, trg, _, _, _, _ = splits
        elif len(splits) == 2:
            src, trg = splits
        else:
            continue
        vocab.add(src)
    f.close()
    return vocab


def get_fine_grained_eval_dict(args, dict_source, opts, feats=None):
    f = open(dict_source, encoding=args.encoding, errors='surrogateescape')
    morph2dict = collections.defaultdict(TDict)
    reverse_dict = args.reverse_dict
    vocab = set()

    srcd, trgd = opts.src_data, opts.trg_data

    lines = f.readlines()
    for line in lines:
        splits = line.split()
        if len(splits) == 5:  # morphological dictionary
            src, trg, src_lem, trg_lem, morph = splits
            full_morph = morph
        elif len(splits) == 6:  # morphological dictionary
            src, trg, src_lem, trg_lem, morph, trg_morph = splits
            full_morph = (morph, trg_morph)
        elif len(splits) == 2:
            morph = src_lem = trg_lem = None
            src, trg = splits
            full_morph = None
        else:
            continue

        if feats is None:
            feats = args.filter_dict

        if feats and morph:
            good_feats = filter(lambda x: x[0] != "-", feats)
            # the whole chunk is negative
            bad_feats = filter(lambda x: x[0] == "-", feats)
            if not any([all([y.lower() in morph.lower()
                        for y in x.split("+")]) for x in good_feats]):
                continue
            if any([all([y.lower() in morph.lower()
                         for y in x[1:].split("+")]) for x in bad_feats]):
                continue

        if reverse_dict:
            tmp = src; src = trg; trg = tmp
            tmp_lem = src_lem; src_lem = trg_lem; trg_lem = tmp_lem

        mdict = morph2dict[morph]
        mdict.srctrg2info[(src, trg)].append((full_morph, src_lem, trg_lem))

        dt = DictTypes
        get_dict = mdict.get_dtype_view

        try:
            src_ind = srcd.word2ind[src]
            trg_ind = trgd.word2ind[trg]
            vocab.add(src)

            src_oov = src_ind >= srcd.first_oov_index
            trg_oov = trg_ind >= trgd.first_oov_index
            to_add = {dt.ALL}
            trg_nonce = trg_ind in trgd.nonce_ids
            src_nonce = src_ind in srcd.nonce_ids

            if not src_oov and not trg_oov:
                to_add.add(dt.IN_VOCAB)
            else:
                to_add.add(dt.EITHER_OOVS)

                if src_oov and trg_oov:
                    to_add.add(dt.BOTH_OOVS)
                if src_oov:
                    to_add.add(dt.SRC_OOVS)
                    to_add.add(dt.ONE_SIDE_OOV)
                if trg_oov:
                    to_add.add(dt.TRG_OOVS)
                    to_add.add(dt.ONE_SIDE_OOV)

                if trg_nonce and src_nonce:
                    to_add.add(dt.BOTH_NONCES)
                elif src_nonce:
                    to_add.add(dt.SRC_NONCES)
                    to_add.add(dt.ONE_SIDE_NONCE)
                elif trg_nonce:
                    to_add.add(dt.TRG_NONCES)
                    to_add.add(dt.ONE_SIDE_NONCE)
                if trg_nonce or src_nonce:
                    to_add.add(dt.EITHER_NONCES)

            # If the source word has another OOV target candidate and has been
            # added to trg_oovs or both_oovs because of that its other non-OOV
            # translations should also be added to these dictionaries (to evaluate
            # performance correctly) OR that src-trg pair should be removed
            # from the dictionary
            if not trg_oov:
                if src_ind in get_dict(dt.TRG_OOVS):
                    to_add.add(dt.TRG_OOVS)
                    to_add.add(dt.ONE_SIDE_OOV)
                if src_ind in get_dict(dt.BOTH_OOVS):
                    to_add.add(dt.BOTH_OOVS)
            if not trg_nonce:
                if src_ind in get_dict(dt.TRG_NONCES):
                    to_add.add(dt.TRG_NONCES)
                    to_add.add(dt.ONE_SIDE_NONCE)
                if src_ind in get_dict(dt.BOTH_NONCES):
                    to_add.add(dt.BOTH_NONCES)

            for dtype in to_add:
                get_dict(dtype)[src_ind].add(trg_ind)

        except KeyError:
            # this happens if there is an oov for which
            # we don't have an embedding
            pass
    vocab = sorted(vocab, key=lambda x: srcd.word2ind[x])
    bd = BigDict(morph2dict, vocab)
    avg_trans_per_word = 0

    a, invoc, b = bd.get_dictionary(DictTypes.IN_VOCAB)

    for src, trgs in invoc.items():
        avg_trans_per_word += len(trgs)
    avg_trans_per_word /= len(invoc)
    #print("AVG translation count: ", avg_trans_per_word)

    f.close()
    return bd


########################
#      TRAIN           #
########################


def build_train_dict(args, opts):
    src, trg = opts.src_data, opts.trg_data
    rev_dict = args.reverse_dict

    if args.numerals:
        print('Using only \'numerals\' training dictionary', file=sys.stderr)
        return _get_numerals_dictionary(src.word2ind, trg.word2ind)
    elif args.identical:
        src1, trg1, tag1 = _read_dictionary_from_file(
            args.dictionary, opts, args.encoding, rev_dict, train=True,
            filter_feats=args.filter_dict, balance_pos=args.balance_pos)
        src2, trg2, tag2 = _get_identical_dictionary(
            src.word2ind, trg.word2ind)

        return src1 + src2, trg1 + trg2, tag1 + tag2
    else:
        return _read_dictionary_from_file(
            args.dictionary, opts, args.encoding, rev_dict, train=True,
            filter_feats=args.filter_dict, balance_pos=args.balance_pos)

#### Wrappers for the _read_dictionary_from_file function
def build_val_dict(args, opts):
    rev_dict = args.reverse_dict
    return _read_dictionary_from_file(
        args.validation, opts, args.encoding, rev_dict,
        filter_feats=args.filter_dict, balance_pos=args.balance_pos)


def build_test_dict(args, opts):
    rev_dict = args.reverse_dict
    return _read_dictionary_from_file(
        args.test_dict, opts, args.encoding, rev_dict)


def _get_identical_dictionary(src_word2ind, trg_word2ind):
    src_indices, trg_indices = [], []
    print('Using identical strings as dictionary...')
    intersect = set(src_word2ind.keys()).intersection(set(trg_word2ind.keys()))
    print(f'Found {len(intersect)} identical strings.')
    for word in intersect:
        src_indices.append(src_word2ind[word])
        trg_indices.append(trg_word2ind[word])
    return src_indices, trg_indices, []


def _get_numerals_dictionary(src_word2ind, trg_word2ind):
    src_indices, trg_indices = [], []
    numeral_regex = re.compile('^[0-9]+$')
    src_numerals = {word for word in src_word2ind.keys()
                    if numeral_regex.match(word) is not None}
    trg_numerals = {word for word in trg_word2ind.keys()
                    if numeral_regex.match(word) is not None}
    numerals = src_numerals.intersection(trg_numerals)
    for word in numerals:
        src_indices.append(src_word2ind[word])
        trg_indices.append(trg_word2ind[word])
    return src_indices, trg_indices, []


def _read_dictionary_from_file(
        file_path, opts, encoding, rev_dict=False, train=False,
        filter_feats=None, balance_pos=False, ignore_tags=True):
    src_data, trg_data = opts.src_data, opts.trg_data

    f = open(file_path, encoding=encoding, errors='surrogateescape')
    src_indices, trg_indices, tag_indices = [], [], []

    pairs_added = set()

    lines = f.readlines()
    nlines = 0
    coverage = 0

    pos2tag_counter = collections.defaultdict(Counter)
    tag2dictindices = collections.defaultdict(list)
    relevant_pos = ["ADJ", "N", "V"]

    for line in lines:
        splits = line.split()
        if len(splits) == 5: # morphological dictionary
            src, trg, _, _, tag = splits
        elif len(splits) == 6:  # morphological dictionary between unrelated ls
            src, trg, _, _, tag, _ = splits
        elif len(splits) == 2:
            src, trg = splits
            tag = None
        else:
            continue

        # used to limit the initial seed dictionary to specific morphology
        if filter_feats and tag and not any(
                [all([y.lower() in tag.lower()
                 for y in x.split("+")]) for x in filter_feats]):
            continue
        if (src, trg) in pairs_added and ignore_tags:
            continue
        pairs_added.add((src, trg))

        nlines += 1

        if rev_dict:
            tmp = src
            src = trg
            trg = tmp
        try:
            src_ind = src_data.word2ind[src]
            trg_ind = trg_data.word2ind[trg]

            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
            coverage += 1
            if tag and opts.tag2ind:
                # lang=None because the dictionaries should
                # have already processed tags
                tag_ind = opts.tag2ind[crop_tag(tag, opts.morph_feats)]
                tag_indices.append(tag_ind)

            if tag:
                pos = get_pos(tag)
                if pos in relevant_pos:
                    pos2tag_counter[pos][tag] += 1
                    tag2dictindices[tag].append(len(src_indices)-1)

        except KeyError as e:
            pass

    if balance_pos:
        removed_lines, src_indices, trg_indices, tag_indices =\
           balance_dict_based_on_pos(
               pos2tag_counter, tag2dictindices, src_indices,
               trg_indices, tag_indices)
        coverage -= removed_lines
        nlines -= removed_lines

    coverage = coverage/nlines
    print(f"Retrieved dictionary {file_path.split('/')[-1]}, coverage: {coverage:7.2%}")
    f.close()
    if train:
        return src_indices, trg_indices, tag_indices
    else:
        return (src_indices, trg_indices, tag_indices), coverage


def balance_dict_based_on_pos(
        pos2tag_counter, tag2dictindices, src_indices,
        trg_indices, tag_indices):
    min_total = -1
    min_pos = None

    for pos, counter in pos2tag_counter.items():
        total = sum(counter.values())
        print(f"{pos}: {total}")
        if min_total < 0 or total < min_total:
            min_total = total
            min_pos = pos

    to_remove = []
    for pos, counter in pos2tag_counter.items():
        if pos == min_pos:
            continue
        while sum(counter.values()) > min_total:
            common_tag = counter.most_common(1)[0][0]
            counter[common_tag] -= 1
            indices = tag2dictindices[common_tag]
            index = indices.pop()
            to_remove.append(index)

    new_src, new_trg, new_tag = [], [], []
    for i in range(len(src_indices)):
        if i not in to_remove:
            new_src.append(src_indices[i])
            new_trg.append(trg_indices[i])
            if len(tag_indices) == len(src_indices):
                new_tag.append(tag_indices[i])

    return len(to_remove),  new_src, new_trg, new_tag

