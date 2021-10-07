from enum import Enum
import enum
from src.utils.commons import read_um, get_um_path
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import pandas as pd


class InputType(enum.IntEnum):
    SRC = 1
    TRG = 2


class InputData():
    def __init__(self, type, lang, umsource):
        self.type = type
        self.lang = lang
        self.words = None
        self.word2ind = None
        self.oov_words = None

        self.nonces = set()

        # ids for words forms we don't have a lemma for
        # note this is only for words in unimorph!
        self.nonce_ids = set()

        self.first_oov_index = 0
        self.matrix = None
        self.oov_matrix = None
        self.full_matrix = None

        self.um = None
        self.tags = None
        self.tag_ids = None

        self.ranked_lemmas = []
        self.lemma2ranked_ind = {}

        self.lemma2words, self.word2morph, self.word2lemma = \
            read_um(get_um_path(umsource, lang), lang=lang)

    def retrieve_word2ind(self):
        self.word2ind = {word: i for i, word in enumerate(self.words)}

    def is_nonce(self, oov):
        if oov not in self.word2lemma:
            # can't tell
            return False

        # to be a true nonce all forms for all of its lemmas have to be oovs
        lemmas = self.word2lemma[oov]
        for lemma in lemmas:
            paradigm = self.lemma2words[lemma]
            for w in paradigm:
                if w in self.word2ind and \
                        self.word2ind[w] < self.first_oov_index:  # not oov
                    return False
        return True

    def get_nonce_ids(self):
        if not self.oov_words:
            return

        self.nonces = set()
        self.nonce_ids = set()

        for oov in self.oov_words:
            if self.is_nonce(oov):
                self.nonces.add(oov)

        for n in self.nonces:
            self.nonce_ids.add(self.word2ind[n])

    def get_lemma_rank(self, sum_of_all=False):
        # lemma rank is the sum of ranks of all its words
        lemmas_and_ranks = []

        for lemma, words in self.lemma2words.items():
            total_rank = 0 if sum_of_all else len(self.words)

            for w in words:
                if w in self.word2ind:
                    wind = self.word2ind[w]
                    if sum_of_all:
                        total_rank += wind
                    elif wind < total_rank:
                        total_rank = wind
                elif sum_of_all:
                    total_rank += len(self.words)  # OOV has the highest rank
            lemmas_and_ranks.append([lemma, total_rank])

        sorted_lem_ranks = sorted(lemmas_and_ranks, key=lambda x: x[1])
        self.ranked_lemmas = [i[0] for i in sorted_lem_ranks]
        self.ranks = [i[1] for i in sorted_lem_ranks]

        self.lemma2ranked_ind = {
            l: i for i, l in enumerate(self.ranked_lemmas)}

    def initialize(self, words, oov_words, matrix, oov_matrix):
        self.words = words
        self.oov_words = oov_words
        self.oov_matrix = oov_matrix
        self.matrix = matrix
        self.first_oov_index = len(words)
        if oov_words and oov_matrix is not None:
            self.words += self.oov_words
            self.full_matrix = np.concatenate(
                (self.matrix, self.oov_matrix), axis=0)
        self.retrieve_word2ind()
        self.get_nonce_ids()

        if self.lemma2words:
            self.get_lemma_rank()


class Options(object):
    def __init__(self, args, xp, dtype):
        self.xp = xp
        self.dtype = dtype
        self.encoding = args.encoding
        self.umsource = args.umsource

        self.src_data = InputData(InputType.SRC, args.src_lang, args.umsource)
        self.trg_data = InputData(InputType.TRG, args.trg_lang, args.umsource)

        normal_filter_dict, oov_filter_dict = None, None
        if args.lemma_filter:
            normal_filter_dict = self.get_filter_dict(args.lemma_filter)
        if args.oov_lemma_filter:
            oov_filter_dict = self.get_filter_dict(args.oov_lemma_filter)

        self.filter_fun = lambda x: self._filter_fun(
            normal_filter_dict, oov_filter_dict, x)
        self.tag2ind = None

    def _filter_fun(self, normal_filter_dict, oov_filter_dict, word):
        if normal_filter_dict and word in normal_filter_dict:
            return normal_filter_dict[word]
        elif oov_filter_dict and word in oov_filter_dict:
            return oov_filter_dict[word]
        else:
            # All goes through the filter if there is no filtering dictionary
            return 1

    def get_filter_dict(self, fpath):
        word2res = {}
        non_filtered = 0
        misclassified = 0
        lems_in_um = 0
        with open(fpath, "r") as f:
            for line in f.readlines():
                splits = line.strip().split(' ')
                if len(splits) == 2:
                    wform, res = splits

                    if wform in self.trg_data.lemma2words.keys():
                        lems_in_um += 1
                        if res == "0":
                            misclassified += 1
                            res = "1"
                    else:
                        non_filtered += 1
                    word2res[wform] = int(res)
        print("Non filtered embeddings (classified as lemma):", non_filtered)
        print(f"Missclassified: {misclassified} ({misclassified/lems_in_um})")
        return word2res

    def get_data(self, input_type):
        if input_type == InputType.SRC:
            return self.src_data
        elif input_type == InputType.TRG:
            return self.trg_data
        else:
            return None
