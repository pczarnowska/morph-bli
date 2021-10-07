import argparse
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import random

import os
import sys
current_dir = os.getcwd()
print(current_dir)
sys.path.append(f'{current_dir}/../../project_m1/code')
from src.utils.tag_utils import *

rds_path = "/rds/user/pjc211/hpc-work/tagger/unimorph_schema.txt"
if Path(rds_path).is_file():
    UNIMORPH_SCHEMA_SC = rds_path
else:
    UNIMORPH_SCHEMA_SC = "unimorph_schema.txt"

unimorph_features = {}
feat_types = set()

with open(UNIMORPH_SCHEMA_SC, "r", errors='surrogateescape') as f:
    unimorph_features["notum"] = "POS"

    for line in f.readlines():
        split = line.strip().split()
        if len(split) == 3:
            feat_type, _, feat = split
            unimorph_features[feat] = feat_type
            feat_types.add(feat_type)


class NamedEnum(Enum):
    def __str__(self):
        return self.value


class UnimorphReader:
    def __init__(self, um_file, supported_feats=None,
                 merge_tags=False, balance_pos=True):
        self.supported_feats = supported_feats

        # pos2ftypes holds all possible feature types that go with
        # that POS (not values!)
        self.word2lemma2featmaps, pos2ftypes = self.process_um(um_file)

        self.balance_pos = balance_pos

        if not merge_tags:
            tag_fun = lambda x: self._get_string_tags(x, pos2ftypes=pos2ftypes)
        else:
            tag_fun = lambda x: self._merge_tags(x, keep_pos_separate=True)

        self.form2lemma_tag_all, self.form2lemma_tag_unique =\
            self._get_form2lemma_tag(tag_fun)

    def get_form2lemma_tag(self, unique=False):
        if unique:
            return self.form2lemma_tag_unique
        else:
            return self.form2lemma_tag_all

    def process_um(self, file, lowercase=True):
        word2lemma2featmaps = {}
        pos2ftypes = defaultdict(lambda: defaultdict(int))

        with open(file, "r", errors='surrogateescape') as f:
            lines = f.readlines()
            maxi = 0

            pos2count = defaultdict(int)
            for line in lines:
                split = line.strip().split("\t")
                if len(split) == 3:
                    lemma, word, tag = split

                    if lowercase:
                        lemma, word = lemma.lower(), word.lower()
                    if lemma.count(' ') > 0 or word.count(' ') > 0:
                        continue

                    featmap = self._get_ftype2val(tag, lemma)
                    POS = featmap["POS"]
                    pos2count[POS] += 1

                    lemma2featmaps = word2lemma2featmaps.setdefault(word, {})
                    entry = lemma2featmaps.setdefault(lemma, [])

                    if featmap not in entry:
                        entry.append(featmap)
                    for k in featmap.keys():
                        pos2ftypes[POS][k] += 1

        return word2lemma2featmaps, pos2ftypes

    def _get_ftype2val(self, full_tag, lemma):
        feat_map = {}
        for val in full_tag.lower().split(";"):
            if not self.supported_feats or unimorph_features[val] \
                    in self.supported_feats:
                ftype = unimorph_features[val]
                feat_map[ftype] = val
        feat_map["lemma"] = lemma
        return feat_map

    def _get_form2lemma_tag(
            self, tags2strings_fun, min_tag_frequency=0,
            remove_ambiguous=False):
        form_lemma_tag = []
        form_lemma_tag_unique = []

        form2lemma_tag = defaultdict(set)
        repeated_forms = set()
        all_forms_len = len(self.word2lemma2featmaps)
        tag_counts = Counter()

        lemma_counter = Counter()
        lemma_threshold = 10

        all_forms =  list(self.word2lemma2featmaps.keys())
        random.shuffle(all_forms)

        for form in all_forms:
            lemma2featmaps= self.word2lemma2featmaps[form]
            if len(lemma2featmaps) > 1:
                repeated_forms.add(form)
            else:
                for l, fmap in lemma2featmaps.items():
                    if len(fmap) > 1:
                        repeated_forms.add(form)

            for lemma, tags in lemma2featmaps.items():
                tags = tags2strings_fun(tags)
                for i, tag in enumerate(tags):
                    lemma_counter[lemma] += 1

                    tag_counts[tag] += 1
                    form2lemma_tag[form].add((lemma, tag))

        # if min_tag_frequency > 0:
        #     pass

        if remove_ambiguous:
            unhappy_tags = set()
            new_to_rem = set()
            print(f"Removing ambiguous {len(repeated_forms)}" +
                  f" out of {all_forms_len}")
            for form in repeated_forms:
                for l, t in form2lemma_tag[form]:
                    unhappy_tags.add(t)
                del form2lemma_tag[form]

        print(f"Repeated forms {len(repeated_forms)} out of {all_forms_len}")
        for form, elems in form2lemma_tag.items():
            sorted_elems = sorted(elems, key=lambda x: x[1])
            l, t = sorted_elems[0]
            form_lemma_tag_unique.append((form, (l, t)))

            for e in elems:
                form_lemma_tag.append((form, e))

        return form_lemma_tag, form_lemma_tag_unique


################################# TAGS PROCESSING FUNCTIONS ################

    def _get_string_tags(self, tags, pos2ftypes):
        new_tags = [{key:value for key, value in fmap.items()
                     if key != "lemma"} for fmap in tags]
        return [';'.join(sorted(list(ftype2val.values()))) 
                for ftype2val in new_tags]

    def _merge_tags(self, tags, keep_pos_separate=False):
        """
        Get only one tag (string) for a word by merging multiple feature values
        into a single value. For example if a form can have two possible tags:
        N;ACC;SG and N;NOM;SG
        we get
        ACC/NOM;N;SG
        The merged features appear in orthographical order.

        :param tags: a list of tags (ftype2val dictionaries)
        :param keep_pos_separate: If true then don't mix tags for different POS
            - this can cause returning more than one tag
        :return:
        """

        lem = None
        for ftype2val in tags:
            if lem is None:
                lem = ftype2val["lemma"]
            else:
                if ftype2val["lemma"] != lem:
                    print(f"Different lemma: {lem} vs {ftype2val['lemma']}")

        if keep_pos_separate:
            pos2tags = {}
            for ftype2val in tags:
                pos_tag_list = pos2tags.setdefault(ftype2val["POS"], [])
                pos_tag_list.append(ftype2val)
            return [self._merge_tags_helper(tags)
                    for pos, tags in pos2tags.items()]
        else:
            return [self._merge_tags_helper(tags)]

    def _merge_tags_helper(self, tag_set):
        ftype2vals = {}
        for ftype2val in tag_set:
            for ftype, val in ftype2val.items():
                if ftype == "lemma":
                    continue
                val_set = ftype2vals.setdefault(ftype, set())
                val_set.add(val)

        new_tag = []
        for ftype, vals in ftype2vals.items():
            vals = list(vals)
            single_val = '/'.join(sorted(vals))
            new_tag.append(single_val)
        return ';'.join(sorted(new_tag))
