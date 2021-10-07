import argparse
import random
import os
import sys
from collections import defaultdict

import pywikibot
from sklearn.model_selection import train_test_split
import argparse

site = pywikibot.Site()

current_dir = os.getcwd()
sys.path.append(f'{current_dir}/../code')
from src.utils.tag_utils import *
from src.utils.commons import get_um_path


langcode2lname = {
    "fra":"French", "spa":"Spanish", "cat":"Catalan",
    "por":"Portuguese", "ita":"Italian", "oci":"Occitan", "eng":"English"}


def get_lines(file, examples_alread_added, first_lang=False, verbose=False):
    """
    Use target language
    :param file:
    :return:
    """
    f = open(file, errors='surrogateescape')
    to_ret = set()
    lines = f.readlines()

    lemmas_src, lemmas_trg, tags = set(), set(), set()

    lemmas_src2forms = defaultdict(set)
    lemmas_trg2forms = defaultdict(set)

    i = 0
    skipped_lines = set()
    lem_repeated = set()

    lemreps = set()
    totallems = set()

    for line in lines:
        splits = line.split()
        i += 1
        if len(splits) == 5:  # morphological dictionary
            src, trg, lem_src, lem_trg, tag = splits
            tag1 = tag
            tag2 = tag
        elif len(splits) == 6:
            src, trg, lem_src, lem_trg, tag1, tag2 = splits
        elif len(splits) == 2:
            src, trg = splits
            lem_src, lem_trg = "X", "X"
            tag = None
            tag1 = None
            tag2 = None
        else:
            continue

        if lem_trg.count(' ') > 0 or trg.count(' ') > 0:
            continue

        if first_lang:
            lem, wform, tag = lem_src, src, tag1
        else:
            lem, wform, tag = lem_trg, trg, tag2

        if tag is not None:
            new_line = "\t".join([lem, wform, tag]) + "\n"
        else:
            new_line = None

        totallems.add(lem)
        if lem in examples_alread_added and \
                f"{wform}-{tag}" in examples_alread_added[lem]:
            if verbose: print(f"continue {src}-{tag}")
            skipped_lines.add(new_line)
            continue
        elif lem in examples_alread_added:
            lemreps.add(lem)
            lem_repeated.add(new_line)

        if new_line != None:
            to_ret.add(new_line)
            tags.add(tag)

        if tag:
            lemmas_src2forms[lem_src].add(f"{src}-{tag}")
            lemmas_trg2forms[lem_trg].add(f"{trg}-{tag}")
        else:
            lemmas_src2forms[lem_src].add(f"{src}")
            lemmas_trg2forms[lem_trg].add(f"{trg}")

    # nskipped = len(skipped_lines)
    # nrepeated = len(lem_repeated)
    # print(file)
    #  print("Skipped lines because in train/dev:", nskipped, ":", nskipped/(nskipped+len(to_ret)))
    # print("#lines with lem repeated", nrepeated, ":", nrepeated/len(to_ret))
    # print("Ration of repeated lems", len(lemreps), ":", len(lemreps)/len(totallems))
    #
    # print(f"lang1 lemmas {len(lemmas_src2forms)}, lang2 {len(lemmas_trg2forms)}")
    # if first_lang:
    #     x = [len(fs) for l, fs in lemmas_src2forms.items()]
    #     print(f"(first lang) src/lemma: {len(to_ret)/len(lemmas_src2forms)}")
    # else:
    #     x = [len(fs) for l, fs in lemmas_trg2forms.items()]
    #     print(f"(second lang) trg/lemma: {len(to_ret)/len(lemmas_trg2forms)}")

    f.close()
    return to_ret, lemmas_src2forms, lemmas_trg2forms, tags


def get_um_lines(lang, umsource, examples_alread_added, tags, lowercase=True):
    fname = get_um_path(umsource, lang)

    to_ret = set()
    unmatched_tags = set()
    unused_lemmas = set()
    len_unknown_tags = 0

    with open(fname, "r", errors='surrogateescape') as f:
        for line in f.readlines():
            split = line.strip().split("\t")
            if len(split) == 3:
                lemma, word, feats = split
                if lemma in unused_lemmas:
                    continue

                if " " in lemma or " " in word:
                    continue
                if lowercase:
                    lemma, word = lemma.lower(), word.lower()

                feats = ";".join(sorted(feats.split(";")))
                if lang:
                    feats = mapping_function(feats, lang)
                for f in feats:
                    if lemma in examples_alread_added and \
                            f"{word}-{f}" in examples_alread_added[lemma]:
                        unused_lemmas.add(lemma)
                    # the following is for MUSE dict
                    elif len(examples_alread_added) == 1 and word in \
                            examples_alread_added["X"]:
                        unused_lemmas.add(lemma)
                    else:
                        new_line = "\t".join([lemma, word, f]) + "\n"
                        if f not in tags and False:
                            unmatched_tags.add(f)
                            len_unknown_tags += 1
                        else:
                            to_ret.add(new_line)

    print("\nAdding new UNIMORPH LINES:", len(to_ret))
    print("Skipped lemmas", len(unused_lemmas))
    return to_ret


def retrieve_words(lang, category, limit=None):
    if not lang or not category:
        return []

    cat = pywikibot.Category(site, f'Category:{lang} {category}')
    result = []
    for e, i in enumerate(cat.members()):
        title = i.title()
        if len(title.split(" ")) > 1:
            continue
        result.append(title)
        if limit and len(result) == limit:
            break
    return result


def get_extra_lems(lang, examples_alread_added, lowercase=True):
    lems_names = ["adverbs", "pronouns", "prepositions", "determiners",
                  "exclamations",
                  "conjunctions", "interjections", "numerals", "particles"]

    unused_lemmas = set()
    extra_lems = set()
    for name in lems_names:
        extra_lems = extra_lems.union(retrieve_words(lang, name))

    to_ret = set()
    for lem in extra_lems:
        if lowercase:
            lem = lem.lower()
        if lem in examples_alread_added:
            unused_lemmas.add(lem)
        else:
            new_line = "\t".join([lem, lem, "NotUM"]) + "\n"
            to_ret.add(new_line)

    print("\nExtra lem lines:", len(to_ret))
    print("Ignored lems: ", len(unused_lemmas))
    return to_ret


def merge_dicts(d1, d2):
    for k, vals2 in d2.items():
        if k in d1:
            vals1 = d1[k]
            for v in vals2:
                vals1.add(v)
        else:
            d1[k] = vals2
    return d1


def getnew(x):
    parts = x.strip().split("\t")
    return "\t".join(parts[:-1])


def action(
        odir, base_dict_dir, lang1 ,lang2, first_lang=False,
        include_unimorph=False, umsource=None, go_beyond_um=False):
    if not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)

    print("------", umsource)

    out_file = lambda x: f"{odir}/{lang1}-{lang2}.tag.{x}.txt"
    source_file = lambda x: f"{base_dict_dir}/{lang1}-{lang2}.{x}.txt"

    target_lang = lang1 if first_lang else lang2
    print(f"Data retrieved for language: {target_lang}")

    src_lemmas2forms, trg_lemmas2forms, all_tags = {}, {}, set()

    split2lines = {}

    if "muse" in base_dict_dir or "dinu" in base_dict_dir:
        train_key = "clean.train.shuf"
    else:
        train_key = "train"
    splits = ["test",  "dev", train_key]

    for split in splits:  # this order should be maintained
        # so that we're not training on anything the model will be tested on
        examples_alread_added =\
            src_lemmas2forms if first_lang else trg_lemmas2forms
        lines, slems2forms, tlems2forms, ntags = get_lines(
            source_file(split), examples_alread_added, first_lang=first_lang)
        split2lines[split] = lines
        print(f"Initial lines in {split}: {len(lines)}")
        src_lemmas2forms = merge_dicts(src_lemmas2forms, slems2forms)
        trg_lemmas2forms = merge_dicts(trg_lemmas2forms, tlems2forms)
        all_tags.update(ntags)

    for k in splits:
        for j in splits:
            if k != j:
                assert not split2lines[k].intersection(split2lines[j])

    # add examples from unimorph which have not been present in the
    # morph dicitonary
    if include_unimorph:
        if not umsource:
            print("Can't add unimorph data. Source not provided.")
            return
        examples_alread_added =\
            src_lemmas2forms if first_lang else trg_lemmas2forms
        lines = get_um_lines(
            target_lang, umsource, examples_alread_added, all_tags)

        for k in splits:
            if "train" not in k: assert not lines.intersection(split2lines[k])

        todev = int(0.1 * len(lines))
        print(f"Number of additional train unimorph entries: {len(lines)-todev}")
        print(f"Number of additional dev unimorph entries: {todev}\n")
        lines = list(lines)
        if "dev" in splits:
            split2lines["dev"] = split2lines["dev"].union(lines[:todev])
            split2lines[train_key] =\
                split2lines[train_key].union(lines[todev:])
        else:
            split2lines[train_key] =\
                split2lines[train_key].union(lines)

    if go_beyond_um:
        extra_lines = list(get_extra_lems(
            langcode2lname[target_lang], examples_alread_added))
        todev = int(0.2 * len(extra_lines))
        print("Extra wiktionary lines in dev:", len(extra_lines) - todev)
        print("Extra wiktionary lines in dev:", todev)
        if "dev" in splits:
            split2lines["dev"] =\
                split2lines["dev"].union(extra_lines[:todev])
            split2lines[train_key] =\
                split2lines[train_key].union(extra_lines[todev:])
        else:
            split2lines[train_key] =\
                split2lines[train_key].union(extra_lines)

    for k in splits:
        for j in splits:
            if k != j:
                assert not split2lines[k].intersection(split2lines[j])

    for split in splits:
        if split == train_key:
            ksplit = "train"
        else:
            ksplit = split
        with open(out_file(ksplit), "w") as f1:
            lines = list(split2lines[split])
            print(f"Final lines in {split}: {len(lines)}")
            random.shuffle(lines)
            f1.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dict_dir')
    parser.add_argument('--lang1')
    parser.add_argument('--lang2')
   # parser.add_argument('--dev_size', default=0.2, type=float)
   # parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--out_dir')
    parser.add_argument('--first_lang', action='store_true')
    parser.add_argument('--include_unimorph', action='store_true')
    parser.add_argument('--go_beyond_unimorph', action='store_true')
    parser.add_argument('--umsource')

    args = parser.parse_args()
    action(args.out_dir, args.base_dict_dir, args.lang1, args.lang2, first_lang=args.first_lang, include_unimorph=args.include_unimorph,
           umsource=args.umsource, go_beyond_um=args.go_beyond_unimorph)


if __name__ == "__main__":
    main()
