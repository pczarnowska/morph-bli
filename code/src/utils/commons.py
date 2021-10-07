from tabulate import tabulate
from enum import Enum
import numpy as np
import os
from src.utils import embeddings
from src.utils.tag_utils import mapping_function, get_pos, get_ftype2val
from collections import defaultdict

import sys
from src.utils.cupy_utils import *


def set_precision(prec):
    # Choose the right dtype for the desired precision
    if prec == 'fp16':
        return 'float16'
    elif prec == 'fp32':
        return 'float32'
    elif prec == 'fp64':
        return 'float64'


def select_matrix_library(cuda):
    # NumPy/CuPy management
    if cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
    else:
        xp = np
    return xp


############# READING FILES/GETTING PATHS ########################

def get_wn_source_paths(root, lang):
    # Where can wordnets be found - which directories within 
    # the main wordnet dir
    source_paths = [f'{root}/cldr/wn-cldr-{lang}.tab',
                              f'{root}/wikt/wn-wikt-{lang}.tab',
                              f'{root}/open/wn-data-{lang}.tab']
    return [sp for sp in source_paths if os.path.isfile(sp)]


def read_wn(wnsource, langname, pos=[]):
    concept2form = {}
    for fname in get_wn_source_paths(wnsource, langname):
        with open(fname, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.strip()
                splits = line.split("\t")
                if len(splits) != 3:
                    continue

                concept, _, form = splits
                cpos = concept.split("-")[1].upper()
                if " " in form or (pos and cpos not in pos):
                    continue

                # if concept not in concept2form:
                form_set = concept2form.setdefault(concept, set())
                form_set.add(form)
    return concept2form


def get_um_path(umsource, lang):
    pa = f"{umsource}/{lang}/clean_{lang}"
    p0 = f"{umsource}/{lang}/fixed_{lang}"

    p = f"{umsource}/{lang}/new_{lang}"
    p2 = f"{umsource}/{lang}/{lang}"

    if os.path.isfile(pa):
        print(f"Using tge cleanest version of unimorph for {lang} " +
              "with good genders.")
        return pa
    elif os.path.isfile(p0):
        print(f"Using a version of unimorph for {lang} with fixed genders.")
        return p0
    elif os.path.isfile(p):
        print(f"Using an alternative version of unimorph for {lang}.")
        return p
    elif os.path.isfile(p2):
        return p2
    else:
        return None


def read_um_full(fname, lang=None, lowercase=True):
    paradigm = {}
    pos2ftypes = defaultdict(set)
    print("UM", lang)
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            split = line.split("\t")
            if len(split) == 3:
                lemma, form, morph = split

                if lowercase:
                    lemma, form = lemma.lower(), form.lower()

                if " " in lemma or " " in form:
                    continue
                pos = morph.split(";")[0]
                if (pos, lemma) not in paradigm:
                    paradigm[(pos, lemma)] = {}
                morph = ";".join(sorted(morph.split(";")))
                if lang:
                    morphs = mapping_function(morph, lang)
                for morph in morphs:
                    paradigm[(pos, lemma)][morph] = form
                ftype2val = get_ftype2val(morph)
                pos = ftype2val["POS"]
                pos2ftypes[pos] |= set(ftype2val.keys())
    print(lang)
    sortpos = sorted(pos2ftypes.keys())
    for p in sortpos:
        print(p, sorted(pos2ftypes[p]))
    return paradigm


def read_um(file, special=False, lang=None, lowercase=True):
    word2feats = defaultdict(set)
    word2lemmas = defaultdict(set)
    lemma2word = defaultdict(set)

    word2lemmas_pos = defaultdict(set)
    lemma_pos2word = defaultdict(set)

    with open(file, "r", errors='surrogateescape') as f:
        for line in f.readlines():
            split = line.strip().split("\t")
            if len(split) == 3:
                lemma, word, feats = split

                if lowercase:
                    lemma, word = lemma.lower(), word.lower()
                if " " in lemma or " " in word:
                    continue

                feats = ";".join(sorted(feats.split(";")))
                if lang:
                    feats = mapping_function(feats, lang)
                else:
                    feats = [feats]

                lemma2word[lemma].add(word)
                word2lemmas[word].add(lemma)

                for feat in feats:
                    pos_tag = get_pos(feat)
                    word2feats[word].add(feat)
                    word2lemmas_pos[word].add(f"{lemma}_{pos_tag}")
                    lemma_pos2word[f"{lemma}_{pos_tag}"].add(word)

    if special:
        return (lemma2word, word2feats, word2lemmas,
                (word2lemmas_pos, lemma_pos2word))
    return (lemma2word, word2feats, word2lemmas)


def get_file_paths(
        root_path, in_name="", in_dir_name=None,
        not_in_dir_name=[], not_in_file_name=[]):
    fnames = []

    for root, dirs, fs in os.walk(root_path.rstrip("/") + "/", topdown=True):
        if in_dir_name:
            if type(in_dir_name) == list:
                res = [indir not in root for indir in in_dir_name]
                if sum(res) != 0:
                    continue
            else:
                if in_dir_name not in root:
                    continue
        s = sum([1 if n in root else 0 for n in not_in_dir_name])
        if s > 0:
            continue
        for name in fs:
            s2 = sum([1 if n in name else 0 for n in not_in_file_name])

            if type(in_name) == list:
                cond = all([x in name for x in in_name])
            else:
                cond = in_name in name

            if cond and s2 <= 0:
                fnames.append(root.rstrip("/") + "/" + name)
    return sorted(fnames)


def get_dir_paths(root_path, in_final_dir_name=""):
    fnames = []
    for root, dirs, fs in os.walk(root_path.strip("/") + "/", topdown=True):
        for name in dirs:
            if in_final_dir_name in name:
                fnames.append(root.strip("/") + "/" + name)
    return sorted(fnames)


############# GETTING RESULTS #############################

class TType(Enum):
    trans = 0
    morph = 1
    lemma = 2


def get_results(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return None, None, None
        tmp = lines[-1].split("%")
        if len(tmp) != 5:
            return None, None, None
        tacc, macc, lacc = tmp[1][-5:], tmp[2][-5:], tmp[3][-5:]
    return tacc, macc, lacc


def get_info(file_path):
    splits = file_path.split("/")
    name = splits[-1] if file_path[-1] != "/" else splits[-2]
    method = splits[-2]
    submethod = method.split("-")[-1]
    setts = name.split("-")
    morph = setts[-1].strip(".txt")
    k = setts[-2]
    return method, submethod, k, morph


#################### LATEX TABLES ########################

def generate_latex_table(
        header, first_col, data, file_path=None,
        transpose=False, bold_fun=lambda x: x):
    if transpose:
        data = np.array(data).T
        tmp = header
        header = [header[0]] + first_col
        first_col = tmp[1:]

    new_data = []
    for i, row in enumerate(data):
        new_data.append([first_col[i].replace("_", "\_")] + bold_fun(row))

    table = tabulate(new_data, header, tablefmt="latex_raw")
    print(table)
    if file_path:
        with open(file_path, "w") as f:
            f.write(table)
    return table


def generate_tsv_table(header, data, file_path):
    with open(file_path, "w") as f:
        for name in header:
            f.write(name + "\t")
        f.write("\n")
        for row in data:
            for r in row:
                f.write(r + "\t")
            f.write("\n")


############# BOLDING

def bold_below(row, threshold):
    new_row = []
    for i, val in enumerate(row):
        val = float(val)
        if val < threshold:
            new_row.append('\\textbf{{{0:#.3f}}}'.format(val))
        else:
            new_row.append('{0:#.3f}'.format(val))
    return new_row


def bold_max(row):
    new_row = []

    max = 0.0
    for val in row:
        val = float(val)
        if val > max:
            max = val

    for i, val in enumerate(row):
        val = float(val)
        if val == max:
            new_row.append('\\textbf{{{0:#.2f}}}'.format(val))
        else:
            new_row.append('{0:#.2f}'.format(val))
    return new_row
