'''
Decode model
'''
import argparse
from functools import partial

import torch
import regex as re

from dataloader import BOS, EOS, UNK_IDX
from model import decode_beam_search, decode_greedy
from util import maybe_mkdir
from tqdm import tqdm
tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')

from dataloader import BOS_IDX, EOS_IDX, EOF_IDX, EOF


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, help='Dev/Test file')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--nonorm', default=False, action='store_true')
    return parser.parse_args()


def setup_inference(max_len=100, beam_size=5, decode='greedy', nonorm=False):
    decode_fn = None
    if decode == 'greedy':
        decode_fn = partial(decode_greedy, max_len=max_len)
    elif decode == 'beam':
        decode_fn = partial(
            decode_beam_search,
            max_len=max_len,
            nb_beam=beam_size,
            norm=not nonorm)
    return decode_fn


def read_file(filename):
    lemmas, tags, golds = [], [], []
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            lemma, gold, tag = line.strip().split('\t')
            lemmas.append(lemma)
            tags.append(tag)
            golds.append(gold)
    return lemmas, tags, golds


def encode(model, lemma, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in lemma:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    attr = [0] * (len(model.attr_c2i) + 1)
    for tag in tags:
        if tag in model.attr_c2i:
            attr_idx = model.attr_c2i[tag] - tag_shift
        else:
            attr_idx = -1
        if attr[attr_idx] == 0:
            attr[attr_idx] = model.attr_c2i.get(tag, 0)

    return (torch.tensor(src, device=device).view(len(src), 1),
            torch.tensor(attr, device=device).view(1, len(attr)))


def an_encode(model, form, device):
    src = []
    src.append(model.src_c2i[BOS])
    for char in form:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    return (torch.tensor(src, device=device).view(len(src), 1))


def reinflect_form(model, device, decode_fn, tag, lemma, lemma_feats):
    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]

    if any([all([y.lower() in tag.lower() for y in x.split("+")]) for x in
            lemma_feats]):
        return lemma

    tag_splits = tag.split(';')

    src = encode(model, lemma, tag_splits, device)
    pred, _ = decode_fn(model, src)
    pred_out = ''.join(decode_trg(pred))
    return pred_out


def reinflect(model_source, lemmas, tags, multi=False):
    decode_fn = setup_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open(model_source, mode='rb'), map_location=device)
    model = model.to(device)

    forms = []
    lemma_feats = ["NFIN", "ADJ+SG+MASC+NOM", "N;+SG+NOM", "ADJ+SG+MASC", "N;+SG", "NOTUM"]

    for lemma, tag in tqdm(zip(lemmas, tags), total=len(lemmas)):

        if multi or isinstance(tag, list): # we need to inflect many times
            preds = []

            for t in tag:
                pred_out = reinflect_form(model, device, decode_fn, t, lemma,
                                          lemma_feats)
                preds.append(f"{pred_out}")
            if len(preds) == 0:
                preds.append(lemma)

            forms.append(preds)
        else:
            pred_out = reinflect_form(model, device, decode_fn, tag, lemma, lemma_feats)
            forms.append(f"{pred_out}")
    return forms


def analyse(model_source, forms):
    def get_lem_and_tag(chars):
        lem, tag = [], []
        to_add = tag
        for ch in chars:
            if ch == EOF:
                to_add = lem
                continue
            to_add.append(ch)

        try:
            tag = ';'.join(sorted(tag[0].split(";")))
        except:
            # print("problem", lem, tag)
            tag = "NFIN"

        return ''.join(lem), tag

    decode_fn = setup_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open(model_source, mode='rb'), map_location=device)
    model = model.to(device)

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]
    lemmas = []
    tags = []

    for wform in tqdm(forms, total=len(forms)):
        src = an_encode(model, wform, device)
        pred, _ = decode_fn(model, src)

        lem, tag = get_lem_and_tag(decode_trg(pred))
        lemmas.append(lem)
        tags.append(tag.upper())
    return lemmas, tags


def main():
    opt = get_args()
    lemmas, tags, golds = read_file(opt.in_file)
    thr=1000
    forms = reinflect(opt.model, lemmas[:thr], tags[:thr])
    correct = 0
    for gold, form in zip(golds[:thr], forms):
        if gold == form:
            correct += 1
        else:
            print(f"{form} {gold}")
    print(f"Acc: {correct/thr}")


if __name__ == '__main__':
    with torch.no_grad():
        main()
