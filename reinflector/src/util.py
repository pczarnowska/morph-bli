import logging
import os
import random
import string
import sys
from collections import namedtuple, defaultdict
from enum import Enum
from functools import partial

import numpy as np
from tqdm import tqdm

from dataloader import BOS_IDX, EOS_IDX, EOF_IDX, EOF, UNK_IDX

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class NamedEnum(Enum):
    def __str__(self):
        return self.value


def log_grad_norm(self, grad_input, grad_output, logger=None):
    try:
        logger.debug('')
        logger.debug('Inside %r backward', self.__class__.__name__)
        logger.debug('grad_input size: %r', grad_input[0].size())
        logger.debug('grad_output size: %r', grad_output[0].size())
        logger.debug('grad_input norm: %r', grad_input[0].detach().norm())
    except:
        pass


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm**(1. / norm_type)
    return total_norm


def maybe_mkdir(filename):
    '''
    maybe mkdir
    '''
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_logger(log_file, log_level='info'):
    '''
    create logger and output to file and stdout
    '''
    assert log_level in ['info', 'debug']
    fmt = '%(asctime)s %(levelname)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logger = logging.getLogger()
    log_level = {'info': logging.INFO, 'debug': logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode='a')
    filep.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(filep)
    return logger


def get_temp_log_filename(prefix='exp', dir='scratch/explog'):
    id = id_generator()
    fp = f'{dir}/{prefix}-{id}'
    maybe_mkdir(fp)
    return fp


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


Eval = namedtuple('Eval', 'desc long_desc res')


class Evaluator(object):
    pass


class BasicEvaluator(Evaluator):
    def __init__(self, target_chars):
        self.target_chars = target_chars

    def evaluate(self, predict, ground_truth):
        '''
        evaluate single instance
        '''
        correct = 1
        if len(predict) == len(ground_truth):
            for elem1, elem2 in zip(predict, ground_truth):
                if elem1 != elem2:
                    correct = 0
                    break
        else:
            correct = 0
        dist = edit_distance(predict, ground_truth)
        return correct, dist

    def get_lem_and_tag(self, indxs):
        lem, tag = [], []
        to_add = tag
        for indx in indxs:
            if indx == EOF_IDX:
                to_add = lem
                continue
            to_add.append(indx)
        return lem, tag

    def evaluate_lemma_tag(self, predict, ground_truth, verbose=False):
        '''
        evaluate single instance
        '''
        correct_lemma = 1
        correct_tag = 1

        pred_lem, pred_tag = self.get_lem_and_tag(predict)
        truth_lem, truth_tag = self.get_lem_and_tag(ground_truth)

        if verbose:
            print("LEMMA", "".join([self.target_chars[i] for i in pred_lem]), "   ",
                "".join([self.target_chars[i] for i in truth_lem]))

            print("TAG", ";".join([self.target_chars[i] for i in pred_tag]), 
                "   ", ";".join([self.target_chars[i] for i in truth_tag]))

        if len(pred_lem) == len(truth_lem):
            for elem1, elem2 in zip(pred_lem, truth_lem):
                if elem1 != elem2:
                    correct_lemma = 0
                    break
        else:
            correct_lemma = 0

        if len(pred_tag) == len(truth_tag):
            for elem1, elem2 in zip(pred_tag, truth_tag):
                if elem1 != elem2:
                    correct_tag = 0
                    break
        else:
            correct_tag = 0
        dist = edit_distance(pred_lem, truth_lem)

        return correct_lemma, correct_tag, dist

    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        '''
        evaluate all instances
        '''
        correct, correct_tag, correct_lem, distance, nb_sample = 0, 0, 0, 0, 0

        tag_predict = EOF in self.target_chars

        for src, trg in tqdm(data_iter(), total=nb_data):
            pred, _ = decode_fn(model, src)
            nb_sample += 1
            trg = trg.view(-1).tolist()
            trg = [x for x in trg if x != BOS_IDX and x != EOS_IDX]

            corr_full, dist = self.evaluate(pred, trg)
            if tag_predict:
                corr_lem, corr_tag, dist = self.evaluate_lemma_tag(pred, trg)
            else:
                corr_tag = 0
                corr_lem = corr_full
            correct += corr_full
            correct_tag += corr_tag
            correct_lem += corr_lem
            distance += dist

        acc = round(correct / nb_sample * 100, 4)
        acc_tag = round(correct_tag / nb_sample * 100, 4)
        acc_lem = round(correct_lem / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        if tag_predict:
            return [
                Eval('acc', 'full accuracy (lem and tag)', acc),
                Eval('acc_lem', 'lemma accuracy', acc_lem),
                Eval('acc_tag', 'tag accuracy', acc_tag),
                Eval('dist', 'average edit distance (lem)', distance)
            ]
        else:
            return [
                Eval('acc', 'word accuracy', acc),
                Eval('dist', 'average edit distance', distance)
            ]


class MultiOptionsEvaluator(BasicEvaluator):
    def __init__(self, target_chars, source_chars):
        self.target_chars = target_chars
        self.source_chars = source_chars

    def res_better(self, res_lem, res_tag, res_dist,
                   corr_lem, corr_tag, dist):
        if corr_lem < res_lem:
            return False
        if corr_lem > res_lem:
            return True

        if corr_tag < res_tag:
            return False
        if corr_tag > res_tag:
            return True

        if dist > res_dist:
            return False
        else:
            return True

    def multi_evaluate_lemma_tag(
            self, predict, ground_truths, verbose=False):
        res_lem, res_tag, res_dist = 0, 0, 10000

        for gtruth, gtag in ground_truths:
            if gtag is None:
                corr_lem, corr_tag, dist = self.evaluate_lemma_tag(
                    predict, gtruth, verbose=verbose)
            else:
                predict_w, predict_tag = predict

                corr_lem, dist = self.evaluate(predict_w, gtruth)
                corr_tag = (predict_tag == gtag).item()

                if verbose:
                    print("".join([self.target_chars[i] for i in predict_w]),
                          "   ",
                          "".join([self.target_chars[i] for i in gtruth]))
                    print(predict_tag.item(), gtag.item())

            if self.res_better(
                    res_lem, res_tag, res_dist, corr_lem, corr_tag, dist):
                res_lem, res_tag, res_dist = corr_lem, corr_tag, dist

        return res_lem, res_tag, res_dist

    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        '''
        evaluate all instances
        '''
        correct_full, correct_lem, correct_tag, distance, nb_sample =\
            0, 0, 0, 0, 0

        str_src2all_trg = defaultdict(list)
        str_src2src = {}

        for example in data_iter():
            if len(example) == 2:
                src, trg = example
                tag = None
                unk_in = sum([1 if x == UNK_IDX else 0 for x in trg]) > 0
                if unk_in in trg:
                    continue
            else:
                src, trg, tag = example

            str_src2all_trg[str(src)].append((trg, tag))
            str_src2src[str(src)] = src

        for i, (str_src, all_trg) in tqdm(
                enumerate(str_src2all_trg.items()), total=len(str_src2all_trg)):
            src = str_src2src[str_src]
            pred, _ = decode_fn(model, src)
            nb_sample += 1

            all_options = []
            for trg, tag in all_trg:
                trg = trg.view(-1).tolist()
                trg = [x for x in trg if x != BOS_IDX and x != EOS_IDX]
                all_options.append((trg, tag))

            corr_lem, corr_tag, dist = self.multi_evaluate_lemma_tag(
                pred, all_options, verbose=i < 10)

            corr_full, _ = self.evaluate(pred, trg)
            correct_full += corr_full
            correct_lem += corr_lem
            correct_tag += corr_tag
            distance += dist
        acc = round(correct_full / nb_sample * 100, 4)
        acc_lem = round(correct_lem / nb_sample * 100, 4)
        acc_tag = round(correct_tag / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        return [
                Eval('acc', 'full accuracy (lem and tag)', acc),
                Eval('acc_lem', 'lemma accuracy', acc_lem),
                Eval('acc_tag', 'tag accuracy', acc_tag),
                Eval('dist', 'average edit distance (lem)', distance)
        ]


def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])