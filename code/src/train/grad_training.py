import torch.nn as nn
import torch
from torch import cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collections import defaultdict
import logging
import random
import numpy as np
from tqdm import tqdm
from functools import  partial
import os
import sys

from src.utils.commons import *

from src.utils.dictionary_utils import build_val_dict
from src.test.eval_translation_dev import get_normal_translations, rate_translation


tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class Translator(nn.Module):
    def __init__(self, src_m, trg_m, reg_alpha):
        super().__init__()
        self.target_size = trg_m.shape[0]
        print(type(src_m))

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.src_embed = torch.tensor(
            src_m, dtype=torch.float32,
            requires_grad=False, device=self.device)
        self.trg_embed = torch.tensor(
            trg_m, dtype=torch.float32,
            requires_grad=False, device=self.device)

        w = torch.empty(src_m.shape[1], src_m.shape[1])
        nn.init.orthogonal_(w)
        self.W = nn.Parameter(w)
        self.ident = torch.eye(
            trg_m.shape[1], requires_grad=False, device=self.device)
        self.cross_ent = nn.CrossEntropyLoss()

        self.reg_alpha = reg_alpha

    def forward(self, x):
        semb = self.src_embed[x]

        if len(semb.shape) == 1:
            semb = semb.view(-1, semb.shape[0])
        trans = torch.mm(semb, self.W)
        us = torch.mm(trans, self.trg_embed.t())
        return us

    def ort_regularizer(self, loss, bsize):
        ww = torch.mm(self.W, self.W.t())
        ort_loss = torch.norm((ww - self.ident))

        # cross entropy loss already divides by bsize
        loss = loss + (self.reg_alpha * ort_loss)/bsize
        return loss

    def loss(self, scores, target, bsize):
        tar = torch.tensor(target, device=self.device).view(-1)

        loss = self.cross_ent(scores, tar)
        loss = self.ort_regularizer(loss, bsize)
        return loss


class Trainer(object):
    def __init__(self, logger, model, src_indices, trg_indices, args):
        super().__init__()
        self.logger = logger
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model
        self.model = self.model.to(self.device)

        # to save models after different epochs
        self.models = list()

        self.last_devloss = float('inf')
        self.src_indices = np.array(src_indices)
        self.trg_indices = np.array(trg_indices)

        self.min_lr = None
        self.optimizer = None
        self.scheduler = None

        self.dtype = set_precision(args.precision)
        self.xp = select_matrix_library(args.cuda)

    def setup_training(self, optimizer, lr, min_lr, momentum, cooldown):
        self.min_lr = min_lr

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr, momentum=momentum)
        elif optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        else:
            raise ValueError

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=0,
            cooldown=cooldown,
            factor=0.5,
            min_lr=min_lr)

    def save_training(self, model_fp):
        save_objs = (self.optimizer.state_dict(), self.scheduler.state_dict())
        torch.save(save_objs, open(f'{model_fp}.progress', 'wb'))

    def load_training(self, model_fp):
        assert self.model is not None
        optimizer_state, scheduler_state = torch.load(
            open(f'{model_fp}.progress', 'rb'))
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)

    def train(self, epoch_idx, batch_size):
        logger, model, src_indices, trg_indices =\
            self.logger, self.model, self.src_indices, self.trg_indices

        logger.info('At %d-th epoch with lr %f.', epoch_idx,
                    self.optimizer.param_groups[0]['lr'])
        model.train()

        shuf = list(range(len(src_indices)))
        random.shuffle(shuf)
        srcb = src_indices[shuf]
        trgb = trg_indices[shuf]
        if batch_size > 0:
            srcb = split2nsized(srcb, batch_size)
            trgb = split2nsized(trgb, batch_size)

        avg_loss = 0
        for src_batch, trg_batch in tqdm(zip(srcb, trgb), total=len(srcb)):
            y_pred = model(src_batch)
            loss = model.loss(y_pred, trg_batch, batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            logger.debug('loss %f', loss)
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss/len(srcb)

    def calc_loss(
            self, mode, src_indices, trg_indices, batch_size, epoch_idx=-1):
        self.model.eval()

        srcb = src_indices
        trgb = trg_indices
        if batch_size > 0:
            srcb = split2nsized(srcb, batch_size)
            trgb = split2nsized(trgb, batch_size)

        avg_loss = 0
        for src_batch, trg_batch in tqdm(zip(srcb, trgb), total=len(srcb)):
            y_pred = self.model(src_batch)
            loss = self.model.loss(y_pred, trg_batch, batch_size)
            avg_loss += loss.item()

        avg_loss = avg_loss/len(srcb)
        self.logger.info(
            'Average %s loss value per instance is %f at the end of epoch %d',
            mode, avg_loss, epoch_idx)
        return avg_loss

    def update_lr_and_stop_early(self, epoch_idx, devloss, estop):
        prev_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(devloss)
        curr_lr = self.optimizer.param_groups[0]['lr']

        stop_early = True
        if (self.last_devloss - devloss) < estop and \
            prev_lr == curr_lr == self.min_lr:
            self.logger.info(
                'Early stopping triggered with epoch %d (previous dev loss: %f, current: %f)',
                epoch_idx, self.last_devloss, devloss)
            stop_status = stop_early
        else:
            stop_status = not stop_early
        self.last_devloss = devloss
        return stop_status

    def save_model(self, epoch_idx, devloss, eval_res, model_fp):
        if model_fp is None:
            return
        fp = model_fp + '.nll_{:.4f}.{}.epoch_{}'.format(
            devloss, eval_res, epoch_idx)
        torch.save(self.model, open(fp, 'wb'))
        self.models.append((fp, devloss, eval_res))

    def reload_best(self):
        best_fp, _, best_res = self.models[0]
        best_acc_fp, _, best_acc = self.models[0]
        best_devloss_fp, best_devloss, _ = self.models[0]
        for fp, devloss, res in self.models:
            if res >= best_acc:
                best_acc_fp, best_acc = fp, res
            if devloss <= best_devloss:
                best_devloss_fp, best_devloss = fp, devloss
        self.model = None
        best_fp = best_acc_fp
        self.logger.info(f'loading {best_fp} for testing and returning')
        self.load_model(best_fp)
        return set([best_fp])

    def load_model(self, model):
        assert self.model is None
        self.logger.info('load model in %s', model)
        self.model = torch.load(
            open(model, mode='rb'), map_location=self.device)
        self.model = self.model.to(self.device)

        print("loaded...", type(self.model))
        epoch = int(model.split('_')[-1])
        return epoch

    def cleanup(self, save_fps, model_fp):
        for fp, _, _ in self.models:
            if fp in save_fps:
                continue
            os.remove(fp)
        os.remove(f'{model_fp}.progress')

    def simple_eval(
            self, mode, src_m, trg_m, src_inds, trg_inds, opts, epoch_idx=-1):
        self.model.eval()
        trans_src_m = src_m.dot(self.model.W.cpu().data)

        src2trg = defaultdict(set)
        for s, t in zip(src_inds, trg_inds):
            src2trg[s].add(t)
        src = sorted(src2trg.keys())

        translations = get_normal_translations(
            None, trans_src_m, trg_m, src, None,
            chunk_size=1000, xp=self.xp, dtype=self.dtype)
        acc, _, _ = rate_translation(
            src, src2trg, translations, opts,
            unimorph_only=False, out_file=None)

        self.logger.info(
            '%s acc is %f at the end of epoch %d', mode, acc, epoch_idx)
        return acc, trans_src_m


def split2nsized(lst, size):
    """
    Split the list into chunks of the specified size. Useful for working with
    the multiprocessing Pool.
    """
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def get_logger(log_file=None, log_level='info'):
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

    if log_file is not None:
        filep = logging.FileHandler(log_file, mode='a')
        filep.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(filep)
    return logger


def grad_train(
        x, z, src_indices, trg_indices, tag_indices,
        opts, validate_fun=None, test_fun=None, args=None):
    """
    Main training function.
    :param x:               source embedding matrix
    :param z:               target embedding matrix
    :param src_indices:
    :param trg_indices:
    :param opts:
    :param test_fun:
    :return:                transformed matrices x and z
    """
    if opts.verbose:
        print("Entering the training loop:")

    model = Translator(x, z, args.reg_alpha)
    model_out = args.model_out
    if model_out is None:
        print("Unknown output file for the model.")
        return

    path = os.path.dirname(model_out)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            print("File exists error.")
            return
    logger = get_logger(model_out + '.log')
    trainer = Trainer(logger, model, src_indices, trg_indices, args)
    trainer.setup_training(
        args.optimizer, args.lr, args.min_lr, args.momentum, args.cooldown)

    (dev_src_inds, dev_trg_inds, _), dev_cov = build_val_dict(args, opts)

    start_epoch = 0
    for epoch_idx in range(start_epoch, start_epoch + args.epochs):
        train_loss = trainer.train(epoch_idx, args.bs)
        print('Average TRAIN loss value per instance is %f at the end of epoch %d' % (train_loss, epoch_idx))

        with torch.no_grad():
            devloss = trainer.calc_loss(
                "DEV", dev_src_inds, dev_trg_inds, args.bs, epoch_idx)
            trainer.simple_eval(
                "TRAIN", x, z, src_indices, trg_indices, opts, epoch_idx)
            eval_res, _ = trainer.simple_eval(
                "DEV", x, z, dev_src_inds, dev_trg_inds, opts, epoch_idx)

        if trainer.update_lr_and_stop_early(epoch_idx, devloss, args.estop):
            break

        trainer.save_model(epoch_idx, devloss, eval_res, model_out)
        trainer.save_training(model_out)
    with torch.no_grad():
        save_fps = trainer.reload_best()
    trainer.cleanup(save_fps, model_out)

    print(type(trainer.model.W.data))
    try:
        data = np.array(trainer.model.W.data.cpu())
    except:
        data = np.array(trainer.model.W.data)

    return x, z, data
