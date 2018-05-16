
import argparse
import copy
from collections import Counter
from functools import reduce
from itertools import chain
import logging
import numpy as np
from operator import mul
import os
from sklearn.metrics import hamming_loss
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from dataset import Dictionary
from model import DemoPredictor

class Experiment:
    def __init__(self, args, logger, label_size):
        self.args = args
        self.logger = logger
        Dict = Dictionary(
                args.data_path+'dict_'+args.task+args.partial_ratio+'.json')
        self.dict = Dict.dict
        self.attr_len = Dict.attr_len
        self.all_the_poss = reduce(mul, Dict.attr_len, 1)
        self.logger.info("Experiment initializing . . . ")
        self.model = DemoPredictor(
                        logger, self.dict.__len__(),
                        args.item_emb_size, label_size, Dict.attr_len, args.num_negs,
                        args.rnn_type, args.rnn_size, args.rnn_layer, args.rnn_drop,
                        ).cuda()
        self.select_optimizer()
        self.criterion = nn.NLLLoss()
        self.logger.info(self.model)

    def select_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if(self.args.opt == 'Adam'):
            self.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif(self.args.opt == 'RMSprop'):
            self.optimizer =  optim.RMSprop(parameters, lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay,
                                        momentum=self.args.momentum)
        elif(self.args.opt == 'SGD'):
            self.optimizer =  optim.SGD(parameters, lr=self.args.learning_rate)
        elif(self.args.opt == 'Adagrad'):
            self.optimizer =  optim.Adagrad(parameters, lr=self.args.learning_rate)
        elif(self.args.opt == 'Adadelta'):
            self.optimizer =  optim.Adadelta(parameters, lr=self.args.learning_rate)

    def run_epoch(self, data_loader, trainable=False):
        num_samples = data_loader.dataset.__len__()
        num_steps = (num_samples // self.args.batch_size) + 1
        self.logger.info("== {} mode : {} steps for {} samples =="
            .format(data_loader.dataset.data_type, num_steps, num_samples))

        # change the mode
        if trainable:
            self.model.train()
        else:
            self.model.eval()

        # step training or evaluation with given batch size
        loss_sum = 0
        self.y_counter = Counter()
        self.y_em_counter = Counter()
        self.hm_acc = self.em = self.num_users = 0
        for i, batch in enumerate(data_loader):
            t0 = time.clock()
            if trainable:
                self.optimizer.zero_grad()
            logit, loss = self.model(batch)
            
            if trainable:
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_max_norm)
                self.optimizer.step()
            ls = loss.data.cpu().numpy()
            loss_sum += ls[0]

            self.accumulate_score(logit, batch[2].numpy(), batch[3].numpy())

            if (i+1) % self.args.print_per_step == 0:
                hm, p, r, f1 = self.get_score()
                t1 = time.clock()
                self.logger.info("<step {}> Loss={:5.3f}, time:{:5.2f}, Hamming={:2.4f}, P:{:2.4f}, R:{:2.4f}, F1:{:2.4f}"
                                    .format(i+1, ls[0], t1-t0, hm, p, r, f1))
        hm, p, r, f1 = self.get_score()
        return loss_sum / num_steps, hm, p, r, f1

    def accumulate_score(self, logit, onehot, observed):
        start = 0
        pred = []
        logit = logit.transpose(1,0)
        for al in self.attr_len:
            end = start + al
            pred.append(np.argmax(logit[start:end], 0) + start)
            start += al

        y_pred = np.asarray(pred).transpose(1,0)
        y_true = np.asarray([[j for j, l in enumerate(oh) if l] \
                                for i, oh in enumerate(onehot)])
        batch_size = y_true.shape[0]

        print('y pred :', y_pred)
        print('ob :', observed)
        sys.exit()

        for y in zip(y_pred, y_true):
            self.y_counter[str(y[1])] += 1
            if np.array_equal(y[0], y[1]):
                self.y_em_counter[str(y[1])] += 1
                # count exact matchings for evaluating wP, wR, wF1
                self.em += 1

        # accumulate hamming loss
        y_pred = list(chain.from_iterable(
                [yp + (i*sum(self.attr_len)) for i, yp in enumerate(y_pred)]))
        y_true = list(chain.from_iterable(
                [[j + (i*sum(self.attr_len)) for j, l in enumerate(oh) if l]  \
                    for i, oh in enumerate(onehot)]))

        hm_loss = hamming_loss(y_true, y_pred)
        self.hm_acc += batch_size * hm_loss
        self.num_users += batch_size

    def get_score(self):
        hm_loss = self.hm_acc / self.num_users
        wP = 0
        for y, cnt in self.y_counter.items():
            wP += self.y_em_counter[y] / cnt
        wP /= len(self.y_counter)

        wR = self.em / self.num_users
        if wP == 0 and wR == 0:
            wP = wR = wF1 = 0
        else:
            wF1 = (2 * wP * wR) / (wP + wR)
        return hm_loss, wP, wR, wF1


