
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

from dataset_rh import Dictionary
from model_rh import DemoPredictor

class Experiment:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        Dict = Dictionary(
                args.data_path+'dict_'+args.task_type+args.partial_ratio+'.json')
        self.dict = Dict.dict
        self.attr_len = Dict.attr_len
        self.all_the_poss = reduce(mul, Dict.attr_len, 1)
        self.logger.info("Experiment initializing . . . ")

        #attr_len = np.asarray(self.attr_len)[tasks]
        # build models
        self.model = []
        def build_models(tasks_list):
            for tasks in tasks_list:
                self.model.append(DemoPredictor(logger, self.dict.__len__(),
                            args.item_emb_size, Dict.attr_len, args.num_negs, args.user_rep,
                            args.rnn_type, args.rnn_size, args.rnn_layer, args.rnn_drop,
                            args.learning_form, args.partial_training, tasks = tasks).cuda())
        #self.task_sepa = [[0,1,2,3,4]]
        #self.task_sepa = [[0],[1],[2],[3],[4]]
        self.task_sepa = [args.tasks]
        #models = [[0],[1],[2],[3],[4]]
        #models = [[0]]
        build_models(self.task_sepa)
        for model in self.model:
            self.select_optimizer(model)
            self.logger.info(model)
        #self.tasks = [0, 1, 2, 3, 4]
        self.tasks = args.tasks

    def select_optimizer(self, model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if(self.args.opt == 'Adam'):
            model.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif(self.args.opt == 'RMSprop'):
            model.optimizer =  optim.RMSprop(parameters, lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay,
                                        momentum=self.args.momentum)
        elif(self.args.opt == 'SGD'):
            model.optimizer =  optim.SGD(parameters, lr=self.args.learning_rate,
                                        momentum=self.args.momentum)
        elif(self.args.opt == 'Adagrad'):
            model.optimizer =  optim.Adagrad(parameters, lr=self.args.learning_rate)
        elif(self.args.opt == 'Adadelta'):
            model.optimizer =  optim.Adadelta(parameters, lr=self.args.learning_rate)

    def run_epoch(self, data_loader, sample_type, trainable=False):
        num_samples = data_loader.dataset.__len__()
        num_steps = (num_samples // self.args.batch_size) + 1
        self.num_steps = num_steps
        self.logger.info("== {} mode : {} steps for {} samples =="
            .format(data_loader.dataset.data_type, num_steps, num_samples))


        self.y_counter = Counter()
        self.y_em_counter = Counter()
        self.yp_counter = Counter()
        self.yt_counter = Counter()
        self.hm_acc = self.em = self.num_users = 0
        self.attr_em = [0, 0, 0, 0, 0]
        self.attr_cnt = [0 if i in self.tasks else 1 for i in range(len(self.attr_len))]

        loss_sum = 0
        for i, batch in enumerate(data_loader):
            t0 = time.clock()
            self.step = i+1
            f_logit = None
            for t_idx, _ in enumerate(self.task_sepa):
                model = self.model[t_idx]

                # change the mode
                if trainable:
                    model.train()
                    model.optimizer.zero_grad()
                else:
                    model.eval()

                start = 0
                delete_idx = []
                for a_idx, al in enumerate(self.attr_len):
                    end = start + al
                    if not a_idx in model.tasks:
                        delete_idx.extend(list(range(start, end)))
                    start += al
                onehot = np.delete(batch[2], delete_idx, 1)
                observed = np.delete(batch[3], delete_idx, 1)

                logit, loss = model((batch[0], batch[1], onehot, observed), sample_type)

                if trainable:
                    loss.backward()
                    nn.utils.clip_grad_norm(model.parameters(), self.args.grad_max_norm)
                    model.optimizer.step()
                ls = loss.data.cpu().numpy()
                loss_sum += ls[0]

                if f_logit is None:
                    f_logit = logit
                else:
                    f_logit = np.concatenate((f_logit, logit), 1)

            self.accumulate_score(f_logit, batch[2], batch[3], self.tasks, trainable, sample_type)

            if (i+1) % self.args.print_per_step == 0:
                hm, p, r, f1 = self.get_score()
                t1 = time.clock()
                self.logger.info("<step {}> Loss={:5.3f}, time:{:5.2f}, Hamming={:2.3f}, P:{:2.3f}, R:{:2.3f}, F1:{:2.3f}"
                                    .format(self.step, loss_sum/(self.step), t1-t0, hm, p, r, f1))
                self.logger.info("Accuracy - gender:{:3.1f}, marital:{:3.1f}, age:{:3.1f}, income:{:3.1f}, edu:{:3.1f}"
                                    .format(100*self.attr_em[0]/self.attr_cnt[0],
                                            100*self.attr_em[1]/self.attr_cnt[1],
                                            100*self.attr_em[2]/self.attr_cnt[2],
                                            100*self.attr_em[3]/self.attr_cnt[3],
                                            100*self.attr_em[4]/self.attr_cnt[4]))
        hm, p, r, f1 = self.get_score()
        return loss_sum / num_steps, hm, p, r, f1

    def accumulate_score(self, logit, onehot, observed, tasks, trainable, sample_type):
        if not self.args.partial_eval: observed = np.zeros_like(observed)

        y_pred, y_true = [],[]
        y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
                                for i, oh in enumerate(onehot)])

        for b_idx, ob in enumerate(observed):
            pred, true = [],[]
            start = 0
            for a_idx, al in enumerate(self.attr_len):
                if not a_idx in tasks: continue

                end = start + al
                # eval for observe attr
                if sum(ob[start:end]):
                    self.attr_cnt[a_idx] += 1
                    #if trainable and a_idx!=sample_type: continue
                    #p = np.argmax(logit[b_idx][start:end] - th[a_idx]) + start
                    p = np.argmax(logit[b_idx][start:end], 0) + start
                    t = sum(y_numbering[b_idx][start:end])
                    if p == t:
                        self.attr_em[a_idx] += 1
                    pred.append(p)
                    true.append(t)
                #else:
                #    p = sum(y_numbering[b_idx][start:end])
                #    t = sum(y_numbering[b_idx][start:end])
                #    pred.append(p)
                #    true.append(t)
                start += al
            if pred and true:
                y_pred.append(pred)
                y_true.append(true)

        self.num_users += len(y_true)

        for yp in y_pred:
            for p in yp:
                self.yp_counter[p] += 1
        for yp in y_true:
            for p in yp:
                self.yt_counter[p] += 1

        for y in zip(y_pred, y_true):
            self.y_counter[str(y[1])] += 1
            if np.array_equal(y[0], y[1]):
                self.y_em_counter[str(y[1])] += 1
                # count exact matchings for evaluating wP, wR, wF1
                self.em += 1
            # calculate and accumulate hamming loss
            self.hm_acc += hamming_loss(y[1], y[0])

    def get_score(self):
        hm_loss = self.hm_acc / self.num_users
        wP = 0
        for y, cnt in self.y_counter.items():
            wP += self.y_em_counter[y] / cnt
        ## for debugging
        #if self.step == self.num_steps:
        #    for i in range(0, 18):
        #        print('{} : y-pred / y-true : {}, {}'
        #                .format(i, self.yp_counter[i], self.yt_counter[i]))
        #    print(len(self.y_em_counter), len(self.y_counter), wP / len(self.y_em_counter))
        ##
        wP /= len(self.y_counter)
        wR = self.em / self.num_users
        if wP == 0 and wR == 0:
            wP = wR = wF1 = 0
        else:
            wF1 = (2 * wP * wR) / (wP + wR)
        return hm_loss, wP, wR, wF1
