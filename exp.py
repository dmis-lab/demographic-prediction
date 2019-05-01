
import argparse
import copy
from collections import Counter
from functools import reduce
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
from model.baselines import AvgPooling
from model.etna import ETNADemoPredictor


class Experiment:
    def __init__(self, args, logger):

        self.args = args
        self.logger = logger
        Dict = Dictionary(
                    data_path=os.path.join(args.data_path, args.dataset),
                    task_type=args.task_type)
        self.dict = Dict.dict
        self.attr_len = Dict.attr_len
        self.all_the_poss = reduce(mul, Dict.attr_len, 1)
        self.logger.info("Experiment initializing . . . ")

        # build models
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if args.model_type == 'POP':
            self.model = 'POP'
        elif any([True if args.model_type == m else False for m in ['JNE', 'SNE']]):
            self.model = AvgPooling(logger, args.model_type, self.dict.__len__(), 
                        args.item_emb_size, Dict.attr_len, args.no_cuda).to(device)
        elif any([True if args.model_type == m else False for m in ['ETN', 'ETNA']]):
            self.model = ETNADemoPredictor(logger, args.model_type, self.dict.__len__(), 
                    args.item_emb_size, Dict.attr_len, args.no_cuda).to(device)
        else:
            sys.exit()

        if args.model_type != 'POP':
            self.select_optimizer(self.model)
        self.logger.info(self.model)
        self.step_count = 0


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
                                        weight_decay=self.args.weight_decay,
                                        momentum=self.args.momentum)
        elif(self.args.opt == 'Adagrad'):
            model.optimizer =  optim.Adagrad(parameters, lr=self.args.learning_rate)
        elif(self.args.opt == 'Adadelta'):
            model.optimizer =  optim.Adadelta(parameters, lr=self.args.learning_rate)

    def adjust_lr(self):
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] *= self.args.lr_decay

    def run_epoch(self, epoch, data_loader, dataset, trainable=False):
        num_samples = data_loader.dataset.__len__()
        num_steps = (num_samples // self.args.batch_size) + 1
        self.num_steps = num_steps
        self.logger.info("== {} mode : {} steps for {} samples == \n"
                        .format(data_loader.dataset.data_type, num_steps, num_samples))


        self.y_em_counter, self.yp_counter, self.yt_counter = Counter(), Counter(), Counter()

        self.hm_acc = self.num_users = 0
        loss_sum = 0
        for i, (x, x_mask, y, ob) in enumerate(data_loader):
            t0 = time.clock()
            self.step = i+1
            self.step_count += 1
                
            # change the mode
            if self.args.model_type != 'POP':
                if trainable:
                    self.model.train()
                    self.model.optimizer.zero_grad()
                else:
                    self.model.eval()

                prob, loss = self.model(x, x_mask, y, ob, trainable)
                
                if trainable:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_max_norm)
                    self.model.optimizer.step()

                ls = loss.item()
                loss_sum += ls
            else:
                prob = None

            self.accumulate_score(prob, y, ob)

            if (i+1) % self.args.print_per_step == 0:
                hm, macP, macR, macF1, wP, wR, wF1 = self.get_score()
                t1 = time.clock()
                self.logger.info("< Step {} > Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
                                .format(self.step, loss_sum/self.step, t1-t0, hm))
                self.logger.info("macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
                                .format(macP, macR, macF1))
                self.logger.info("weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f}"
                                .format(wP, wR, wF1))
        
        hm, macP, macR, macF1, wP, wR, wF1 = self.get_score()
        return loss_sum / num_steps, hm, macP, macR, macF1, wP, wR, wF1


    def accumulate_score(self, prob, label, observed):
        y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(ll)] \
                                        for i, ll in enumerate(label)])

        if self.args.model_type == 'POP':
            popular = [[0, 1, 0, 1, 0, 0, 0, 1] \
                        for _ in range(y_numbering.shape[0])]
            prob = popular
        
        for b_idx, ob in enumerate(observed):
            pred, true = [],[]
            start = 0
            for a_idx, al in enumerate(self.attr_len):
                end = start + al
                if sum(ob[start:end]):
                    p = np.argmax(prob[b_idx][start:end], 0) + start
                    t = sum(y_numbering[b_idx][start:end])
                    pred.append(p)
                    true.append(t)
                start += al
            
            if pred and true:
                self.yp_counter[str(pred)] += 1
                self.yt_counter[str(true)] += 1
                if np.array_equal(pred, true):
                    self.y_em_counter[str(true)] += 1

                # calculate and accumulate hamming loss
                self.hm_acc += hamming_loss(true, pred)

                self.num_users += 1

    def get_score(self):
        # for divide-by-zero exception
        if not self.num_users: num_users = 1
        else: num_users = self.num_users

        hm_loss = self.hm_acc / num_users

        macP = macR = macF1 = wP = wR = wF1 = 0
        
        # macro and weighted Precision
        for y, cnt in self.yp_counter.items():
            if y in self.y_em_counter.keys():
                macP += (self.y_em_counter[y] / cnt)
                if y in self.yt_counter.keys():
                    wP += (self.y_em_counter[y] / cnt) * self.yt_counter[y]
        macP /= len(self.yt_counter)
        wP /= num_users

        # macro and weighted Recall
        for y, cnt in self.yt_counter.items():
            if y in self.y_em_counter.keys():
                wR += self.y_em_counter[y]
                macR += (self.y_em_counter[y] / cnt)
        macR /= len(self.yt_counter)
        wR /= num_users

        # calculate F1 using computed precision and recall.
        # this code includes exception.
        if macP == 0 and macR == 0:
            macF1 = 0
        else:
            macF1 = (2 * macP * macR) / (macP + macR)
        if wP == 0 and wR == 0:
            wF1 = 0
        else:
            wF1 = (2 * wP * wR) / (wP + wR)
        return hm_loss, macP, macR, macF1, wP, wR, wF1


