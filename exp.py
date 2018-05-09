
import argparse
import copy
from itertools import chain
import logging
import numpy as np
import os
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_fscore_support as f1_score
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
                        args.data_path+'dict.json')
        self.dict = Dict.dict
        self.attr_len = Dict.attr_len
        self.logger.info("Experiment initializing . . . ")
        self.model = DemoPredictor(
                        logger, self.dict.__len__(),
                        args.item_emb_size, label_size, Dict.attr_len,
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
            .format(data_loader.dataset.data_type, num_samples, num_steps))
        
        # change the mode
        if trainable:
            self.model.train()
        else:
            self.model.eval()
        
        # step training or evaluation with given batch size
        loss_sum = hm_sum = p_sum = r_sum = f1_sum = 0
        for i, batch in enumerate(data_loader):
            print(batch)
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
        
            hm, p, r, f1 = get_score(logit, batch[2].numpy(), self.attr_len)
            hm_sum += hm
            p_sum += p
            r_sum += r
            f1_sum += f1
            
            t1 = time.clock()
            
            if (i+1) % self.args.print_per_step == 0:
                self.logger.info("<step {}> Loss={:5.3f}, time:{:5.2f}".format(i+1, ls[0], t1-t0))
                self.logger.info("Hamming={:4.2f}, P:{:4.2f}, R:{:4.2f}, F1:{:4.2f}"
                                    .format(hm, p, r, f1))

        return loss_sum / num_steps, \
                hm_sum / num_steps, \
                p_sum / num_steps, \
                r_sum / num_steps, \
                f1_sum / num_steps

def get_score(logit, onehot, attr_len):
    start = 0
    pred = []
    logit = logit.transpose(1,0)
    for al in attr_len:
        end = start + al
        pred.append(np.argmax(logit[start:end], 0) + start)
        start += al
    
    y_pred = np.asarray(pred).transpose(1,0)
    y_pred = list(chain.from_iterable(
            [yp + (i*sum(attr_len)) for i, yp in enumerate(y_pred)]))
    y_true = list(chain.from_iterable(
            [[j + (i*sum(attr_len)) for j, l in enumerate(oh) if l]  \
                for i, oh in enumerate(onehot)]))
    
    hm_loss = hamming_loss(y_true, y_pred)
    p, r, f1, _ = f1_score(y_true, y_pred, average='weighted')

    return hm_loss, p, r, f1
