
import argparse
import copy
import logging
import numpy as np
import os
import sys
import time
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from dataset import Dictionary
from evaluation import get_score
from model import DemoPredictor

class Experiment:
    def __init__(self, args, logger, label_size):
        self.args = args
        self.logger = logger
        Dict = Dictionary(
                        args.data_path+'dict.json')
        self.attr_len = Dict.attr_len
        self.dict = Dict.dict
        self.logger.info("Experiment initializing . . . ")
        self.model = DemoPredictor(
                        logger, self.dict.__len__(),
                        args.item_emb_size, label_size,
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

    def run_epoch(self, data_loader, epoch, trainable=False):
        # change the mode
        if trainable: 
            self.model.train()
        else: 
            self.model.eval()
       
        # step training or evaluation with given batch size
        loss_sum = []
        for i, batch in enumerate(data_loader):
            t0 = time.clock()
            if trainable: 
                self.optimizer.zero_grad()
            pred, loss = self.model(batch, self.attr_len)
            if trainable:
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_max_norm)
                self.optimizer.step()
            ls = loss.data.cpu().numpy()
            loss_sum += ls[0]
        # TODO
        hm_loss, wp, wr, wf1 = get_score(pred, y, self.attr_len)
        # eval(pred)

        return 0



