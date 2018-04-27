
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

from model import DemoPredictor

class Experiment:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.logger.info("\n*** Experiment initializing . . . ***")
        self.model = DemoPredictor(args, logger).cuda()
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
        for i, batch in enumerate(data_loader):
            t0 = time.clock()
            if trainable: 
                self.optimizer.zero_grad()
            logit = self.model(batch)
            if (loss is None): continue
            if trainable:
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_max_norm)
                self.optimizer.step()
            ls = loss.data.cpu().numpy()
            loss_sum += ls[0]
        return r



