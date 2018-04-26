
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

#from model import Model

class Experiment:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.logger.info("\n*** Experiment initializing . . . ***")
        self.model = Model(args, logger)
        self.select_optimizer()
        
        self.criterion = nn.NLLLoss(size_average=size_average, reduce=reduce)
        
        self.model = self.model.cuda()
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

    def run_epoch(self, data, epoch, trainable=False):
        if not trainable: 
            self.model.eval()
        else: 
            self.model.train()
        
        if self.args.save_output and not trainable: # analysis mode
            num_steps = data.num_data
            if epoch > 0: 
                model_analysis = True
                output_txt = open("./analysis/model_analysis/" + \
                                self.logger.name + \
                                '(ep:' + str(epoch) + ').txt', 'w')
            else: 
                model_analysis = False
                output_txt = None
        else:
            num_steps = len(data.batch_idx)
            model_analysis = False
            output_txt = None

        self.TP = self.FP = self.FN = 0
        self.bi_TP = self.bi_FP = self.bi_FN = 0
        self.id_TP = self.id_FP = self.id_FN = 0

        loss_sum = acc_sum = recall_sum = prec_sum = f1_sum = 0.
        for i in range(num_steps):
            ### for debugging
            #if trainable and 100 == 1: break
            #if not trainable and i == 2: break
            ###
            t0 = time.clock()
            batch = data.get_batch(i)
            if trainable: 
                self.optimizer.zero_grad()
                label_s = np.array([])
            else:
                label_s = batch['label_s']
            if model_analysis: # analysis mode
                input_text = ""
                for s in batch['context']:
                    for w in s:
                        w = w.data.cpu().numpy()
                        input_text += data.word_dict[w[0]]
                        input_text += " "
                output_txt.write("********************\n")
                output_txt.write("text : "+input_text+'\n')
                syn_doc = []
                for d in batch['synopsis']:
                    for p in d:
                        syn_par = ""
                        for w in p:
                            w = w.data.cpu().numpy()
                            syn_par += data.word_dict[w[0]]
                            syn_par += " "
                        syn_doc.append(syn_par)
                output_txt.write("synopsis : "+str(syn_doc)+'\n')
                output_txt.write("label : " + \
                        str(batch['label'].data.cpu().numpy().reshape(-1).tolist())+'\n')
            span_logit, log_likeli, viterbi_seq, mask = self.model(batch, 
                                                            model_analysis,
                                                            output_txt)
            if self.args.loss == 'word':
                num_of_words = np.sum(batch['len_context']).astype(float)
                #if not trainable:
                    #print(batch['context'])
                    #print('span logit : ', span_logit[0])
                    #print('num of words : ', num_of_words)
                logsoft = torch.log(F.softmax(span_logit, -1))
                if self.args.tf_loss_scale:
                    loss_w0 = torch.ones(logsoft.size()[0], logsoft.size()[1], 1) * 1
                    loss_w1 = torch.ones(logsoft.size()[0], logsoft.size()[1], 1) * 5
                    loss_w = torch.cat((loss_w0, loss_w1), -1)
                    logsoft = logsoft * Variable(loss_w).cuda()
                span_loss = self.criterion(logsoft.view(-1, self.args.num_tags), batch['label'].view(-1))
                #print('span logit2 : ', span_logit)
                span_loss = torch.mul(span_loss, mask.view(-1))
                #print('span logit3 : ', span_logit)
                span_loss = (torch.sum(span_loss) / num_of_words)
                #print('span logit4 : ', span_logit)
            else:
                span_loss = self.criterion(span_logit.view(-1, self.args.num_tags), batch['label'].view(-1))
            if self.args.use_crf:
                log_likeli = torch.mul(log_likeli, -self.args.crf_loss_weight)
                loss = torch.add(span_loss, 1, log_likeli)
            else: loss = span_loss
            
            if (loss is None): continue
            if trainable:
                #print('???')
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_max_norm)
                self.optimizer.step()

                # exponential moving average decay for variables
                """
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = self.ema(name, param.data)
                """
            ls = loss.data.cpu().numpy()
            loss_sum += ls[0]
            ###
            #if trainable: continue
            ###
            # accumulate score
            if not self.args.use_crf:
                self.accumulate_score(span_logit, batch['label'], batch['is_spoiler'],
                                        batch['len_context'], label_s)  # without CRF
                t1 = time.clock()
                # Evaluation
                if (i+1) % self.args.print_per_step == 0:
                    f1, p, r, id_f1, id_p, id_r,\
                            bi_f1, bi_p, bi_r = self.get_f1_score()
                    self.writer.add_scalar('data/f1', f1, i+1)
                    self.writer.add_scalar('data/p', p, i+1)
                    self.writer.add_scalar('data/r', r, i+1)
                    self.writer.add_scalar('data/loss', ls[0], i+1)
                    self.logger.info("[step {}] Loss={:5.3f}, time:{:5.2f}".format(i+1, ls[0], t1-t0))
                    self.logger.info("(word-level) F1={:4.2f}/{:4.2f}, P:{:4.2f}/{:4.2f}, R:{:4.2f}/{:4.2f}"
                            .format(f1, id_f1, p, id_p, r, id_r, t1-t0))
                    if not trainable:
                        self.logger.info("(binary) F1={:4.2f}, P:{:4.2f}, R:{:4.2f}".format(bi_f1, bi_p, bi_r))
            else:
                self.accumulate_score(viterbi_seq, batch['label'], batch['is_spoiler'],
                                        batch['len_context'], batch['label_s'])  # for CRF
                t1 = time.clock()
                # Evaluation
                if (i+1) % self.args.print_per_step == 0:
                    f1, p, r, id_f1, id_p, id_r, \
                            bi_f1, bi_p, bi_r = self.get_f1_score()
                    self.logger.info("[step {}] Loss={:5.3f}, time:{:5.2f}".format(i+1, ls[0], t1-t0))
                    self.logger.info("\t(word-level) F1={:4.2f}/{:4.2f}, P:{:4.2f}/{:4.2f}, R:{:4.2f}/{:4.2f}"
                            .format(f1, id_f1, p, id_p, r, id_r, t1-t0))
                    if not trainable:
                        self.logger.info("\t(binary) F1={:4.2f}, P:{:4.2f}, R:{:4.2f}".format(bi_f1, bi_p, bi_r))
        
        if model_analysis: output_txt.close()
        #if trainable: return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        f1, p, r, id_f1, id_p, id_r, \
                bi_f1, bi_p, bi_r = self.get_f1_score()
        print('TP, FP, FN : ', self.TP, self.FP, self.FN)
        print('(bi) TP, FP, FN : ', self.bi_TP, self.bi_FP, self.bi_FN)
        return loss_sum/num_steps, f1, p, r, id_f1, id_p, id_r, bi_f1, bi_p, bi_r


    def accumulate_score(self, logit, label, is_spoiler, len_context, label_s):
        
        label = label.data.cpu().numpy()
        is_spoiler = is_spoiler.data.cpu().numpy()
        if not self.args.use_crf:
            logit = F.softmax(logit, -1)
            logit = logit.data.cpu().numpy()
            pred = np.argmax(logit, 2)  # without CRF
            th_pred = np.greater(np.transpose(
                            logit, (2,1,0))[1], self.args.class_th)
            pred = np.logical_or(pred, np.transpose(th_pred, (1,0))).astype(int)
            #print('pred : ', pred)
        else: pred = logit
        

        # masking the predictions of the model
        #print('len context : ', len_context)
        mask = np.tile(np.expand_dims(len_context, 1),
                        (1, pred.shape[1]))
        arange = np.tile(np.expand_dims(np.arange(pred.shape[1]), 0),
                        (pred.shape[0], 1))
        mask = np.greater(mask, arange)
        #print('mask : ', mask)
        id_pred_P = np.tile(np.expand_dims(is_spoiler, 1), 
                            (1, label.shape[1]))
        pred = pred * mask
        #print(pred)
        
        ###
        #sys.exit()
        ###
        
        # ideal word-level f1
        id_pred_P = id_pred_P * mask
        id_pred_N = np.equal(id_pred_P, 0)
        
        # word-level f1
        pred_P = np.greater(pred, 0)
        label_T = np.greater(label, 0)
        pred_N = np.equal(pred, 0)
        label_F = np.equal(label, 0)
        
        # binary f1
        if not label_s.shape[0]:
            bi_pred_P = 0
            bi_pred_N = 0
            bi_label_T = 0
            bi_label_F = 0
        else:
            len_sen = []
            for c in label_s:
                tmp = []
                for s in c:
                    tmp.append(len(s))
                len_sen.append(tmp)
            
            bi_pred_P, bi_pred_N = [],[]
            bi_label_T, bi_label_F = [],[]
            for p_idx, p in enumerate(pred):
                start = end = 0
                for ls in len_sen[p_idx]:
                    end = start + ls
                    if np.amax(p[start:end]) == 1:
                        bi_pred_P.append(1)
                        bi_pred_N.append(0)
                    else:
                        bi_pred_P.append(0)
                        bi_pred_N.append(1)
                    if np.amax(label[p_idx][start:end]) == 1:
                        bi_label_T.append(1)
                        bi_label_F.append(0)
                    else:
                        bi_label_T.append(0)
                        bi_label_F.append(1)
                    start += ls
            bi_pred_P = np.asarray(bi_pred_P)
            bi_pred_N = np.asarray(bi_pred_N)
            bi_label_T = np.asarray(bi_label_T)
            bi_label_F = np.asarray(bi_label_F)
        
        # accumulate true positive, false positive, and false negative
        self.TP += np.sum(label_T * pred_P)
        self.FP += np.sum(label_F * pred_P)
        self.FN += np.sum(label_T * pred_N)
        self.id_TP += np.sum(label_T * id_pred_P)
        self.id_FP += np.sum(label_F * id_pred_P)
        self.id_FN += np.sum(label_T * id_pred_N)
        self.bi_TP += np.sum(bi_label_T * bi_pred_P)
        self.bi_FP += np.sum(bi_label_F * bi_pred_P)
        self.bi_FN += np.sum(bi_label_T * bi_pred_N)


    def get_f1_score(self):
        f1, p, r = self.calculate_f1(self.TP, self.FP, self.FN)
        id_f1, id_p, id_r = self.calculate_f1(self.id_TP, self.id_FP, self.id_FN)
        bi_f1, bi_p, bi_r = self.calculate_f1(self.bi_TP, self.bi_FP, self.bi_FN)
        return f1, p, r, id_f1, id_p, id_r, bi_f1, bi_p, bi_r

    def calculate_f1(self, TP, FP, FN):
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f1 = (2*p*r) / (p + r)
        return f1*100, p*100, r*100


