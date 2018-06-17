import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import Counter
from functools import reduce
import numpy as np
from random import randint
import sys
import time

class DemoPredictor(nn.Module):
    def __init__(self, logger, len_dict,
                item_emb_size, attr_len, num_negs, user_rep,
                rnn_type, rnn_size, rnn_layer, rnn_drop,
                learning_form, partial_training, tasks=[0,1,2,3,4]):
        super(DemoPredictor, self).__init__()
        self.attr_len = attr_len
        self.cum_len = np.concatenate(([0], np.cumsum(np.asarray(attr_len)[tasks])))
        self.num_negs = num_negs
        self.user_rep = user_rep
        self.learning_form = learning_form
        self.partial_training = partial_training
        self.tasks = tasks
        self.optimizer = None

        label_size = sum([al for i, al in enumerate(attr_len) if i in tasks])

        self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)
        #self.init_item_emb_weight(glove_mat)

        # choose the way how represent users given the histories of them
        if user_rep == "Average":
            user_size = item_emb_size
        elif user_rep == "RNN":
            self.history_encoder = getattr(nn, rnn_type)(
                                        input_size=item_emb_size,
                                        hidden_size=rnn_size,
                                        num_layers=rnn_layer,
                                        bias=True,
                                        batch_first=True,
                                        dropout=rnn_drop,
                                        bidirectional=False)
            user_size = rnn_size

        # choose a learning method
        if learning_form == "seperated":
            self.W_all = nn.ModuleList()
            for i, al in enumerate(attr_len):
                if i in tasks:
                    self.W_all.append(nn.Linear(user_size, attr_len[i], bias=False))
        elif learning_form == "structured":
            self.W = nn.Linear(user_size, label_size, bias=False)

        # generate all the possible structured vectors
        all_attr = []
        for num_class in attr_len:
            all_class = [[1 if i==j else 0 for j in range(num_class)] for i in range(num_class)]
            all_attr.append(all_class)

        def combinate(list1, list2):
            out = []
            for l1 in list1:
                for l2 in list2:
                    out.append(l1+l2)
            return out

        self.all_possible = Variable(torch.from_numpy(np.asarray(
                                reduce(combinate, all_attr)))).float().cuda()


    def init_item_emb_weight(self, glove_mat):
        glove_mat = torch.from_numpy(glove_mat).cuda()
        self.glove_emb.weight.data = glove_mat
        self.glove_emb.weight.requires_grad = True

    def draw_sample(self, batch_size, label):
        # weight [batch, all_posible]
        # find label index
        labels = label.cpu().data.numpy()
        np_all_possible = self.all_possible.cpu().data.numpy()
        target_idx = []
        for label in labels:
            target_idx.append(np.where((np_all_possible == label).all(axis=1))[0][0])

        # sampling based on uniform weight
        weight = torch.FloatTensor(batch_size, self.all_possible.size(0)).uniform_(0, 1)
        # sample index [batch, num_neg]
        sample_idx = torch.multinomial(weight, self.num_negs).numpy()

        # check if target idx included in sample
        for i, sample in enumerate(sample_idx):
            while target_idx[i] in sample:
                sample[np.where(sample== target_idx[i])] = randint(0, self.all_possible.size(0)-1)
        sample_idx = Variable(torch.from_numpy(sample_idx.astype(int))).cuda()
        neg_samples = []
        for sample in sample_idx:
            neg_samples.append(self.all_possible[sample].unsqueeze(0))
        
        return torch.cat(neg_samples, 0)

    def forward(self, batch):
        x, x_mask, y, ob = batch
        x = Variable(x).cuda()
        x_mask = Variable(x_mask).cuda()
        y = Variable(torch.from_numpy(y)).cuda().float()
        x_len = torch.sum(x_mask.long(), 1)
        ob = Variable(torch.from_numpy(ob)).cuda().float()
        
        # represent items
        embed = self.item_emb(x)
        
        # get negative samples
        #neg_samples = self.draw_sample(x.size(0), y)
        
        # represent users
        if self.user_rep == 'Average':
            user_rep = []
            for i, emb in enumerate(embed):
                user_rep.append(torch.sum(emb, 0)/x_len[i].float())
            user_rep = torch.stack(user_rep, 0)
        elif self.user_rep == 'RNN':
            rnn_out, _ = self.history_encoder(embed)
            bg = Variable(torch.arange(0, rnn_out.size(0)*rnn_out.size(1), rnn_out.size(1))).long().cuda()
            x_idx = bg + x_len -1
            user_rep = rnn_out.contiguous().view(-1, rnn_out.size(-1))\
                            .index_select(dim=0, index=x_idx)
        
        # mask to unknown attributes in training
        if self.learning_form == 'seperated':
            for i, W in enumerate(self.W_all):
                if i == 0:
                    W_user = W(user_rep)
                else:
                    W_user = torch.cat((W_user, W(user_rep)), 1)
            
            def compute_loss(WU, full_label, observed, start, end, weight=None):
                W_user = WU.transpose(1,0)[start:end].transpose(1,0)
                y = full_label.transpose(1,0)[start:end].transpose(1,0)
                ob = observed.transpose(1,0)[start:end].transpose(1,0)
                if not self.partial_training:
                    ob = Variable(torch.ones(ob.size())).float().cuda()
                
                W_compact = W_user * ob
                c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]
                if c_idx:
                    c_idx = Variable(torch.from_numpy(np.asarray(c_idx))).long().cuda()
                    W_compact = torch.index_select(W_compact, 0, c_idx)
                    y_c = torch.index_select(y, 0, c_idx) 
                    all_possible = [[1 if i==j else 0 for j in range(end-start)] \
                                    for i in range(end-start)]
                    all_possible = Variable(torch.from_numpy(np.asarray(
                                        all_possible))).float().cuda()
                    denom = 0
                    for case in all_possible:
                        denom += torch.sum(W_compact*case, 1).exp()
                    obj = torch.sum(W_compact*y_c, 1).exp() / denom

                    if weight is not None:
                        weighted = torch.sum(y_c * weight, 1)
                        loss = -torch.sum(obj.log()*weighted)
                    else:
                        loss = -torch.sum(obj.log())
                    batch_size = y_c.size(0)
                else: 
                    loss = 0
                    batch_size = 1
                logit = W_user.data.cpu().numpy()
                logit = F.softmax(W_user).data.cpu().numpy()
                return logit, loss / batch_size
            
            loss = 0
            for i, t in enumerate(self.tasks):
                #if t == 0:
                #    weight = Variable(torch.from_numpy(np.asarray([1, 2]))).float().cuda()
                #else: weight = None
                weight = None
                lg, ls = compute_loss(W_user, y, ob, self.cum_len[i], self.cum_len[i+1], weight)
                loss += ls
                if i == 0:
                    logit = lg
                else:
                    logit = np.concatenate((logit, lg), 1)
        elif self.learning_form == 'structured':
            W_user = self.W(user_rep)
            W_compact = W_user * ob
            
            # we use negative sampling for efficient optimization
            neg_logs = []
            for idx, w_c in enumerate(W_compact):
                neg = neg_samples[idx]
                neg_logs.append(F.sigmoid(-(neg*w_c)).log().sum().unsqueeze(0))

            neg_loss = torch.sum(torch.cat(neg_logs), 1)
            pos_loss = torch.sum(torch.log(F.sigmoid(W_compact*y)), 1)
            loss = -torch.sum(pos_loss+neg_loss)/W_compact.size(0)
            logit = W_user.data.cpu().numpy()
        return logit, loss


