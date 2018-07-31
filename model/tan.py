import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import pickle
from collections import Counter
from functools import reduce
import numpy as np
from random import randint
import sys
import time

from .common import combinate, draw_neg_sample

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=5000)

class TANDemoPredictor(nn.Module):
    def __init__(self, logger, len_dict, item_emb_size,
                attention_layer, attr_len, learning_form,
                use_negsample, partial_training, uniq_input, tasks=[0,1,2]):
        super(TANDemoPredictor, self).__init__()

        self.logger = logger
        self.attr_len = attr_len
        self.cum_len = np.concatenate(([0], np.cumsum(np.asarray(attr_len)[tasks])))
        self.use_negsample = use_negsample
        self.attention_layer = attention_layer
        self.partial_training = partial_training
        self.learning_form = learning_form
        self.uniq_input = uniq_input
        self.tasks = tasks
        self.optimizer = None
        label_size = sum([al for i, al in enumerate(attr_len) if i in tasks])
        self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)
        with open('./data/preprd/ocb/idx2brand.pkl', 'rb') as f:
            self.brand2idx = pickle.load(f)

        user_size = item_emb_size

        # choose the way to represent users given the histories of them
        if attention_layer == 1:
            self.item_att_W = nn.ModuleList([nn.Linear(item_emb_size, 1) for i in range(3)])

        elif attention_layer == 2 and learning_form=='structured':
            self.item_att_W = nn.ModuleList([nn.Linear(item_emb_size, 1) for i in range(3)])
            self.attr_att_W = nn.Linear(item_emb_size,1)

        else: # attention_layer == 2 and learning_form=='seperated'
            self.item_att_W = nn.ModuleList([nn.Linear(item_emb_size, 1) for i in range(3)])
            self.attr_att_W = nn.ModuleList([nn.Linear(item_emb_size, 1) for i in range(3)])

        # appropriate prediction layer for output type
        if learning_form == 'seperated':
            self.W_all = nn.ModuleList()
            for i, al in enumerate(attr_len):
                if i in tasks:
                    self.W_all.append(nn.Linear(user_size*3, attr_len[i], bias=False))
        else:
            # structured prediction
            self.W = nn.Linear(user_size*3, label_size, bias=False)

        # generate all the possible structured vectors
        all_attr = []
        for num_class in attr_len:
            all_class = [[1 if i==j else 0 for j in range(num_class)] for i in range(num_class)]
            all_attr.append(all_class)

        self.all_possible = Variable(torch.from_numpy(np.asarray(
                                reduce(combinate, all_attr)))).float().cuda()


    def init_item_emb_weight(self, glove_mat):
        glove_mat = torch.from_numpy(glove_mat).cuda()
        self.glove_emb.weight.data = glove_mat
        self.glove_emb.weight.requires_grad = True


    def visualize(self, brand, att_scores, num_purchase, labels, logits):
        def to_onehot(logit, label):
            attr_len = [0,2,6,8]
            pred2gender = ['male', 'female']
            pred2age = ['young', 'adult', 'middle-age', 'old']
            pred2marital = ['married', 'single']

            pred, true = [], []
            for attr_idx in range(len(attr_len)-1):
                p = np.argmax(logit[attr_len[attr_idx]:attr_len[attr_idx+1]])
                t = np.argmax(label[attr_len[attr_idx]:attr_len[attr_idx+1]])
                if attr_idx == 0 :
                    p = pred2gender[p]
                    t = pred2gender[t]
                elif attr_idx == 1 :
                    p = pred2age[p]
                    t = pred2age[t]
                else :
                    p = pred2marital[p]
                    t = pred2marital[t]
                pred.append(p)
                true.append(t)
            return pred, true

        if len(att_scores) == 3:
            attr_type = ['gender', 'age', 'marital']
            for type_idx, typ in enumerate(attr_type):
                wf = open("./save/att_vis/att_vis_{}_{}.tsv".format(typ,
                time.strftime("%H%M", time.gmtime())), 'w')
                brand = brand.type(torch.cuda.FloatTensor)
                att_score = att_scores[type_idx].squeeze()
                for i, cnt in enumerate(num_purchase):
                    logit = logits[i]
                    label = labels[i]
                    pred, true = to_onehot(logit, label)

                    cnt = cnt.item()
                    brand_list = brand[i][:cnt].data.cpu().numpy().astype(int).tolist()
                    brand_name_list = [self.brand2idx[brand_idx] for brand_idx in brand_list]
                    att = att_score[i].data.cpu().numpy() * 100
                    att = att.tolist()[:cnt]
                    str_tmp = [str(true), str(pred),
                               ' '.join(str(e) for e in brand_name_list),
                               ' '.join(str(e) for e in att)]
                    str_tmp = '\t'.join(str_tmp) + '\n'
                    wf.write(str_tmp)
                wf.close()

    def forward(self, process, batch, trainable=False):

        def get_attention(w, emb):
            att_u = F.tanh(w(embed))
            att_score = F.softmax(att_u,1)
            attnd_emb = embed * att_score
            rep = F.relu(torch.sum(attnd_emb, 1))
            return rep, att_score

        def item_attention(embed):
            # embed : [B,K,emb] --> att_u [B,K,1] for each attribute
            batch = embed.size(0)
            attr_rep = []
            att_scores = []
            for attr_w in self.item_att_W:
                rep, att = get_attention(attr_w, embed)
                attr_rep.append(rep.unsqueeze(2))
                att_scores.append(att)
            # user_rep : [B, 3(num attr), emb]
            user_rep = torch.cat(attr_rep, 2).view(batch,-1)

            return user_rep, att_scores

        def attr_attention(embed, attr_att_W):
            batch = embed.size(0)
            if self.learning_form=='seperated':
                user_rep = []
                for attr_w in attr_att_W:
                    user_rep.append(get_attention(attr_w, embed).unsqueeze(2))
                user_rep = torch.cat(user_rep, 2).view(batch,-1)
            else:
                user_rep = get_attention(attr_att_W, embed)
            return user_rep


        x, x_mask, x_uniq, x_uniq_mask, y, ob = batch
        epoch, step = process
        x = Variable(x).cuda()
        x_mask = Variable(x_mask).cuda()
        x_uniq = Variable(x_uniq).cuda()
        x_uniq_mask = Variable(x_uniq_mask).cuda()
        y = Variable(torch.from_numpy(y)).cuda().float()
        x_len = torch.sum(x_mask.long(), 1)
        ob = Variable(torch.from_numpy(ob)).cuda().float()
        # get negative samples
        #neg_samples = draw_neg_sample(x.size(0), self.attr_len, y, ob)
        # represent items
        embed = self.item_emb(x)
        uniq_emb = self.item_emb(x_uniq)
        # get negative samples
        #neg_samples = self.draw_sample(x.size(0), y)

        if self.uniq_input:
            user_rep, att_scores = item_attention(uniq_emb)
            num_purchase = torch.sum((x_uniq!=0), 1)
        else:
            user_rep, att_scores = item_attention(embed)
            num_purchase = torch.sum((x!=0), 1)
        #W_compact = W_user * ob

        if self.attention_layer==2:
            # with item level & attr self attention
            # attr_rep : [B, K, 5(num attr)]
            attr_rep = user_rep
            # get self attended attribute vector
            # user_attr_rep : [B, emb] * 5
            user_rep = attr_attention(attr_rep, self.attr_att_W)

        # appropriate prediction layer for each output type
        if self.learning_form == 'structured':
            W_user = self.W(user_rep)

        else:
            '''
            user_attr_rep = user_rep.transpose(0, 1)
            for i, W in enumerate(self.W_all):
                if i == 0:
                    W_user = W(user_attr_rep[i])
                else:
                    W_user = torch.cat((W_user, W(user_attr_rep[i])), 1)
            '''
            for i, W in enumerate(self.W_all):
                if i == 0:
                    W_user = W(user_rep)
                else:
                    W_user = torch.cat((W_user, W(user_rep)), 1)
        # masking to distinguish between known and unknown attributes
        #W_compact = W_user * ob
        #y_c = y * ob

        # all attr are observed in new-user prediction
        def compute_loss(WU, full_label, observed, start, end, weight=None):
            W_user = WU.transpose(1,0)[start:end].transpose(1,0)
            y = full_label.transpose(1,0)[start:end].transpose(1,0)
            ob = observed.transpose(1,0)[start:end].transpose(1,0)
            # change all observe for new_user
            if not self.partial_training:
                ob = (torch.ones(ob.size())).float().cuda()

            W_compact = W_user * ob

            c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]

            if c_idx:
                c_idx = (torch.from_numpy(np.asarray(c_idx))).long().cuda()
                W_compact = torch.index_select(W_compact, 0, c_idx)
                y_c = torch.index_select(y, 0, c_idx)
                all_possible = [[1 if i==j else 0 for j in range(end-start)] \
                                for i in range(end-start)]
                all_possible = (torch.from_numpy(np.asarray(
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
                loss = torch.tensor(0, requires_grad=True).float().cuda()
                batch_size = 1
            logit = W_user.data.cpu().numpy()
            logit = F.softmax(W_user, dim=1).data.cpu().numpy()
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

        if not trainable:
            if self.uniq_input:
                self.visualize(x_uniq, att_scores, num_purchase, y, logit)
            else:
                self.visualize(x, att_scores, num_purchase, y, logit)

        return logit, loss
