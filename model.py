import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from functools import reduce
import numpy as np
import sys
import time

class DemoPredictor(nn.Module):
    def __init__(self, logger, len_dict,
                item_emb_size, label_size, attr_len, num_negs,
                rnn_type, rnn_size, rnn_layer, rnn_drop):
        super(DemoPredictor, self).__init__()
        self.num_negs = num_negs
        self.item_emb = nn.Embedding(len_dict, item_emb_size)
        #self.init_item_emb_weight(glove_mat)

        self.history_encoder = getattr(nn, rnn_type)(
                                    input_size=item_emb_size,
                                    hidden_size=rnn_size,
                                    num_layers=rnn_layer,
                                    bias=True,
                                    batch_first=True,
                                    dropout=rnn_drop,
                                    bidirectional=False)
        self.W = nn.Linear(rnn_size, label_size, bias=False)

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

        self.all_posible = Variable(torch.from_numpy(np.asarray(
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
                sample[np.where(sample== target_idx[i])] = randint(0,384)

        sample_idx = Variable(torch.from_numpy(sample_idx.astype(int))).cuda()

        neg_samples = []
        for sample in sample_idx:
            neg_samples.append(self.all_possible[sample].unsqueeze(0))
        return torch.cat(neg_samples, 0)

    def forward(self, batch):
        x, x_mask, y, ob = batch
        x = Variable(x).cuda()
        x_mask = Variable(x_mask).cuda()
        y = Variable(y).cuda().float()
        x_len = torch.sum(x_mask.long(), 1)
        ob = Variable(ob).cuda().float()

        # represent items
        embed = self.item_emb(x)
        neg_samples = self.draw_sample(x.size(0), y)

        # represent users
        rnn_out, _ = self.history_encoder(embed)
        bg = Variable(torch.arange(0, rnn_out.size(0)*rnn_out.size(1), rnn_out.size(1))).long().cuda()
        x_idx = bg + x_len -1

        user_rep = rnn_out.contiguous().view(-1, rnn_out.size(-1))\
                        .index_select(dim=0, index=x_idx)

        # mask to unknown attributes in training
        W_user = self.W(user_rep)
        W_compact = W_user * ob
        c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]
        c_idx = Variable(torch.from_numpy(np.asarray(c_idx))).long().cuda()
        W_compact = torch.index_select(W_compact, 0, c_idx)
        y_c = torch.index_select(y, 0, c_idx)

        '''
        compute the denominator which is used for normalization. (<<- this operation was deleted.)

        old version loss calculation code
        denom = 0
        for case in self.all_posible:
            denom += torch.sum(W_user*case, 1).exp()

        obj = torch.sum(W_user*y, 1).exp() / denom
        logit = W_user.data.cpu().numpy()
        loss = -torch.sum(torch.log(obj))
        '''
        # we use negative sampling for efficient optimization
        neg_logs = []
        for idx, w_c in enumerate(W_compact):
            neg = neg_samples[idx]
            neg_logs.append(F.sigmoid(-(neg*w_c)).log().sum().unsqueeze(0))

        neg_loss = torch.sum(torch.cat(neg_logs), 1)
        pos_loss = torch.sum(torch.log(F.sigmoid(W_compact*y_c)), 1)
        loss = -torch.sum(pos_loss+neg_loss)/W_compact.size(0)

        return W_user.data.cpu().numpy(), loss
