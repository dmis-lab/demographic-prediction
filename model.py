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
                item_emb_size, label_size, attr_len,
                rnn_type, rnn_size, rnn_layer, rnn_drop):
        super(DemoPredictor, self).__init__()
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

        self.all_posible = reduce(combinate, all_attr)


    def init_item_emb_weight(self, glove_mat):
        glove_mat = torch.from_numpy(glove_mat).cuda()
        self.glove_emb.weight.data = glove_mat
        self.glove_emb.weight.requires_grad = True

    def forward(self, batch):
        x, x_mask, y = batch
        x = Variable(x).cuda()
        x_mask = Variable(x_mask).cuda()
        y = Variable(y).cuda().float()
        x_len = torch.sum(x_mask, 1).long()

        # represent items
        embed = self.item_emb(x)
        
        # represent users
        rnn_out, _ = self.history_encoder(embed)
        bg = Variable(torch.arange(0, rnn_out.size(0)*rnn_out.size(1), rnn_out.size(1))).long().cuda()
        x_idx = bg + x_len -1

        user_rep = rnn_out.contiguous().view(-1, rnn_out.size(-1))\
                        .index_select(dim=0, index=x_idx)
            
        # compute the denominator which is used for normalization.
        W_user = self.W(user_rep)
        W_user_ = W_user.data.cpu().numpy()
        denom = 0
        for case in self.all_posible:
            denom += np.exp(np.dot(W_user_, case))
        
        obj = torch.sum(W_user*y, 1).exp() / Variable(torch.from_numpy(denom)).cuda().float()
        pred = W_user.data.cpu().numpy().argmax(1)
        return pred, -torch.sum(torch.log(obj))

    

