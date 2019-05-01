import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import numpy as np
import sys

from .functions import draw_neg_sample, compute_loss

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=99999)

class AvgPooling(nn.Module):
    def __init__(self, logger, model_type, len_dict, item_emb_size, attr_len, no_cuda):
        super(AvgPooling, self).__init__()
        
        self.model_type = model_type
        self.logger = logger
        self.embed_dim = item_emb_size
        self.attr_len = attr_len
        self.optimizer = None
        self.no_cuda = no_cuda
        
        if self.model_type == 'JNE':
            self.structured_output = False
            self.use_negsample = False
        elif self.model_type == 'SNE':
            self.structured_output = True
            self.use_negsample = True
        else:
            sys.exit()

        self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)

        # choose a learning method
        if self.structured_output:
            label_size = sum([al for i, al in enumerate(attr_len)])
            self.W = nn.Linear(item_emb_size, label_size, bias=True)
        else:
            self.W_all = nn.ModuleList()
            for i, al in enumerate(attr_len):
                self.W_all.append(nn.Linear(item_emb_size, attr_len[i], bias=True))


    def forward(self, x, x_mask, y, ob, trainable=False):
        y = torch.from_numpy(y).float()
        ob = torch.from_numpy(ob).float()
        if torch.cuda.is_available() and not self.no_cuda:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y = y.cuda()
            ob = ob.cuda()
        x_len = torch.sum(x_mask.long(), 1)

        # represent items
        embed = self.item_emb(x)
        
        # represent users
        user_rep = []
        for j, emb in enumerate(embed):
            user_rep.append(torch.sum(emb, 0)/x_len[j].float())
        user_rep = torch.stack(user_rep, 0)


        # masking to distinguish between known and unknown attributes
        if self.structured_output: 
            # SNE (share emb, structured)
            W_user = self.W(user_rep)
        else: # JNE
            for i, W in enumerate(self.W_all):
                if i == 0:
                    W_user = W(user_rep)
                else:
                    W_user = torch.cat((W_user, W(user_rep)), 1)
        
        W_compact = W_user * ob
        y_c = y * ob

        if self.use_negsample:
            # using negative sampling for efficient optimization
            neg_samples = draw_neg_sample(x.size(0), self.attr_len, y, ob)
            neg_logs = []
            for idx, w_c in enumerate(W_compact):
                neg = neg_samples[idx].cuda()
                neg_logs.append(torch.sigmoid(-(neg*w_c).sum(0)).log())
            neg_loss = torch.stack(neg_logs).sum(0)
            pos_loss = torch.sigmoid((W_compact*y_c).sum(1)).log().sum(0)
            loss = (-torch.sum(pos_loss+neg_loss)/W_compact.size(0))
            logit = W_user.data.cpu().numpy()
        else:
            # all attr are observed in new-user prediction
            loss = 0
            s = e = 0
            for i, t in enumerate(self.attr_len):
                e += t
                lg, ls = compute_loss(W_user, y, ob, s, e, self.no_cuda)
                loss += ls
                if i == 0:
                    logit = lg
                else:
                    logit = np.concatenate((logit, lg), 1)
                s = e
        
        return logit, loss


