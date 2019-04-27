import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import numpy as np
import sys
import time

from .functions import compute_loss

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=5000)

class ETNADemoPredictor(nn.Module):
    def __init__(self, logger, model_type, len_dict, item_emb_size, attr_len, no_cuda):
        super(ETNADemoPredictor, self).__init__()

        #
        tasks = [0,1,2]
        #
        self.logger = logger
        self.model_type = model_type
        self.attr_len = attr_len
        self.cum_len = np.concatenate(([0], np.cumsum(np.asarray(attr_len)[tasks])))
        self.tasks = tasks
        self.item_emb_size = item_emb_size
        self.no_cuda = no_cuda
        self.optimizer = None

        label_size = sum([al for i, al in enumerate(attr_len) if i in tasks])

        self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)
        self.emb_tran = nn.ModuleList([nn.Linear(item_emb_size, item_emb_size) for i in range(len(attr_len))])
        
        if model_type == 'ETNA':
            self.item_att_W = nn.ModuleList([nn.Linear(item_emb_size, 1) for i in range(len(attr_len))])

        # appropriate prediction layer for output type
        self.W_all = nn.ModuleList()
        for i, al in enumerate(attr_len):
            if i in tasks:
                self.W_all.append(nn.Linear(item_emb_size, attr_len[i], bias=False))

    
    def forward(self, x, x_mask, y, ob, trainable=False):

        def get_attention(w, embed, len):
            # Attention score with non linear
            att_u = F.relu(w(embed).squeeze(2))
            
            att_score = torch.zeros(att_u.size())
            if torch.cuda.is_available() and not self.no_cuda:
                att_score = att_score.cuda()
            
            for i, l in enumerate(len):
                candi = att_u[i][:l]
                a = F.softmax(candi, 0)
                att_score[i][:l] = a

            attnd_emb = embed * att_score.unsqueeze(2)

            rep = torch.tanh(torch.sum(attnd_emb, 1))
            return rep, att_score

        def item_attention(embed, share_emb, len):
            # embed : [B,K,emb] --> att_u [B,K,1] for each attribute
            batch = embed[0].size(0)
            attr_rep = []
            att_scores = []

            for i, attr_w in enumerate(self.item_att_W):
                rep, att = get_attention(attr_w, embed[i], len)
                attr_rep.append(rep.unsqueeze(2))
                att_scores.append(att)
            # user_rep : [B, 3(num attr), emb]
            user_rep = torch.cat(attr_rep, 2).view(batch,-1)

            return user_rep, att_scores


        def attr_attention(embed, attr_att_W, len):
            batch = embed.size(0)
            if self.learning_form=='separated':
                user_rep = []
                for attr_w in attr_att_W:
                    rep, att_score = get_attention(attr_w, embed, len)
                    user_rep.append(rep.unsqueeze(2))
                user_rep = torch.cat(user_rep, 2).view(batch,-1)
            else:
                user_rep = get_attention(attr_att_W, embed, len)
            return user_rep


        y = torch.from_numpy(y).float()
        ob = torch.from_numpy(ob).float()
        if torch.cuda.is_available() and not self.no_cuda:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y = y.cuda()
            ob = ob.cuda()
        x_len = torch.sum(x_mask.long(), 1)
        
        # Shared Embedding Layer
        embed = self.item_emb(x)
        
        embeds = []
        for tran_w in self.emb_tran:
            # Embedding Transformation Layer
            attr_embed = F.relu(tran_w(embed))
            embeds.append(attr_embed)
        
        if self.model_type == 'ETNA':
            # Task-Specific Attention Layer
            user_rep, att_scores = item_attention(embeds, False, x_len)
        else:
            # In ETN, user representations are computed by summing item embedding vectors.
            user_rep = torch.stack(embeds)
            x_mask_ = x_mask.unsqueeze(0).unsqueeze(3).expand(user_rep.size())
            user_rep = user_rep*x_mask_.float()
            user_rep = user_rep.sum(2).transpose(1,0).contiguous().view(y.size(0), -1)
            # add a non-linear
            user_rep = torch.sigmoid(user_rep)

        # Prediction Layer
        for i, W in enumerate(self.W_all):
            if i == 0:
                W_user = W(user_rep[:,:self.item_emb_size])
            else:
                W_user = torch.cat((W_user, W(user_rep[:,i*self.item_emb_size:(i+1)*self.item_emb_size])), 1)

        # masking to distinguish between known and unknown attributes
        W_compact = W_user * ob
        y_c = y * ob

        # all attr are observed in new-user prediction
        loss = 0
        for i, t in enumerate(self.tasks):
            weight = None
            lg, ls = compute_loss(W_compact, y, self.cum_len[i], self.cum_len[i+1], self.no_cuda, weight)
            loss += ls
            if i == 0:
                logit = lg
            else:
                logit = np.concatenate((logit, lg), 1)
        return logit, loss


