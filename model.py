import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
import sys
import time

class DemoPredictor(nn.Module):
    def __init__(self, args, logger):
        super(DemoPredictor, self).__init__()
        self.unk_emb = nn.Embedding(unk_word_size, 
                                    self.item_emb_size,
                                    padding_idx=0)
        self.unk_emb.weight.requires_grad = False
        self.item_emb = nn.Embedding(len(glove_mat), 
                                    self.item_emb_size)
        self.init_item_emb_weight(glove_mat)
        
        # model selection

    def init_item_emb_weight(self, glove_mat):
        glove_mat = torch.from_numpy(glove_mat).cuda()
        self.glove_emb.weight.data = glove_mat
        self.glove_emb.weight.requires_grad = True

    def forward(self, batch):
        return 0

    def emb_char(self, context_c, batch_size, T, C):
        c = F.dropout(self.char_emb(
            context_c.view(-1, C)), self.char_drop)\
            .view(batch_size, T, C, self.char_emb_size)

        c = c.permute(0, 3, 1, 2)
        
        conv_c = F.relu(self.conv(c))
        conv_c = torch.squeeze(F.max_pool2d(conv_c, (1, conv_c.size(-1))), 3).permute(0, 2, 1)
        return conv_c
    
    def emb_word(self, context):
        L = np.prod(context.size()).astype(float)
        B = context.size(0)
        T = context.size(1)
        D = self.word_emb_size
        
        # select word indice which should be inserted into glove embedding layer
        glove_idx_mask = torch.ge(context.view(-1), self.unk_word_size)
        glove_idx = torch.masked_select(
                                Variable(torch.arange(L), requires_grad=False).cuda(),
                                glove_idx_mask)
        glove_context = torch.masked_select(context.view(-1), glove_idx_mask)
        glove_context -= self.unk_word_size
        
        # for unknown words
        unk_idx_mask = torch.lt(context.view(-1), self.unk_word_size)
        unk_idx = torch.masked_select(
                                Variable(torch.arange(L), requires_grad=False).cuda(),
                                unk_idx_mask)
        unk_context = torch.masked_select(context.view(-1), unk_idx_mask)
        
        # gathering two types of word representations
        word_emb = Variable(torch.zeros(int(L), D)).cuda()
        if glove_context.size():
            emb_glove = self.glove_emb(glove_context.long()).float()
            word_emb[glove_idx.data.cpu().numpy().astype(int).tolist()] = emb_glove
        if unk_context.size():
            emb_unk = self.unk_word_emb(unk_context.long()).float()
            word_emb[unk_idx.data.cpu().numpy().astype(int).tolist()] = emb_unk
        word_emb = word_emb.view(B, T, int(L//(B*T)), D).squeeze(2)
        return word_emb


