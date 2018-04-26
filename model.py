import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
import sys
import time
from att_syn import ATT_SYN

from utils.debug import Error
#np.set_printoptions(threshold=np.inf)

class Model(nn.Module):
    def __init__(self, args, logger):
        super(Model, self).__init__()

        self.unk_word_emb = nn.Embedding(unk_word_size, 
                                        self.word_emb_size,
                                        padding_idx=0)
        self.unk_word_emb.weight.requires_grad = False
        self.glove_emb = nn.Embedding(len(glove_mat), self.word_emb_size)
        self.init_glove_emb_weight(glove_mat)
        if self.use_char_emb:
            self.char_emb = nn.Embedding(char_vocab_size, 
                                        self.char_emb_size,
                                        padding_idx=0)
            self.conv = nn.Conv2d(self.char_emb_size, args.char_out_size, (1, 5))
        
        # model selection
        self.span_detector = ATT_SYN(args.num_tags,
                                    args.rnn_type,
                                    args.word_emb_size,
                                    args.char_out_size,
                                    args.rnn_size,
                                    args.rnn_layer,
                                    args.rnn_drop,
                                    args.use_crf,
                                    args.use_synopsis,
                                    args.use_em,
                                    args.use_genre,
                                    args.use_metadata,
                                    args.use_word_emb,
                                    args.use_char_emb,
                                    args.span_model,
                                    args.att_mecha)


    def init_glove_emb_weight(self, glove_mat):
        glove_mat = torch.from_numpy(glove_mat).cuda()
        self.glove_emb.weight.data = glove_mat
        self.glove_emb.weight.requires_grad = True

    def forward(self, batch, model_analysis=False, output_txt=None):
        B = batch['context'].size(0)
        T = batch['context'].size(1)
        if self.use_char_emb:
            C = batch['context_c'].size(2)
        if self.use_synopsis:
            J = batch['synopsis'].size(2)
            if self.use_char_emb:
                F = batch['synopsis_c'].size(3)

        # representing input text
        if self.use_word_emb and self.use_char_emb:
            w_emb = self.emb_word(batch['context'])
            cl_emb = self.emb_char(batch['context_c'], B, T, C)
            input_text = torch.cat((w_emb, cl_emb), 2)
        elif self.use_word_emb:
            w_emb = self.emb_word(batch['context'])
            input_text = w_emb
        elif self.use_char_emb:
            cl_emb = self.emb_char(batch['context_c'], B, T, C)
            input_text = cl_emb


        if self.use_em:
            batch['exact'] = batch['exact'].unsqueeze(2).float()
            input_text = torch.cat((input_text, batch['exact']), 2)
        if self.use_metadata:
            batch['release'] = batch['release'].view(-1, 1, 1).expand(B, T, 1).float()
            batch['runtime'] = batch['runtime'].view(-1, 1, 1).expand(B, T, 1).float()
            batch['country'] = batch['country'].view(-1, 1, batch['country'].size(1))\
                                        .expand(B, T, batch['country'].size(1)).float()
            input_text = torch.cat((input_text, \
                                    batch['release'], \
                                    batch['runtime'], \
                                    batch['country']), 2)
        if self.use_genre:
            batch['genre'] = batch['genre'].view(-1, 1, batch['genre'].size(1))\
                                        .expand(B, T, batch['genre'].size(1)).float()
            input_text = torch.cat((input_text, batch['genre']), 2)


        # representing synopses
        if self.use_synopsis:
            len_synopsis = batch['len_synopsis']
            if self.use_word_emb and self.use_char_emb:
                w_emb = self.emb_word(batch['synopsis'])
                c_emb = self.emb_char(batch['synopsis_c'], B, J, F)
                input_syn = torch.cat((w_emb, c_emb), 2)
            elif self.use_word_emb:
                w_emb = self.emb_word(batch['synopsis'])
                input_syn = w_emb
            elif self.use_char_emb:
                c_emb = self.emb_char(batch['synopsis_c'], B, J, F)
                input_syn = s_emb
        else: 
            input_syn = 0
            len_synopsis = 0
        span_logit, log_likeli, viterbi_seq, mask = self.span_detector(
                                                input_text,
                                                input_syn,
                                                batch['label'],
                                                batch['len_context'],
                                                len_synopsis,
                                                model_analysis,
                                                output_txt)
        return span_logit, log_likeli, viterbi_seq, mask

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


