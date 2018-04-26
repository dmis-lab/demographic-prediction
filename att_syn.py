import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF

import numpy as np
import sys
import time

from utils.debug import Error
from utils.utils import get_mask
#np.set_printoptions(threshold=np.inf)


class ATT_SYN(nn.Module):
    def __init__(self, num_tags, rnn_type, word_emb_size, char_out_size, rnn_size, rnn_layer, \
                rnn_drop, use_crf, use_synopsis, \
                use_em, use_genre, use_metadata, \
                use_word_emb, use_char_emb, \
                span_model, att_mecha):
        super(ATT_SYN, self).__init__()
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.use_crf = use_crf
        self.use_synopsis = use_synopsis
        self.use_em = use_em
        self.use_genre = use_genre
        self.use_metadata = use_metadata
        self.span_model = span_model
        self.att_mecha = att_mecha
        self.mdu_dim = 100

        doc_input_size = syn_input_size = 0
        if use_word_emb:
            doc_input_size += word_emb_size
            syn_input_size += word_emb_size
        if use_char_emb:
            doc_input_size += char_out_size
            syn_input_size += char_out_size
        if use_em:
            doc_input_size += 1
        if use_genre:
            doc_input_size += 23
        if use_metadata:
            doc_input_size += 68
        
        self.contextual_enc_f1 = nn.LSTMCell(
                                    input_size=doc_input_size,
                                    hidden_size=rnn_size)
        self.contextual_enc_b1 = nn.LSTMCell(
                                    input_size=doc_input_size,
                                    hidden_size=rnn_size)
        self.contextual_enc_f2 = nn.LSTMCell(
                                    input_size=rnn_size,
                                    hidden_size=rnn_size)
        self.contextual_enc_b2 = nn.LSTMCell(
                                    input_size=rnn_size,
                                    hidden_size=rnn_size)
        self.contextual_enc_f3 = nn.LSTMCell(
                                    input_size=rnn_size,
                                    hidden_size=rnn_size)
        self.contextual_enc_b3 = nn.LSTMCell(
                                    input_size=rnn_size,
                                    hidden_size=rnn_size)
        if span_model != 'LSTM':
            self.synopsis_enc = getattr(nn, self.rnn_type)(
                                        input_size=syn_input_size,
                                        hidden_size=self.rnn_size,
                                        num_layers=1,
                                        bias=True,
                                        batch_first=True,
                                        dropout=rnn_drop,
                                        bidirectional=True)
            
            self.modeling_enc = getattr(nn, self.rnn_type)(
                                        input_size=4*rnn_size,
                                        hidden_size=self.rnn_size,
                                        num_layers=1,
                                        bias=True,
                                        batch_first=True,
                                        dropout=rnn_drop,
                                        bidirectional=True)
            self.cWh = nn.Linear(2*self.rnn_size, self.mdu_dim, bias=False)
            self.cWu = nn.Linear(2*self.rnn_size, self.mdu_dim, bias=False)
            self.v_hu = nn.Linear(self.mdu_dim, 1, bias=False)
            self.sWh = nn.Linear(2*self.rnn_size, 2*self.rnn_size, bias=False)
            self.sWu = nn.Linear(2*self.rnn_size, 2*self.rnn_size, bias=False)
            self.Ws = nn.Linear(2*self.rnn_size, self.mdu_dim, bias=False)
            
            self.output_li = nn.Linear(6*rnn_size, num_tags)
        else:
            self.output_li = nn.Linear(2*rnn_size, num_tags)
        nn.init.xavier_normal(self.output_li.weight)
        if use_crf:
            self.CRF = CRF(num_tags)
            #nn.init.xavier_normal(self.CRF.start_transitions)
            #nn.init.xavier_normal(self.CRF.end_transitions)
            nn.init.xavier_normal(self.CRF.transitions)
        print("The Attention Summary model has been initialized !")


    def forward(self, input_text, input_syn, label, len_context, len_synopsis, 
                model_analysis, output_txt, alert_score=None):
        B, T, D = input_text.size()
        if self.span_model != 'LSTM':
            _, S, J, _ = input_syn.size()
        else: S = J = 0
        
        # get_mask
        mask = get_mask(B, len_context, T)
        
        # representation layer
        #print('input_text : ', input_text[0])
        #print('lstm ih1 : ', self.contextual_enc_f1.weight_ih)
        #print('lstm hh1 : ', self.contextual_enc_f1.weight_hh)
        #print('lstm bih1 : ', self.contextual_enc_f1.bias_ih)
        #print('lstm bhh1 : ', self.contextual_enc_f1.bias_hh)
        H_, U, m = self.representation_layer(B, T, S, J, D,
                                            input_text,
                                            input_syn,
                                            len_context,
                                            len_synopsis,
                                            mask,
                                            self.span_model)
        #print('H_[0] : ', H_[0])
        #print('lstm ih2 : ', self.contextual_enc_f1.weight_ih)
        if self.span_model != 'LSTM':
            # attention synopsis
            G, output_txt = self.attention_synopsis_layer(B, T, S,
                                                    H_, U, m,
                                                    self.span_model,
                                                    self.att_mecha,
                                                    model_analysis,
                                                    output_txt)
            # output layer
            logit = self.output_layer(B, G, len_context)
        else:
            logit = self.output_layer_lstm(H_)
        
        if model_analysis and not self.use_crf:
            output_txt.write("pred : " + \
                    str(torch.max(logit, 2)[1].data.cpu().numpy().reshape(-1).tolist())+'\n')
        
        # CRF
        if self.use_crf:
            log_likeli = self.CRF(logit.permute(1, 0, 2), \
                                label.permute(1, 0).contiguous(), mask=mask.permute(1, 0))
            viterbi_seq = self.CRF.decode(logit.permute(1, 0, 2), mask=mask.permute(1, 0))
            for seq in viterbi_seq:
                for _ in range(T - len(seq)):
                    seq.append(0)
            if model_analysis:
                output_txt.write("pred : "+str(viterbi_seq.reshape(-1).tolist())+'\n')
            return logit, log_likeli, np.asarray(viterbi_seq), mask
        else: 
            return logit, 0, 0, mask

    def representation_layer(self, B, T, S, J, D, input_text, input_syn,
                            len_context, len_synopsis, mask, span_model):
        # encode the input texts
        h_t_f1, x_state_f1 = self.init_hidden(B, cell=True)
        h_t_b1, x_state_b1 = self.init_hidden(B, cell=True)
        #print('h_t_f1 : ', h_t_f1)
        #print('x_state_f1 : ', x_state_f1)
        #h_t_f2, x_state_f2 = self.init_hidden(B, cell=True)
        #h_t_b2, x_state_b2 = self.init_hidden(B, cell=True)
        #h_t_f3, x_state_f3 = self.init_hidden(B, cell=True)
        #h_t_b3, x_state_b3 = self.init_hidden(B, cell=True)
        
        hiddens, cells = [], []
        x_init = torch.cat((h_t_f1, h_t_b1), 1).unsqueeze(1)
        for time_step in range(input_text.size(1)):
            x_t_f = input_text[ :, time_step, :]
            x_t_b = input_text[ :, input_text.size(1)-time_step-1, :]
            
            h_t_f1, x_state_f1 = self.contextual_enc_f1(x_t_f, (h_t_f1, x_state_f1))
            h_t_b1, x_state_b1 = self.contextual_enc_b1(x_t_b, (h_t_b1, x_state_b1))
            #h_t_f2, x_state_f2 = self.contextual_enc_f2(h_t_f1, (h_t_f2, x_state_f2))
            #h_t_b2, x_state_b2 = self.contextual_enc_b2(h_t_b1, (h_t_b2, x_state_b2))
            #h_t_f3, x_state_f3 = self.contextual_enc_f2(h_t_f2, (h_t_f3, x_state_f3))
            #h_t_b3, x_state_b3 = self.contextual_enc_b2(h_t_b2, (h_t_b3, x_state_b3))
            h_t = torch.cat((h_t_f1, h_t_b1), 1)
            x_state = torch.cat((x_state_f1, x_state_b1), 1)

            hiddens.append(h_t)
            cells.append(x_state)
        H = torch.stack(hiddens, 1)
        m = torch.stack(cells, 1)
        H = H * mask.unsqueeze(2)
        m = m * mask.unsqueeze(2)
        H_ = torch.cat((x_init, H), 1)
        #print(H_[0], H_[1])
        if span_model != 'LSTM':
            # pick paragraphs in the synopsis,  
            # whose length is non-zero
            input_syn = input_syn.view(B*S, J, D)
            len_synopsis = len_synopsis.reshape((-1))
            syn_idx = np.argwhere(len_synopsis > 0).reshape((-1))
            len_syn = np.take(len_synopsis, syn_idx)
            selected_syn = torch.index_select(input_syn, 0, \
                                Variable(torch.from_numpy(syn_idx)).cuda())
            
            # encode the synopses
            s_init = self.init_hidden(selected_syn.size(0), rnn_layer=1)
            s_sort, slen_sort, sidx_unsort = self.sort_sequence(selected_syn, len_syn)
            s_pack = pack_padded_sequence(s_sort, slen_sort, batch_first=True)
            s_rnnout, (s_h, _) = self.synopsis_enc(s_pack, s_init)
            s_h = torch.cat((s_h[s_h.size(0)-2], s_h[s_h.size(0)-1]), 1)
            s_pad, _ = pad_packed_sequence(s_rnnout, batch_first=True)
            syn_rep = self.unsort_sequence(s_h, sidx_unsort)
            
            U = Variable(torch.zeros(B*S, 2*self.rnn_size), requires_grad=True).cuda()
            U[syn_idx.tolist()] = syn_rep
            U = U.view(B, S, -1)
        else:
            U = 0
        return H_, U, m

    def attention_synopsis_layer(self, B, T, S,
                                H_, U, m,
                                span_model, att_mecha, 
                                model_analysis, output_txt):
        H = H_[:, 1:, :]
        D = 2 * self.rnn_size
        
        ## using the synopsis by attention mechanism ##
        
        # reference : 'Knowing When to Look : Adaptive Attention via
        # A Visual Sentinel for Image Captioning'
        
        # calculate the context vector c
        if att_mecha == 'dot':
            pass
        elif att_mecha == 'ff':
            cWh = self.cWh(H)
            cWh_expand = cWh.unsqueeze(2).expand(B, T, S, self.mdu_dim)
            cWu = self.cWu(U).unsqueeze(1).expand(B, T, S, self.mdu_dim)
            z = self.v_hu(F.tanh(cWh_expand + cWu)).squeeze(3)
        elif att_mecha == 'bilinear':
            pass

        alpha = F.softmax(z.view(-1, S), dim=1)  # (B, T, S)
         
        u_expand = U.unsqueeze(1).expand(B, T, S, D)
        
        c = torch.sum(torch.mul(alpha.view(B, T, S, 1).expand(B, T, S, D), \
                    u_expand), 2)
        
        if span_model == 'ATT_SYN':
            # calculate the sentinel s
            sWh = self.sWh(H_[:,:T,:])
            sWu = self.sWu(H_[:,:T,:])
            
            e = F.sigmoid(sWh + sWu)
            s = e * F.tanh(m)
            
            z_hat = self.v_hu(F.tanh(self.Ws(s) + cWh))
            alpha_hat = F.softmax(torch.cat((z, z_hat), 2), 2)
            
            beta = alpha_hat[ : , : ,S].unsqueeze(2)
            
            c_hat = beta * s + (1-beta) * c
            G = torch.cat((H, c_hat), 2)
        elif span_model == 'NoSentinel':
            G = torch.cat((H, c), 2)
        #G = torch.cat((H, U_tilde, torch.mul(H, U_tilde)), 2)
        #G = torch.cat((H, (H - U_tilde).abs()), 2)
        if model_analysis:
            alpha_out = []
            for t in alpha.data.cpu().numpy().round(3).reshape(T,S):
                alpha_out_tmp = []
                for s in t:
                    alpha_out_tmp.append(s)
                alpha_out.append(alpha_out_tmp)
            
            beta_out1 = []
            beta_out2 = []
            for b in beta.data.cpu().numpy().round(3).reshape(-1):
                beta_out1.append(b)
                beta_out2.append(round(1-b,3))

            output_txt.write("alpha : " + str(alpha_out) + '\n')
            output_txt.write("high att : " + \
                    str(torch.max(alpha,1)[1].data.cpu().numpy().reshape(-1).tolist()) + '\n')
            output_txt.write("beta : " + str(beta_out1) + '\n')
            output_txt.write("1-beta : " + str(beta_out2) + '\n')
        return G, output_txt


    def output_layer(self, B, G, len_context):
        m_init = self.init_hidden(B, rnn_layer=1)
        if self.training:
            M, _ = self.modeling_enc(G, m_init)
        else:
            x_sort, xlen_sort, xidx_unsort = self.sort_sequence(G, len_context)
            x_input = pack_padded_sequence(x_sort, xlen_sort, batch_first=True)
            x_rnnout, _ = self.modeling_enc(x_input, m_init)
            x_pad, _ = pad_packed_sequence(x_rnnout, batch_first=True)
            M = self.unsort_sequence(x_pad, xidx_unsort)

        logit = self.output_li(torch.cat((G, M), 2))
        return logit

    def output_layer_lstm(self, H_):
        H = H_[:, 1:, :]
        logit = self.output_li(H)
        return logit

    def sort_sequence(self, data, len_data):
        len_data = torch.from_numpy(len_data)
        _, idx_sort = torch.sort(len_data, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = Variable(idx_sort).cuda()
        idx_unsort = Variable(idx_unsort).cuda()

        sorted_data = data.index_select(0, idx_sort)
        sorted_len = len_data.index_select(0, idx_sort.data.cpu())
        return sorted_data, sorted_len.numpy(), idx_unsort

    def unsort_sequence(self, data, idx_unsort):
        unsorted_data = data.index_select(0, idx_unsort)
        return unsorted_data

    def init_hidden(self, batch_size, rnn_layer=None, cell=None):
        if not rnn_layer : rnn_layer = self.rnn_layer
        if not self.training:
            if not cell:
                if self.rnn_type == 'LSTM':
                    return (Variable(torch.zeros(2*rnn_layer, batch_size, self.rnn_size)).cuda(),
                            Variable(torch.zeros(2*rnn_layer, batch_size, self.rnn_size)).cuda())
                else:
                    return Variable(torch.zeros(2*rnn_layer, batch_size, self.rnn_size)).cuda()
            else:
                if self.rnn_type == 'LSTM':
                    return (Variable(torch.zeros(batch_size, self.rnn_size)).cuda(),
                            Variable(torch.zeros(batch_size, self.rnn_size)).cuda())
                else:
                    return Variable(torch.zeros(batch_size, self.rnn_size)).cuda()
        else:
            if not cell:
                if self.rnn_type == 'LSTM':
                    return (Variable(nn.init.xavier_normal(torch.Tensor(
                            2*rnn_layer, batch_size, self.rnn_size))).cuda(),
                            Variable(nn.init.xavier_normal(torch.Tensor(
                            2*rnn_layer, batch_size, self.rnn_size))).cuda())
                else:
                    return Variable(nn.init.xavier_normal(torch.Tensor(
                            2*rnn_layer, batch_size, self.rnn_size))).cuda()
            else:
                if self.rnn_type == 'LSTM':
                    return (Variable(nn.init.xavier_normal(torch.Tensor(
                            batch_size, self.rnn_size))).cuda(),
                            Variable(nn.init.xavier_normal(torch.Tensor(
                            batch_size, self.rnn_size))).cuda())
                else:
                    return Variable(nn.init.xavier_normal(torch.Tensor(
                            batch_size, self.rnn_size))).cuda()
        

