import copy
import json
import numpy as np
import pickle
import sys
import os
from collections import Counter
from utils.debug import Error
import re
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable

# tokens
w_pad = '@w_p@'
w_unk = '#w_unk#'
c_pad = '@c_p@'
c_unk = '#c_unk#'

class Data():
    def __init__(self, args, logger, data_type):
        self.args = args
        self.logger = logger
        self.data_type = data_type
        self.data = {}
        self.logger.info("\n***** " + data_type + " data *****")

        self.read_data()
        self.num_data = len(self.data['context'])
        self.start_idx = 0
        self.end_idx = self.num_data
        
    def doc_lengths(self, data=None, filter=None):
        if not data: data = self.data
        if filter:
            return [(lc, np.amax(ls)) for lc, ls in zip(data['len_context'], data['len_synopsis'])]
        else:
            return [(np.amax(ls), lc) for lc, ls in zip(data['len_context'], data['len_synopsis'])]

    def read_data(self):
        data = json.load(open(os.path.join(
                                self.args.data_path, 
                                self.data_type + '_' + self.args.glove_size + \
                                str(self.args.word_emb_size) + '_' + \
                                str(self.args.num_tags) +'.json')))
        if self.data_type == 'train': 
            dict = json.load(open(os.path.join(
                                self.args.data_path, 
                                'dict_' + self.args.glove_size + \
                                str(self.args.word_emb_size) + '.json')))
            self.unk_dict = dict['unk_dict']
            self.glove_mat = np.asarray(dict['glove_mat'])
            self.char_dict = dict['char_dict']
            self.word_dict = dict['word_dict']
            self.genre_dict = dict['genre_dict']
            self.country_dict = dict['country_dict']
            self.logger.info("   {} of word vocab in the unk word dict".format(len(self.unk_dict)))
            self.logger.info("   {} of word vocab in the glove mat".format(len(self.glove_mat)))
            self.logger.info("   {} of char vocab in the char dict".format(len(self.char_dict)))
            self.logger.info("   {} of char vocab in the genre dict".format(len(self.genre_dict)))
            self.logger.info("   {} of char vocab in the country dict".format(len(self.country_dict)))
        elif self.data_type == 'valid':
            self.char_dict = []
            self.word_dict = []
            self.genre_dict = []
            self.country_dict = []
        elif self.data_type == 'test':
            self.char_dict = []
            self.word_dict = []
            self.genre_dict = []
            self.country_dict = []
        data_keys = ['is_spoiler', 'label', 'context', 'len_context', 'len_synopsis']
                
        # augmenting genre features
        if self.args.use_genre:
            data_keys.append('genre')
        # augmenting metadata features
        if self.args.use_metadata:
            data_keys.append('release')
            data_keys.append('runtime')
            data_keys.append('country')
        # for evaluation in sentence-level
        if self.data_type != 'train':
            data_keys.append('label_s')
        # use character-level word vectors
        if self.args.use_char_emb:
            data_keys.append('context_c')
            data_keys.append('len_word_c')
        # utilizing synopsis
        if self.args.use_synopsis:
            data_keys.append('synopsis')
            if self.args.use_char_emb:
                data_keys.append('synopsis_c')
                data_keys.append('len_syn_word_c')
        elif self.args.save_output:
            data_keys.append('synopsis')
        
        # data filtering
        if self.args.data_filter and self.data_type == 'train':
            filter_idx = self.filter_data(data)
        else:
            filter_idx = list(range(len(data['context'])))
            self.logger.info("Loaded {} (not filtered) examples from {}"
                                .format(len(filter_idx), self.data_type))
        
        self.data = {}
        for key in data_keys:
            self.data[key] = []
        for i in filter_idx:
            for key in data_keys:
                self.data[key].append(data[key][i])
        _, _ = self.count_pos_neg(self.data['is_spoiler'], 'filtered')
 
        # sorted batch sampling
        self.sample_sorted_batch()

    def count_pos_neg(self, is_spoiler, print_str):
        pos_cnt = np.sum(is_spoiler)
        neg_cnt = len(is_spoiler) - pos_cnt
        self.logger.info("{} pos/neg ratio : {} / {} ({:3.2f}%)"
                    .format(print_str, pos_cnt, neg_cnt, 100*(pos_cnt/len(is_spoiler))))
        return pos_cnt, neg_cnt

    def filter_data(self, data):
        num_examples = len(next(iter(data.values())))
        _, _ = self.count_pos_neg(data['is_spoiler'], 'original')
        # data filtering
        filter_idx = []
        filtered = 0
        for i, l in enumerate(self.doc_lengths(data, filter=True)):
            if l[0] > self.args.context_maxlen_filter \
                or l[1] > self.args.syn_par_maxlen_filter \
                or l[0] < self.args.context_minlen_filter:
                filtered += 1
                continue
            else: filter_idx.append(i)
        self.logger.info("Loaded {}/{} (filtered) examples from {}"
                            .format(num_examples-filtered, num_examples, self.data_type))
        return filter_idx


    def sample_sorted_batch(self):
        self.logger.info("randomly sample batches clustered by length")
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.doc_lengths()],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)])
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        
        if self.data_type == 'train':
            eval_batch_size = self.args.batch_size
        else:  # validation or test
            if self.args.save_output:
                eval_batch_size = 1
            else:
                if self.args.use_synopsis:
                    eval_batch_size = 16
                else:
                    eval_batch_size = 24
        
        batches = [indices[i:i + eval_batch_size]
                    for i in range(0, len(indices), eval_batch_size)]

        #if self.args.data_shuffle:
        #    np.random.shuffle(batches)
        self.batch_idx = batches

    def shuffle(self, data_in=None):
        if data_in: data = data_in
        else: data = self.data
        sort_idx = np.arange(len(data['len_context']))
        np.random.shuffle(sort_idx)
        data_out = {}
        for key in data.keys():
            data_out[key] = []
        for i in sort_idx:
            for key in data.keys():
                data_out[key].append(data[key][i])
        if data_in: return data_out
        else: self.data = data_out

    def pad_batch(self):
        def np_pad(li, maxl, l, padding=0):
            li = np.lib.pad(li, (0, maxl - l), 'constant', constant_values=padding)
            return li

        # pad contexts
        for p_idx, para in enumerate(self.batch['context']):
            if len(para) < self.b_maxlen_context:
                self.batch['context'][p_idx] = np_pad(para, self.b_maxlen_context, len(para))
                
                self.batch['label'][p_idx] = np_pad(self.batch['label'][p_idx],
                                                    self.b_maxlen_context, len(para))
                if self.args.use_char_emb:
                    self.batch['context_c'][p_idx] = np_pad(
                                                    self.batch['context_c'][p_idx],
                                                    self.b_maxlen_context, len(para))
                    self.batch['len_word_c'][p_idx] = np_pad(
                                                    self.batch['len_word_c'][p_idx],
                                                    self.b_maxlen_context, len(para))
            elif len(para) > self.b_maxlen_context:
                para = para[:self.b_maxlen_context]
                self.batch['label'][p_idx] = self.batch['label'][:self.b_maxlen_context]
                if self.args.use_char_emb:
                    self.batch['context_c'][p_idx] = self.batch['context_c'][:self.b_maxlen_context]
                    self.batch['len_word_c'][p_idx] = self.batch['len_word_c'][:self.b_maxlen_word_c]
        # pad synopses
        if self.args.use_synopsis or self.args.save_output:
            for d_idx, doc in enumerate(self.batch['synopsis']):
                if len(doc) < self.b_len_par:
                    for _ in range(self.b_len_par - len(doc)):
                        empty = np.zeros(self.b_maxlen_synopsis)
                        doc.append(empty)
                        self.batch['len_synopsis'][d_idx].append(0)
                    #self.batch['synopsis'][d_idx] = np_pad(doc, self.b_len_par, len(doc))
                    #self.batch['len_synopsis'][d_idx] = np_pad(
                    #                                    self.batch['len_synopsis'][d_idx],
                    #                                    self.b_len_par, len(doc))
                for p_idx, para in enumerate(doc):
                    #print(p_idx, para)
                    #if not para:
                    #    print(para)
                    #    para = [0]
                    #    print(para, len(para))
                    #    sys.exit()
                    if len(para) < self.b_maxlen_synopsis:
                        self.batch['synopsis'][d_idx][p_idx] = np_pad(para, 
                                                            self.b_maxlen_synopsis, len(para))
                        if self.args.use_char_emb:
                            self.batch['synopsis_c'][d_idx][p_idx] = np_pad(
                                                            self.batch['synopsis_c'][d_idx][p_idx],
                                                            self.b_maxlen_synopsis, len(para))
                            self.batch['len_syn_word_c'][d_idx][p_idx] = np_pad(
                                                            self.batch['len_syn_word_c'][d_idx][p_idx],
                                                            self.b_maxlen_synopsis, len(para))
        # pad context_c
        if self.args.use_char_emb:
            for p_idx, para in enumerate(self.batch['context_c']):
                for w_idx, word in enumerate(para):
                    if word == w_pad:
                        word_tmp = []
                        for _ in range(self.b_maxlen_word_c):
                            word_tmp.append(c_pad)
                        self.batch['context_c'][p_idx][w_idx] = copy.deepcopy(word_tmp)
                    else:
                        if len(word) < self.b_maxlen_word_c:
                            for _ in range(self.b_maxlen_word_c - len(word)):
                                word.append(c_pad)
                        elif len(word) > self.b_maxlen_word_c:
                            word = word[:self.b_maxlen_word_c]
            if self.args.use_synopsis:
                for p_idx, para in enumerate(self.batch['synopsis_c']):
                    for w_idx, word in enumerate(para):
                        if word == w_pad:
                            word_tmp = []
                            for _ in range(self.b_maxlen_smr_word_c):
                                word_tmp.append(c_pad)
                            self.batch['synopsis_c'][p_idx][w_idx] = copy.deepcopy(word_tmp)
                        else:
                            if len(word) < self.b_maxlen_smr_word_c:
                                for _ in range(self.b_maxlen_smr_word_c - len(word)):
                                    word.append(c_pad)
                            elif len(word) > self.b_maxlen_smr_word_c:
                                word = word[:self.b_maxlen_smr_word_c]

    def char2idx(self):
        if self.args.use_char_emb:
            for p_idx, para in enumerate(self.batch['context_c']):
                for w_idx, word in enumerate(para):
                    for c_idx, char in enumerate(word):
                        if char in self.char_dict: 
                            self.batch['context_c'][p_idx][w_idx][c_idx] = self.char_dict.index(char)
                        else: 
                            self.batch['context_c'][p_idx][w_idx][c_idx] = self.char_dict.index(c_unk)
        if self.args.use_char_emb:
            for p_idx, para in enumerate(self.batch['context_char']):
                for c_idx, char in enumerate(para):
                    if char in self.char_dict:
                        self.batch['context_char'][p_idx][c_idx] = self.char_dict.index(char)
                    else:
                        self.batch['context_char'][p_idx][c_idx] = self.char_dict.index(c_unk)
        if self.args.use_synopsis:
            if self.args.use_char_emb:
                for p_idx, para in enumerate(self.batch['synopsis_c']):
                    for w_idx, word in enumerate(para):
                        for c_idx, char in enumerate(word):
                            if char in self.char_dict: 
                                self.batch['synopsis_c'][p_idx][w_idx][c_idx] = self.char_dict.index(char)
                            else: 
                                self.batch['synopsis_c'][p_idx][w_idx][c_idx] = self.char_dict.index(c_unk)

    def encode_one_hot(self, data, dictionary):
        one_hot = np.zeros((len(data), len(dictionary)))
        for d in data:
            if d in dictionary:
                one_hot[dictionary.index(d)] = 1
        return one_hot

    def get_batch(self, i):
        self.batch = {}
        for key in self.data.keys():
            self.batch[key] = []
            self.batch[key] = copy.deepcopy(
                    np.asarray(self.data[key])[self.batch_idx[i]].tolist())
        
        # calculate the maximum length of . . . in a mini-batch
        self.b_maxlen_context = np.amax(self.batch['len_context'])
        if self.args.use_synopsis or self.args.save_output:
            self.b_maxlen_synopsis = self.b_len_par = 0
            for ls in self.batch['len_synopsis']:
                lenpar_tmp = len(ls)
                maxtmp = np.amax(ls)
                if self.b_maxlen_synopsis < maxtmp:
                    self.b_maxlen_synopsis = maxtmp
                if self.b_len_par < lenpar_tmp:
                    self.b_len_par = lenpar_tmp
        if self.args.use_char_emb:
            self.b_maxlen_word_c = 0
            for s in self.batch['len_word_c']:
                max_tmp = np.amax(s)
                self.b_maxlen_word_c = max(max_tmp, self.b_maxlen_word_c)
            self.b_maxlen_word_c = max(5, self.b_maxlen_word_c)
            if self.args.use_synopsis:
                self.b_maxlen_smr_word_c = 0
                for s in self.batch['len_smr_word_c']:
                    max_tmp = np.amax(s)
                    self.b_maxlen_smr_word_c = max(max_tmp, self.b_maxlen_smr_word_c)
        
        # pad sequences with the max len
        self.pad_batch()
        
        # mapping words or characters into indices of each dictionary
        self.char2idx()

        # one hot encoding
        if self.args.use_genre:
            self.batch['genre'] = self.encode_one_hot(self.batch['genre'], self.genre_dict)
        if self.args.use_metadata:
            self.batch['country'] = self.encode_one_hot(self.batch['country'], self.country_dict)

        # hard attention : exact match
        if self.args.use_em:
            self.batch['exact'] = []
            for d_idx, d in enumerate(self.batch['context']):
                tmp = []
                for w in d:
                    if w in self.batch['synopsis'][d_idx]:
                        tmp.append(1)
                    else: tmp.append(0)
                self.batch['exact'].append(tmp)

        # convert python lists into 'Variable's of pytorch
        for key in self.batch.keys():
            # to numpy
            self.batch[key] = np.asarray(self.batch[key])
            # to Variable
            if not 'len' in key and not 'label_s' in key:
                self.batch[key] = Variable(torch.from_numpy(self.batch[key])).cuda()
        return self.batch


