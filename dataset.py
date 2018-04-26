import copy
import json
import numpy as np
import os
import re
import sys
import time
import torch
from torch.autograd import Variable

# tokens
h_pad = '@h_p@'
h_unk = '#h_unk#'

class Data():
    def __init__(self, args, logger, data_type):
        self.args = args
        self.logger = logger
        self.data = {}
        self.logger.info("loading " + data_type + " data . . .")

        self.read_data(data_type)

    def read_data(self, data_type):
        data = json.load(open(os.path.join(
                            self.args.data_path, data_type + '.json')))
        # check the lengths of data lists
        d_len = set()
        for k in data.keys():
            d_len.add(len(data[k]))
        assert len(d_len) == 1
        self.logger.info("{} samples are loaded".format(d_len.pop()))
        sys.exit()

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


