from collections import Counter
import copy
import json
from random import randint
import numpy as np
import random
import re
import sys
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'

    def __init__(self, data_path):
        load_file = json.load(open(data_path))
        self.dict = load_file['dict']
        self.attr_len = load_file['attr_len']

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict)

    def add(self, item):
        if item not in self.dict:
            self.dict.append(item)

class DemoAttrDataset(Dataset):
    def __init__(self, logger, data_type, data_path, aug_data_path=None):
        self.data_type = data_type

        self.history = self.label = self.observed = None
        self.read(logger, data_path, aug_data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.history[index], self.label[index], self.observed[index]

    def read(self, logger, data_path, aug_data_path=None):
        data = json.load(open(data_path))
        if aug_data_path is not None:
            aug_data = json.load(open(aug_data_path))
            data['history'] += aug_data['history']
            data['label'] += aug_data['label']
            data['observed'] += aug_data['observed']

        history, label, observed = [],[],[]
        #for i, ob in enumerate(data['observed']):
        #    if sum(ob):
        #        history.append(data['history'][i])
        #        label.append(data['label'][i])
        #        observed.append(data['observed'][i])
        history = data['history']
        label = data['label']
        observed = data['observed']
        ##

        shuffled_idx = list(range(len(history)))
        random.shuffle(shuffled_idx)
        self.history_all = np.asarray(history)[shuffled_idx].tolist()
        # label ex : [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        self.label_all = np.asarray(label)[shuffled_idx].tolist()
        # observed ex : [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.observed_all = np.asarray(observed)[shuffled_idx].tolist()
        if 'val' in self.data_type :
            self.observed_all = np.invert(np.asarray(self.observed_all).astype(bool)).astype(int).tolist()

        self.history = self.history_all
        self.label = self.label_all
        self.observed = self.observed_all

        logger.info("{} {} samples are loaded".format(self.__len__(), self.data_type))

    def shuffle_data(self):
        shuffled_idx = list(range(len(self.history_all)))
        random.shuffle(shuffled_idx)
        self.history_all = np.asarray(self.history_all)[shuffled_idx].tolist()
        self.label_all = np.asarray(self.label_all)[shuffled_idx].tolist()
        self.observed_all = np.asarray(self.observed_all)[shuffled_idx].tolist()

    def sample_data(self, sample_attr):
        self.shuffle_data()

        # y_counter : true full label / sampled_counter : sampled y
        # kn_counter : known label
        y_counter = Counter()
        sampled_counter = Counter()
        kn_counter = Counter()
        unk_counter = Counter()

        #for l in self.label:
        #    self.y_counter[str(l)] += 1
        #    self.sampled_counter[str(l)] = 0
        # y_numbering ex : [ 0  1  2  0  0  5  0  0  0  0  0 11  0  0 14  0  0  0]
        y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
                                for i, oh in enumerate(self.label_all)])

        y_true = []
        y_known = []
        y_unknown = []
        attr_idx_dict = dict()
        for b_idx, ob in enumerate(self.observed_all):
            true = []
            known = []
            unknown = []
            start = 0
            for a_idx, al in enumerate([2,2,4,4,6]):
                end = start + al
                # only use unknown (to be prediction)
                #if not sum(ob[start:end]):
                  # t = sum(y_numbering[b_idx][start:end])
                   #true.append(t)
                t = sum(y_numbering[b_idx][start:end])
                true.append(t)

                if not sum(ob[start:end]):
                    unk = sum(y_numbering[b_idx][start:end])
                    unknown.append(unk)
                else:
                    kn = sum(y_numbering[b_idx][start:end])
                    if kn in attr_idx_dict.keys():
                        attr_idx_dict[kn].append(b_idx)
                    else:
                        attr_idx_dict[kn] = [b_idx]
                    known.append(kn)
                start += al

            # true - all true label / known - observed label
            y_true.append(true)
            y_known.append(known)
            y_unknown.append(unknown)

        l_idx_dict = dict()
        total_l = set([str(l) for l in y_true])
        for l in total_l:
            l_idx_dict[l] = []

        for l_idx, l in enumerate(y_true):
            y_counter[str(l)] += 1
            sampled_counter[str(l)] = 0
            l_idx_dict[str(l)].append(l_idx)

        for l in y_known:
            kn_counter[str(l)] += 1
        for l in y_unknown:
            unk_counter[str(l)] += 1

        #torch.save(y_counter, 'y_counter_10')
        #torch.save(kn_counter, 'kn_counter_10')
        #torch.save(unk_counter, 'unk_counter_10')
        #sys.exit()
        def attr_sample_size(attr_idx_dict, att):
            start = 0
            for idx, n_cls in enumerate([2,2,4,4,6]):
                if idx<att:
                    start += n_cls
                else:
                    end = start+n_cls
                    t_cls = [c for c in range(start,end)]
                    t_num = [len(attr_idx_dict[c]) for c in range(start, end)]
                    #sample_size = int(np.median(np.asarray(t_num)))
                    sample_size = 57693//len(t_cls)
                    return t_cls, sample_size
        '''
        for i in range(18):
            print(i, len(attr_idx_dict[i]))
        for i in range(5):
            t_cls, sample_size = attr_sample_size(attr_idx_dict, i)
            print(t_cls, sample_size)
        sys.exit()
        '''
        t_cls, sample_size = attr_sample_size(attr_idx_dict, sample_attr)

        history = []
        label = []
        observed = []
        #sample_size = 5000
        for c in t_cls:
            # sample from the pool with replacement
            idx_pool = attr_idx_dict[c]

            for _ in range(sample_size - sampled_counter[c]):
                l_idx = idx_pool[randint(0, len(idx_pool)-1)]
                history.append(self.history_all[l_idx])
                label.append(self.label_all[l_idx])
                observed.append(self.observed_all[l_idx])
                sampled_counter[c] += 1

        '''
        if sample_attr=='full':
            counter_ = y_counter
        else:
            counter_ = kn_counter


        for l_idx, l in enumerate(y_true):
            # over sampling
            if counter_[str(l)] < 200 and sampled_counter[str(l)] < 200:
                # sample from the pool with replacement
                idx_pool = l_idx_dict[str(l)]
                for _ in range(200 - sampled_counter[str(l)]):
                    l_idx = idx_pool[randint(0, len(idx_pool)-1)]
                    history.append(self.history_all[l_idx])
                    label.append(self.label_all[l_idx])
                    observed.append(self.observed_all[l_idx])
                    sampled_counter[str(l)] += 1
            # under sampling
            if (counter_[str(l)] > 200 and sampled_counter[str(l)] < 200):
                #or (counter_[str(l)] <= 100 and counter_[str(l)] >= 20):
                history.append(self.history_all[l_idx])
                label.append(self.label_all[l_idx])
                observed.append(self.observed_all[l_idx])
                sampled_counter[str(l)] += 1

            if self.y_counter[str(l)] < 200 and self.sampled_counter[str(l)] < 200:
                for _ in range(200 - self.sampled_counter[str(l)]):
                    history.append(self.history_all[l_idx])
                    label.append(self.label_all[l_idx])
                    observed.append(self.observed_all[l_idx])
                    self.sampled_counter[str(l)] += 1
            if (self.y_counter[str(l)] > 500 and self.sampled_counter[str(l)] < 500)\
                or (self.y_counter[str(l)] <= 500 and self.y_counter[str(l)] >= 200):
                history.append(self.history_all[l_idx])
                label.append(self.label_all[l_idx])
                observed.append(self.observed_all[l_idx])
                self.sampled_counter[str(l)] += 1
            '''
        self.history = history
        self.label = label
        self.observed = observed

    def under_sample(self):
        r = int(len(self.history)*0.85)
        if self.data_type == 'train':
            self.history = self.history[:r]
            self.label = self.label[:r]
            self.observed = self.observed[:r]
        else:
            self.history = self.history[r:]
            self.label = self.label[r:]
            self.observed = self.observed[r:]

        female = male = 0
        for l_idx, l in enumerate(self.label):
            if l[0]: female += 1
            else: male += 1
        print(female, male)

        balancing = False
        if balancing:
            history1 = []
            label1 = []
            observed1 = []
            f = m = 0
            for l_idx, l in enumerate(self.label):
                if f >= male and l[0]: continue
                else:
                    label1.append(l)
                    history1.append(self.history[l_idx])
                    observed1.append(self.observed[l_idx])
                    if l[0]: f += 1
                    else: m += 1
            self.history = history1
            self.label = label1
            self.observed = observed1

        shuffled_idx = list(range(len(self.history)))
        random.shuffle(shuffled_idx)
        self.history = np.asarray(self.history)[shuffled_idx].tolist()
        self.label = np.asarray(self.label)[shuffled_idx].tolist()
        self.observed = np.asarray(self.observed)[shuffled_idx].tolist()

    def lengths(self):
        return [len(h) for h in self.history]


def batchify(batch):
    history, label, observed = [],[],[]
    for ex in batch:
        history.append(ex[0])
        label.append(ex[1])
        observed.append(ex[2])

    # padding
    maxlen_history = max([len(h) for h in history])
    x = torch.LongTensor(len(history), maxlen_history).zero_()
    x_mask = torch.ByteTensor(len(history), maxlen_history).zero_()
    for i, h in enumerate(history):
        x[i, :len(h)].copy_(torch.from_numpy(np.asarray(h)))
        x_mask[i, :len(h)].fill_(1)
    y = np.asarray(label)
    ob = np.asarray(observed)
    return x, x_mask, y, ob

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
                [(-l, np.random.random()) for l in self.lengths],
                dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
