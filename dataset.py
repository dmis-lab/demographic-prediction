import copy
import json
import numpy as np
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
        for i, ob in enumerate(data['observed']):
            if sum(ob):
                history.append(data['history'][i])
                label.append(data['label'][i])
                observed.append(data['observed'][i])
        self.history = history
        self.label = label
        self.observed = observed
        #self.count_class()
        logger.info("{} {} samples are loaded".format(self.__len__(), self.data_type))

    def count_class(self):
        print(self.label[0])
        print(self.observed[0])
        female = male = 0
        history1 = []
        label1 = []
        observed1 = []
        for l_idx, l in enumerate(self.label):
            if female >= 15905 and l[0]: continue
            else: 
                label1.append(l)
                history1.append(self.history[l_idx])
                observed1.append(self.observed[l_idx])
                if l[0]: female += 1
                else: male += 1

        print(female, male, len(label1))

        married = single = 0
        label2 = []
        for l_idx, l in enumerate(label1):
            if married >= 12398 and l[2]: continue
            else: 
                label2.append(l)
                if l[2]: married += 1
                else: single += 1
        
        print(married, single, len(label2))
        
        y = ad = ma = old = 0
        label3 = []
        for l_idx, l in enumerate(label2):
            label3.append(l)
            if l[4]: y += 1
            elif l[5]: ad += 1
            elif l[6]: ma += 1
            else: old += 1
        print(y, ad, ma, old, len(label3))
        
        ul = low = med = high = 0
        label4 = []
        for l_idx, l in enumerate(label3):
            if l[8]: ul += 1
            elif l[9]: low += 1
            elif l[10]: med += 1
            else: high += 1

        print(ul, low, med, high, len(label4))
        sys.exit()
        self.history = history1
        self.label = label1
        self.observed = observed1
        print(len(self.history), len(self.label), len(self.observed))
        
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
    y = torch.from_numpy(np.asarray(label))
    ob = torch.from_numpy(np.asarray(observed))
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
