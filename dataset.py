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
    def __init__(self, data_type, data_path, logger):
        self.data_type = data_type
        logger.info("loading " + data_type + " data . . .")
        
        self.history = self.label = None
        self.read(data_path, logger)
        self.dict = json.load(open('./data/preprd/dict.json'))['dict']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return [self.dict.index(h) if h in self.dict else self.dict.index('<UNK>') \
                for h in self.history[index]], \
                self.label[index]

    def read(self, data_path, logger):
        data = json.load(open(data_path))
        
        # check the lengths of data lists
        d_len = set()
        for k in data.keys():
            d_len.add(len(data[k]))
        assert len(d_len) == 1
        logger.info("{} samples are loaded".format(d_len.pop()))
        
        self.history = data['history']
        self.label = data['label']

    def lengths(self):
        return [len(h) for h in self.history]


def batchify(batch):
    history = []
    label = []
    for ex in batch:
        history.append(ex[0])
        label.append(ex[1])

    # padding
    maxlen_history = max([len(h) for h in history])
    x = torch.LongTensor(len(history), maxlen_history).zero_()
    x_mask = torch.ByteTensor(len(history), maxlen_history).zero_()
    for i, h in enumerate(history):
        x[i, :len(h)].copy_(torch.from_numpy(np.asarray(h)))
        x_mask[i, :len(h)].fill_(1)
    y = torch.from_numpy(np.asarray(label))
    return x, x_mask, y

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


