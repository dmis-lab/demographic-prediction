import json
import numpy as np
import random
import os
import sys
import torch
import re
from torch.utils.data import Dataset
from config import get_args

args = get_args()

# set random seeds
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'

    def __init__(self, data_path, task_type):
        if task_type == 'new_user':
            data_path = os.path.join(data_path, task_type, 'dict.json')
        else:
            data_path = os.path.join(data_path, 'partial', 'dict.json')
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
    def __init__(self, logger, data_type, data_path, task_type, model_type):
        self.data_type = data_type
        self.history = self.label = self.observed = None
        
        if task_type == 'new_user':
            data = json.load(open(os.path.join(data_path, task_type, data_type)))
            history = data['history']
            label = data['label']
        else:
            users = json.load(open(os.path.join(data_path, 'partial', 'users.json')))
            history = users['history']
            label = users['attribute']
            observed = json.load(open(os.path.join(data_path, 'partial', task_type)))['observed']

        shuffled_idx = list(range(len(history)))
        self.history = np.asarray(history)[shuffled_idx].tolist()
        self.label = np.asarray(label)[shuffled_idx].tolist()
        self.observed = np.asarray(observed)[shuffled_idx].tolist()
        
        if 'partial' in task_type and any([True if t in self.data_type else False for t in ['valid', 'test']]):
            self.observed = np.invert(np.asarray(self.observed).astype(bool)).astype(int).tolist()

        logger.info("{} {} samples are loaded".format(self.__len__(), self.data_type))


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.history[index], self.label[index], self.observed[index]


def batchify(batch):
    history, label, observed = [],[],[]
    
    for ex in batch:
        history.append(ex[0])
        label.append(ex[1])
        observed.append(ex[2])

    maxlen_history = max([len(h) for h in history])
    x = torch.LongTensor(len(history), maxlen_history).zero_()
    x_mask = torch.ByteTensor(len(history), maxlen_history).zero_()
    for i, h in enumerate(history):
        x[i, :len(h)].copy_(torch.from_numpy(np.asarray(h)))
        x_mask[i, :len(h)].fill_(1)
    y = np.asarray(label)
    ob = np.asarray(observed)

    return x, x_mask, y, ob


