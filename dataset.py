import copy
import json
import numpy as np
import os
import re
import sys
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

# tokens
h_pad = '@h_p@'
h_unk = '#h_unk#'

class DemoAttrDataset(Dataset):
    def __init__(self, args, logger, data_type):
        self.args = args
        self.logger = logger
        self.data_type = data_type
        self.logger.info("loading " + data_type + " data . . .")
        
        self.history = None
        self.label = None
        self.read_data(data_type)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.history[index], self.label[index]

    def read_data(self, data_type):
        data = json.load(open(os.path.join(
                            self.args.data_path, data_type + '.json')))
        
        # check the lengths of data lists
        d_len = set()
        for k in data.keys():
            d_len.add(len(data[k]))
        assert len(d_len) == 1
        self.logger.info("{} samples are loaded".format(d_len.pop()))
        
        self.history = data['history']
        self.label = data['label']

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
