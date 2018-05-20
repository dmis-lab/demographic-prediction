import numpy as np
import json
from functools import reduce
from itertools import chain
from collections import Counter
from random import randint
from sklearn.metrics import hamming_loss

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

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

class MFDataset(Dataset):
    def __init__(self, args, data_path):
        self.batch_size = args.batch_size
        #self.user_emb --> read_data from openrec
        self.user_emb = self.label = None
        self.read(data_path)

    def read(self, data_path):
        data = json.load(open(data_path))
        self.label = data['label']
        self.observed = data['observed']
        self.user_emb = np.load('./openrec/user_dim100')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        user_emb = torch.from_numpy(self.user_emb[idx])
        label = torch.from_numpy(np.asarray(self.label[idx]))
        observed = torch.from_numpy(np.asarray(self.observed[idx]))
        return [user_emb, label, observed]

class MF2Demo(nn.Module):
    def __init__(self, args):
        super(MF2Demo, self).__init__()
        self.mlp_layer = nn.Sequential(nn.Linear(args.user_emb_dim, 64),
                                    nn.Sigmoid(),
                                    nn.Linear(64, 32),
                                    nn.Sigmoid(),
                                    nn.Linear(32, 18))
        self.dict = Dictionary(
                        args.data_path+'dict.json')
        self.num_negs = args.num_negs
        # generate all the possible structured vectors
        all_attr = []

        for num_class in self.dict.attr_len:
            all_class = [[1 if i==j else 0 for j in range(num_class)] for i in range(num_class)]
            all_attr.append(all_class)

        def combinate(list1, list2):
            out = []
            for l1 in list1:
                for l2 in list2:
                    out.append(l1+l2)
            return out

        self.all_possible = Variable(torch.from_numpy(np.asarray(
                              reduce(combinate, all_attr)))).float().cuda()

    def draw_sample(self, batch_size, label):
        # weight [batch, all_posible]

        # find label index
        labels = label.cpu().data.numpy()
        np_all_possible = self.all_possible.cpu().data.numpy()
        target_idx = []
        for label in labels:
            target_idx.append(np.where((np_all_possible == label).all(axis=1))[0][0])

        # sampling based on uniform weight
        weight = torch.FloatTensor(batch_size, self.all_possible.size(0)).uniform_(0, 1)
        # sample index [batch, num_neg]
        sample_idx = torch.multinomial(weight, self.num_negs).numpy()

        # check if target idx included in sample
        for i, sample in enumerate(sample_idx):
            while target_idx[i] in sample:
                sample[np.where(sample== target_idx[i])] = randint(0,384)

        sample_idx = Variable(torch.from_numpy(sample_idx.astype(int))).cuda()

        neg_samples = []
        for sample in sample_idx:
            neg_samples.append(self.all_possible[sample].unsqueeze(0))
        return torch.cat(neg_samples, 0)

    def forward(self, batch):
        # batch -- [N, user_emb_dim]
        x = Variable(batch[0]).cuda()
        y = Variable(batch[1]).cuda().float()
        ob = Variable(batch[2]).cuda().float()

        neg_samples = self.draw_sample(x.size(0), y)

        W_user = self.mlp_layer(x)
        W_compact = W_user * ob

        c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]
        c_idx = Variable(torch.from_numpy(np.asarray(c_idx))).long().cuda()
        W_compact = torch.index_select(W_compact, 0, c_idx)
        y_c = torch.index_select(y, 0, c_idx)

        neg_logs = []
        for idx, w_c in enumerate(W_compact):
            neg = neg_samples[idx]
            neg_logs.append(F.sigmoid(-(neg*w_c)).log().sum().unsqueeze(0))

        neg_loss = torch.sum(torch.cat(neg_logs), 1)
        pos_loss = torch.sum(torch.log(F.sigmoid(W_compact*y_c)), 1)
        loss = -torch.sum(pos_loss+neg_loss)/W_compact.size(0)

        return W_user.data.cpu().numpy(), loss

class Solver():
    def __init__(self, args):
        self.args = args
        self.model = MF2Demo(args).cuda()
        self.attr_len = self.model.dict.attr_len
        self.optimizer = None
        self.learning_rate = args.learning_rate
        self.select_optimizer()
        self.loss_criterion = nn.NLLLoss()

    def select_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if(self.args.opt == 'Adam'):
            self.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif(self.args.opt == 'RMSprop'):
            self.optimizer =  optim.RMSprop(parameters, lr=self.learning_rate,
                                        weight_decay=self.args.weight_decay,
                                        momentum=self.args.momentum)
        elif(self.args.opt == 'SGD'):
            self.optimizer =  optim.SGD(parameters, lr=self.learning_rate)
        elif(self.args.opt == 'Adagrad'):
            self.optimizer =  optim.Adagrad(parameters, lr=self.args.learning_rate)
        elif(self.args.opt == 'Adadelta'):
            self.optimizer =  optim.Adadelta(parameters, lr=self.args.learning_rate)

    def run_epoch(self, data_loader, epoch, trainable=False):
        if trainable:
            state = 'train'
            self.model.train()
        else:
            state = 'valid'
            self.model.eval()

        # step training or evaluation with given batch size
        loss_sum = []
        self.y_counter = Counter()
        self.y_em_counter = Counter()
        self.hm_acc = self.em = self.num_users = 0
        for idx, batch in enumerate(data_loader):
            if trainable:
                self.optimizer.zero_grad()
            logit, loss = self.model(batch)

            if trainable:
                loss.backward()
                self.optimizer.step()

            ls = loss.data.cpu().numpy()
            loss_sum.append(ls[0])

            # evalutation metric code
            self.accumulate_score(logit, batch[1].numpy(), batch[2].numpy())

            hm_loss, wp, wr, wf1 = self.get_score()

            # eval(pred)
            if idx % 200 == 0:
                print('loss : %.4f | hamming loss %.4f'%\
                        (np.asarray(loss_sum).mean(), hm_loss))
        hm_loss, wp, wr, wf1 = self.get_score()
        print('%s epoch %d summary loss : %.4f | hamming loss %.4f'%(state, epoch,
                np.asarray(loss_sum).mean(), np.asarray(hm_sum).mean()))

    def accumulate_score(self, logit, onehot, observed):
        y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
                                for i, oh in enumerate(onehot)])
        y_pred, y_true = [],[]
        for b_idx, ob in enumerate(observed):
            pred, true = [],[]
            start = 0
            for al in self.attr_len:
                end = start + al
                if not sum(ob[start:end]):
                    pred.append(np.argmax(logit[b_idx][start:end], 0) + start)
                    true.append(sum(y_numbering[b_idx][start:end]))
                start += al
            if pred and true:
                y_pred.append(pred)
                y_true.append(true)

        self.num_users += len(y_true)

        for y in zip(y_pred, y_true):
            self.y_counter[str(y[1])] += 1
            if np.array_equal(y[0], y[1]):
                self.y_em_counter[str(y[1])] += 1
                # count exact matchings for evaluating wP, wR, wF1
                self.em += 1
            # calculate and accumulate hamming loss
            self.hm_acc += hamming_loss(y[1], y[0])

    def get_score(self):
        hm_loss = self.hm_acc / self.num_users
        wP = 0
        for y, cnt in self.y_counter.items():
            wP += self.y_em_counter[y] / cnt
        wP /= len(self.y_counter)

        wR = self.em / self.num_users
        if wP == 0 and wR == 0:
            wP = wR = wF1 = 0
        else:
            wF1 = (2 * wP * wR) / (wP + wR)
        return hm_loss, wP, wR, wF1
'''
def get_score(logit, onehot, attr_len):
    start = 0
    pred = []
    logit = logit.transpose(1,0)
    for al in attr_len:
        end = start + al
        pred.append(np.argmax(logit[start:end], 0) + start)
        start += al

    y_pred = np.asarray(pred).transpose(1,0)
    y_pred = list(chain.from_iterable(
            [yp + (i*sum(attr_len)) for i, yp in enumerate(y_pred)]))
    y_true = list(chain.from_iterable(
            [[j + (i*sum(attr_len)) for j, l in enumerate(oh) if l]  \
                for i, oh in enumerate(onehot)]))

    hm_loss = hamming_loss(y_true, y_pred)
    p, r, f1, _ = f1_score(y_true, y_pred, average='weighted')

    return hm_loss, p, r, f1
'''
def run_mfdm_exp(args):
    train_loader = DataLoader(
                    dataset = MFDataset(args,
                    args.data_path+'train_'+args.task+args.partial_ratio+'.json'),
                    batch_size=args.batch_size)

    valid_loader = DataLoader(
                    dataset = MFDataset(args,
                    args.data_path+'valid_'+args.task+args.partial_ratio+'.json'),
                    batch_size=args.batch_size,
                    shuffle=False)

    solver = Solver(args)
    for epoch in range(args.max_epoch):
        solver.run_epoch(train_loader, epoch, True)
        solver.run_epoch(valid_loader, epoch, False)
