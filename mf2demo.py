import numpy as np
import json
from functools import reduce
from itertools import chain
import torch
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_fscore_support as f1_score
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
        self.user_emb = np.load('./openrec/user_dim100')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        user_emb = torch.from_numpy(self.user_emb[idx])
        label = torch.from_numpy(np.asarray(self.label[idx]))
        return [user_emb, label]

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

        self.all_posible = Variable(torch.from_numpy(np.asarray(
                              reduce(combinate, all_attr)))).float().cuda()


    def forward(self, batch):
        # batch -- [N, user_emb_dim]
        x = Variable(batch[0]).cuda()
        y = Variable(batch[1]).cuda().float()

        W_user = self.mlp_layer(x)

        denom = 0
        for case in self.all_posible:
            denom += torch.sum(W_user*case, 1).exp()

        obj = torch.sum(W_user*y, 1).exp() / denom
        logit = W_user.data.cpu().numpy()
        loss = -torch.sum(torch.log(obj))
        return logit, loss

class Solver():
    def __init__(self, args):
        self.args = args
        self.model = MF2Demo(args).cuda()
        self.optimizer = None
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
        hm_sum = []
        for idx, batch in enumerate(data_loader):
            if trainable:
                self.optimizer.zero_grad()
            logit, loss = self.model(batch)
            print(loss)
            if trainable:
                loss.backward()
                self.optimizer.step()

            ls = loss.data.cpu().numpy()
            loss_sum.append(ls[0])

            # TODO
            hm_loss, wp, wr, wf1 = get_score(logit, batch[1].numpy(), self.model.dict.attr_len)
            hm_sum.append(hm_loss)
            # eval(pred)
            if idx % 200 == 0:
                print('loss : %.4f | hamming loss %.4f'%\
                        (np.asarray(loss_sum).mean(), np.asarray(hm_sum).mean()))
        print('%s epoch %d summary loss : %.4f | hamming loss %.4f'%(state, epoch,
                np.asarray(loss_sum).mean(), np.asarray(hm_sum).mean()))

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

def run_mfdm_exp(args):
    train_loader = DataLoader(
                    dataset = MFDataset(args, args.data_path+'train.json'),
                    batch_size=args.batch_size)

    valid_loader = DataLoader(
                    dataset = MFDataset(args, args.data_path+'valid.json'),
                    batch_size=args.batch_size,
                    shuffle=False)

    solver = Solver(args)
    for epoch in range(args.max_epoch):
        solver.run_epoch(train_loader, epoch, True)
        solver.run_epoch(valid_loader, epoch, False)
