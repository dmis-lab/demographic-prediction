# some operations common in different models

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def draw_neg_sample(batch_size, attr_len, label, observed):
    # weight [batch, all_posible]
    # find label index
    val_label = label*observed
    neg_samples = []
    for val_l in val_label:
        neg_idx = []
        val_y = val_l.nonzero()
        for attr_y in val_y:
            start = end = 0
            for n in attr_len:
                end = start + n
                if start <= attr_y < end:
                    candidate = [i for i in range(start,end) if i!=attr_y]
                    neg = random.sample(candidate, 1)
                    neg_idx.append(neg)
                start += n
        neg_sample = torch.zeros(label.size(1))
        for idx in neg_idx:
            neg_sample[idx] = 1
        neg_samples.append(neg_sample)
    return torch.stack(neg_samples)

def compute_loss(W_user, label, ob, start, end, no_cuda, weight=None):
    W_c = W_user * ob
    W_c = W_c.transpose(1,0)[start:end].transpose(1,0)
    y = label.transpose(1,0)[start:end].transpose(1,0)

    prob = F.softmax(W_c, dim=1).cpu().detach().numpy()

    c_idx = [i for i, s in enumerate(W_c.sum(1).cpu().detach().numpy()) if s]

    if c_idx:
        c_idx = (torch.from_numpy(np.asarray(c_idx))).long()
        if torch.cuda.is_available() and not no_cuda:
            c_idx = c_idx.cuda()
        W_c = torch.index_select(W_c, 0, c_idx)
        y_c = torch.index_select(y, 0, c_idx)
        
        all_possible = [[1 if i==j else 0 for j in range(end-start)] \
                                        for i in range(end-start)]
        all_possible = (torch.from_numpy(np.asarray(
                        all_possible))).float()
        if torch.cuda.is_available() and not no_cuda:
            all_possible = all_possible.cuda()

        denom = 0
        for case in all_possible:
            denom += torch.sum(W_c*case, 1).exp()
        obj = torch.sum(W_c*y_c, 1).exp() / denom

        if weight is not None:
            weighted = torch.sum(y_c * weight, 1)
            loss = -torch.sum(obj.log()*weighted)
        else:
            loss = -torch.sum(obj.log())
        batch_size = y_c.size(0)
    else:
        loss = torch.tensor(0, requires_grad=True).float()
        batch_size = 1

    return prob, loss / batch_size



