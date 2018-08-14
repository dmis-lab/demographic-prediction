# some operations common in different models

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

def combinate(list1, list2):
	out = []
	for l1 in list1:
		for l2 in list2:
			out.append(l1+l2)
	return out

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

def compute_loss(WU, full_label, start, end, weight=None):
	W_user = WU.transpose(1,0)[start:end].transpose(1,0)
	y = full_label.transpose(1,0)[start:end].transpose(1,0)

	c_idx = [i for i, s in enumerate(W_user.sum(1).data.cpu().numpy()) if s]

	if c_idx:
		c_idx = (torch.from_numpy(np.asarray(c_idx))).long().cuda()
		W_user = torch.index_select(W_user, 0, c_idx)
		y_c = torch.index_select(y, 0, c_idx)
		all_possible = [[1 if i==j else 0 for j in range(end-start)] \
						for i in range(end-start)]
		all_possible = (torch.from_numpy(np.asarray(
							all_possible))).float().cuda()
		denom = 0
		for case in all_possible:
			denom += torch.sum(W_user*case, 1).exp()
		obj = torch.sum(W_user*y_c, 1).exp() / denom

		if weight is not None:
			weighted = torch.sum(y_c * weight, 1)
			loss = -torch.sum(obj.log()*weighted)
		else:
			loss = -torch.sum(obj.log())
		batch_size = y_c.size(0)
	else:
		loss = torch.tensor(0, requires_grad=True).float().cuda()
		batch_size = 1
	logit = W_user.data.cpu().numpy()
	logit = F.softmax(W_user, dim=1).data.cpu().numpy()
	return logit, loss / batch_size

def compute_cross_entropy(WU, full_label, start, end, loss_criterion):
	W_user = WU.transpose(1,0)[start:end].transpose(1,0)
	y = full_label.transpose(1,0)[start:end].transpose(1,0).cpu().numpy()

	c_idx = [i for i, ob in enumerate(y.sum(1)) if ob]
	y_c = y[c_idx]
	W_c = W_user[c_idx]

	y_c = torch.from_numpy(np.argmax(y_c, 1)).cuda().long()
	loss = loss_criterion(W_c, y_c)

	logit = W_user.data.cpu().numpy()
	logit = F.softmax(W_user, dim=1).data.cpu().numpy()

	return logit, loss

################################## deprecated code
'''
if self.use_negsample:
	# get negative samples
	neg_samples = draw_neg_sample(x.size(0), self.attr_len, y, ob)
	# using negative sampling for efficient optimization
	neg_logs = []
	for idx, w_c in enumerate(W_compact):
		neg = neg_samples[idx].cuda()
		neg_logs.append(F.sigmoid(-(neg*w_c).sum(0)).log())
	neg_loss = torch.stack(neg_logs).sum(0)
	pos_loss = F.sigmoid((W_compact*y_c).sum(1)).log().sum(0)
	loss = -torch.sum(pos_loss+neg_loss)/W_compact.size(0)
	logit = W_user.data.cpu().numpy()


    c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]

    if c_idx:
        c_idx = Variable(torch.from_numpy(np.asarray(c_idx))).long().cuda()
        W_compact = torch.index_select(W_compact, 0, c_idx)
        y_c = torch.index_select(y, 0, c_idx)
        all_possible = [[1 if i==j else 0 for j in range(end-start)] \
                        for i in range(end-start)]
        all_possible = Variable(torch.from_numpy(np.asarray(
                            all_possible))).float().cuda()
        denom = 0
        for case in all_possible:
            denom += torch.sum(W_compact*case, 1).exp()
        obj = torch.sum(W_compact*y_c, 1).exp() / denom

        if weight is not None:
            weighted = torch.sum(y_c * weight, 1)
            loss = -torch.sum(obj.log()*weighted)
        else:
            loss = -torch.sum(obj.log())
        batch_size = y_c.size(0)
    else:
        loss = torch.tensor(0, requires_grad=True).float().cuda()
        batch_size = 1
    logit = W_user.data.cpu().numpy()
    logit = F.softmax(W_user, dim=1).data.cpu().numpy()
return logit, loss

def compute_loss(WU, full_label, observed, start, end, weight=None):
    W_user = WU.transpose(1,0)[start:end].transpose(1,0)
    y = full_label.transpose(1,0)[start:end].transpose(1,0)
    ob = observed.transpose(1,0)[start:end].transpose(1,0)
    # change all observe for new_user
    if not self.partial_training:
        ob = Variable(torch.ones(ob.size())).float().cuda()

    W_compact = W_user * ob

    c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]

    if c_idx:
        c_idx = Variable(torch.from_numpy(np.asarray(c_idx))).long().cuda()
        W_compact = torch.index_select(W_compact, 0, c_idx)
        y_c = torch.index_select(y, 0, c_idx)
        all_possible = [[1 if i==j else 0 for j in range(end-start)] \
                        for i in range(end-start)]
        all_possible = Variable(torch.from_numpy(np.asarray(
                            all_possible))).float().cuda()
        denom = 0
        for case in all_possible:
            denom += torch.sum(W_compact*case, 1).exp()
        obj = torch.sum(W_compact*y_c, 1).exp() / denom

        if weight is not None:
            weighted = torch.sum(y_c * weight, 1)
            loss = -torch.sum(obj.log()*weighted)
        else:
            loss = -torch.sum(obj.log())
        batch_size = y_c.size(0)
    else:
        loss = torch.tensor(0, requires_grad=True).float().cuda()
        batch_size = 1
    logit = W_user.data.cpu().numpy()
    logit = F.softmax(W_user, dim=1).data.cpu().numpy()
    return logit, loss / batch_size


loss = torch.tensor(0, requires_grad=True).float().cuda()
if self.sampling:
    # when training we will only use loss attribute used in sampling
    for i, t in enumerate(self.tasks):
        weight = None
        lg, ls = compute_loss(W_user, y, ob, self.cum_len[i], self.cum_len[i+1], weight)
        if t==sample_type:
            loss = ls
        if i == 0:
            logit = lg
        else:
            logit = np.concatenate((logit, lg), 1)

else:
    for i, t in enumerate(self.tasks):
        #if t == 0:
        #    weight = Variable(torch.from_numpy(np.asarray([1, 2]))).float().cuda()
        #else: weight = None
        weight = None
        lg, ls = compute_loss(W_user, y, ob, self.cum_len[i], self.cum_len[i+1], weight)
        loss += ls
        if i == 0:
            logit = lg
        else:
            logit = np.concatenate((logit, lg), 1)
return logit, loss

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
			sample[np.where(sample== target_idx[i])] = randint(0, self.all_possible.size(0)-1)
	sample_idx = Variable(torch.from_numpy(sample_idx.astype(int))).cuda()
	neg_samples = []
	for sample in sample_idx:
		neg_samples.append(self.all_possible[sample].unsqueeze(0))

	return torch.cat(neg_samples, 0)


'''
