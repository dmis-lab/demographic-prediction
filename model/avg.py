import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import numpy as np
import sys

from .common import combinate, draw_neg_sample, compute_loss, compute_cross_entropy

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=99999)

class AvgPooling(nn.Module):
	def __init__(self, logger, len_dict, share_emb, uniq_input,
				item_emb_size, attr_len, learning_form, loss_type,
				partial_training, use_negsample, tasks=[0,1,2]):
		super(AvgPooling, self).__init__()
		self.tasks = tasks
		self.cum_len = np.concatenate(([0], np.cumsum(np.asarray(attr_len)[tasks])))
		self.logger = logger
		self.share_emb = share_emb
		self.attr_len = attr_len
		self.learning_form = learning_form
		self.use_negsample = use_negsample
		self.partial_training = partial_training
		self.optimizer = None
		self.loss_type = loss_type

		if loss_type == 'classification':
			weight = torch.load('./data/preprd/ocb/class_loss_weight')
			#self.loss_criterion = nn.ModuleList([nn.CrossEntropyLoss(weight[i]) for i in range(len(attr_len))])
			self.loss_criterion = nn.ModuleList([nn.CrossEntropyLoss() for i in range(len(attr_len))])

		user_size = item_emb_size
		label_size = sum([al for i, al in enumerate(attr_len)])

		if share_emb:
			self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)
		else:
			self.item_emb = nn.ModuleList([nn.Embedding(len_dict, item_emb_size, padding_idx=0)
										for _ in range(len(attr_len))])

		if learning_form == 'seperated':
			self.W_all = nn.ModuleList()
			for i, al in enumerate(attr_len):
			    if i in tasks:
			        self.W_all.append(nn.Linear(user_size, attr_len[i], bias=True))
		else:
			if self.share_emb:
				self.W = nn.Linear(item_emb_size, label_size, bias=True)
			else:
				#self.W = nn.Linear(item_emb_size*len(attr_len), label_size, bias=True)
				self.W = nn.Linear(item_emb_size, label_size, bias=True)

		# choose a learning method

		# generate all the possible structured vectors
		all_attr = []
		for num_class in attr_len:
			all_class = [[1 if i==j else 0 for j in range(num_class)] for i in range(num_class)]
			all_attr.append(all_class)

		self.all_possible = np.asarray(reduce(combinate, all_attr))

	def forward(self, process, batch, rep=None, sampling=False):
		x, x_mask, x_uniq, x_uniq_mask, y, ob = batch
		epoch, step = process
		x = x.cuda()
		x_mask = x_mask.cuda()
		x_uniq = x_uniq.cuda()
		x_uniq_mask = x_uniq_mask.cuda()
		x_len = torch.sum(x_mask.long(), 1)
		y = torch.from_numpy(y).cuda().float()
		ob = torch.from_numpy(ob).cuda().float()

		# change all observe for new_user
		if not self.partial_training:
			ob = torch.ones(ob.size()).float().cuda()

		# represent items
		if self.share_emb:
			embed = self.item_emb(x)

			# represent users
			user_rep = []
			for i, emb in enumerate(embed):
				user_rep.append(torch.sum(emb, 0)/x_len[i].float())
			user_rep = torch.stack(user_rep, 0)

			# using svd
			#user_rep.data = torch.tensor(rep).cuda().float()
		else:
			user_reps = []
			for i in range(len(self.item_emb)):
				embed = self.item_emb[i](x)
				user_rep = []
				for i, emb in enumerate(embed):
					user_rep.append(torch.sum(emb, 0)/x_len[i].float())
				user_rep = torch.stack(user_rep, 0)
				user_reps.append(user_rep)

		# add a non-linear
		#user_rep = F.relu(user_rep)

		# masking to distinguish between known and unknown attributes
		if self.learning_form == 'structured':
			if not self.share_emb:
				user_rep = torch.cat(user_reps,1)
			W_user = self.W(user_rep)
		else:
			if self.share_emb:
				for i, W in enumerate(self.W_all):
				    if i == 0:
				        W_user = W(user_rep)
				    else:
				        W_user = torch.cat((W_user, W(user_rep)), 1)
			else:
				for i, W in enumerate(self.W_all):
				    if i == 0:
				        W_user = W(user_reps[i])
				    else:
				        W_user = torch.cat((W_user, W(user_reps[i])), 1)

		W_compact = W_user * ob
		y_c = y * ob

		if self.loss_type == 'classification':
			loss = 0
			for i, t in enumerate(self.tasks):
				lg, ls = compute_cross_entropy(W_user, y_c, self.cum_len[i], self.cum_len[i+1], self.loss_criterion[i])
				loss += ls

				if i == 0:
					logit = lg
				else:
					logit = np.concatenate((logit, lg), 1)

		elif self.use_negsample:
			# using negative sampling for efficient optimization
			neg_samples = draw_neg_sample(x.size(0), self.attr_len, y, ob)
			neg_logs = []
			for idx, w_c in enumerate(W_compact):
				neg = neg_samples[idx].cuda()
				neg_logs.append(F.sigmoid(-(neg*w_c).sum(0)).log())
			neg_loss = torch.stack(neg_logs).sum(0)
			pos_loss = F.sigmoid((W_compact*y_c).sum(1)).log().sum(0)
			loss = (-torch.sum(pos_loss+neg_loss)/W_compact.size(0))
			logit = W_user.data.cpu().numpy()

		else:
			# all attr are observed in new-user prediction
			loss = 0
			for i, t in enumerate(self.tasks):
				#if t == 0:
				#    weight = Variable(torch.from_numpy(np.asarray([1, 2]))).float().cuda()
				#else: weight = None
				weight = None

				lg, ls = compute_loss(W_compact, y, self.cum_len[i], self.cum_len[i+1], weight)
				loss += ls
				if i == 0:
					logit = lg
				else:
					logit = np.concatenate((logit, lg), 1)
		torch.save(self.item_emb.weight, './save/emb/avg')
		return logit, loss
