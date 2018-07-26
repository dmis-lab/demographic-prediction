import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import numpy as np
import sys

from .common import combinate, draw_neg_sample

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=99999)

class AvgPooling(nn.Module):
	def __init__(self, logger, len_dict,
				item_emb_size, attr_len, num_negs, 
				partial_training, use_negsample):
		super(AvgPooling, self).__init__()
		self.tasks = [0,1,2,3,4]
		self.logger = logger
		self.attr_len = attr_len
		self.use_negsample = use_negsample
		self.num_negs = num_negs
		self.partial_training = partial_training
		self.optimizer = None

		label_size = sum([al for i, al in enumerate(attr_len)])
		self.item_emb = nn.Embedding(len_dict, item_emb_size, padding_idx=0)

		# choose a learning method
		self.W = nn.Linear(item_emb_size, label_size, bias=False)

		# generate all the possible structured vectors
		all_attr = []
		for num_class in attr_len:
			all_class = [[1 if i==j else 0 for j in range(num_class)] for i in range(num_class)]
			all_attr.append(all_class)

		self.all_possible = np.asarray(reduce(combinate, all_attr))

	def forward(self, process, batch, sample_type, sampling=False):
		x, x_mask, y, ob = batch
		epoch, step = process
		x = x.cuda()
		x_mask = x_mask.cuda()
		x_len = torch.sum(x_mask.long(), 1)
		y = torch.from_numpy(y).cuda().float()
		ob = torch.from_numpy(ob).cuda().float()
		# change all observe for new_user
		if not self.partial_training:
			ob = torch.ones(ob.size()).float().cuda()
		
		# get negative samples
		neg_samples = draw_neg_sample(x.size(0), self.attr_len, y, ob)
		# represent items
		embed = self.item_emb(x)
		# represent users
		user_rep = []
		for i, emb in enumerate(embed):
			user_rep.append(torch.sum(emb, 0)/x_len[i].float())
		user_rep = torch.stack(user_rep, 0)
		# add a non-linear
		#user_rep = F.relu(user_rep)

		# masking to distinguish between known and unknown attributes
		W_user = self.W(user_rep)
		W_compact = W_user * ob
		y_c = y * ob
		if self.use_negsample:
			# using negative sampling for efficient optimization
			neg_logs = []
			for idx, w_c in enumerate(W_compact):
				neg = neg_samples[idx].cuda()
				neg_logs.append(F.sigmoid(-(neg*w_c).sum(0)).log())
			neg_loss = torch.stack(neg_logs).sum(0)
			pos_loss = F.sigmoid((W_compact*y_c).sum(1)).log().sum(0)
			loss = -torch.sum(pos_loss+neg_loss)/W_compact.size(0)
			logit = W_user.data.cpu().numpy()
		else:
			# all attr are observated in new-user prediction

			W_compact = W_user * ob

			c_idx = [i for i, s in enumerate(W_compact.sum(1).data.cpu().numpy()) if s]

			if c_idx:
				c_idx = torch.from_numpy(np.asarray(c_idx)).long().cuda()
				W_compact = torch.index_select(W_compact, 0, c_idx)
				y_c = torch.index_select(y, 0, c_idx)
				all_possible = [[1 if i==j else 0 for j in range(end-start)] \
								for i in range(end-start)]
				all_possible = torch.from_numpy(np.asarray(
									all_possible)).float().cuda()
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
				loss = torch.tensor(0, requires_grad=True).cuda().float()
				batch_size = 1
			logit = W_user.data.cpu().numpy()
			logit = F.softmax(W_user, dim=-1).data.cpu().numpy()
		return logit, loss
