from collections import Counter
import copy
import json
from random import randint
import numpy as np
import pickle
import random
import re
import sys
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
#from imblearn.over_sampling import SMOTE

# set random seeds
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

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
	def __init__(self, logger, task_type, data_type, data_path, aug_data_path=None):
		self.data_type = data_type
		self.history = self.label = self.observed = None
		self.read(logger, task_type, data_path, aug_data_path)

	def __len__(self):
		return len(self.label)

	def __getitem__(self, index):
		return self.history[index], self.label[index], self.observed[index], self.svd_rep[index]

	def read(self, logger, task_type, data_path, aug_data_path=None):
		data = json.load(open(data_path))
		if aug_data_path is not None:
			aug_data = json.load(open(aug_data_path))
			data['history'] += aug_data['history']
			data['label'] += aug_data['label']
			data['observed'] += aug_data['observed']

		history, label, observed = [],[],[]
		history = data['history']
		label = data['label']
		
		if 'observed' in data.keys():
			observed = data['observed']
		else:
			observed = np.zeros((len(history), len(label[0])))
		shuffled_idx = list(range(len(history)))
		self.history_all = np.asarray(history)[shuffled_idx].tolist()
		# label ex : [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
		self.label_all = np.asarray(label)[shuffled_idx].tolist()
		# observed ex : [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.observed_all = np.asarray(observed)[shuffled_idx].tolist()
		#if task_type == 'partial' and any([True if t in self.data_type else False for t in ['val', 'test']]):
		#	self.observed_all = np.invert(np.asarray(self.observed_all).astype(bool)).astype(int).tolist()

		with open('./save/rep/svd_matrix', 'rb') as f:
			self.svd_rep = pickle.load(f)
		self.svd_rep = self.svd_rep[shuffled_idx]

		self.reset()
		logger.info("{} {} samples are loaded".format(self.__len__(), self.data_type))

	def shuffle_data(self):
		history_all = copy.deepcopy(self.history_all)
		label_all = copy.deepcopy(self.label_all)
		observed_all = copy.deepcopy(self.observed_all)

		shuffled_idx = list(range(len(history_all)))

		random.shuffle(shuffled_idx)

		idx_zero = shuffled_idx.index(0)
		self.history_all = np.asarray(history_all)[shuffled_idx].tolist()
		self.label_all = np.asarray(label_all)[shuffled_idx].tolist()
		self.observed_all = np.asarray(observed_all)[shuffled_idx].tolist()

		for i in range(len(shuffled_idx)):
			j = shuffled_idx.index(i)
			if history_all[i] != self.history_all[j]:
				print(history_all[i])
				print(self.history_all[j])
			if label_all[i] != self.label_all[j]:
				print(label_all[i])
				print(self.label_all[j])
			if observed_all[i] != self.observed_all[j]:
				print(observed_all[i])
				print(self.observed_all[j])

	def reset(self):
		self.history = copy.deepcopy(self.history_all)
		self.label = copy.deepcopy(self.label_all)
		self.observed = copy.deepcopy(self.observed_all)

	def sample_subset(self, num_data):
		history_all = copy.deepcopy(self.history_all)
		label_all = copy.deepcopy(self.label_all)
		observed_all = copy.deepcopy(self.observed_all)
		self.history, self.label, self.observed = [],[],[]
		for i in range(num_data):
			self.history.append(history_all[i])
			self.label.append(label_all[i])
			self.observed.append(observed_all[i])

	def sample_data_cls(self):
		#self.shuffle_data()

		# y_counter : true full label / sampled_counter : sampled y
		# kn_counter : known label
		y_counter = Counter()
		sampled_counter = Counter()
		kn_counter = Counter()
		unk_counter = Counter()

		self.cls_idx_dict = dict()

		# y_numbering ex : [ 0  1  2  0  0  5  0  0  0  0  0 11  0  0 14  0  0  0]
		y_numbering = copy.deepcopy(
						np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
						for i, oh in enumerate(self.label_all)]))

		y_true = []
		for t_idx, t in enumerate(y_numbering):
			true = []
			start = 0
			for a_idx, al in enumerate([2,2,4,4,6]):
				end = start + al
				if sum(self.observed_all[t_idx][start:end]):
					t = sum(y_numbering[t_idx][start:end])
					true.append(t)
				start += al
			# add data index to class index pool
			if str(true) in self.cls_idx_dict.keys():
				self.cls_idx_dict[str(true)].append(t_idx)
			else:
				self.cls_idx_dict[str(true)] = [t_idx]
			# true - all true label

			y_true.append(true)

	def pick_batch_data(self, num_batches, batch_size):
		history, label, observed = [],[],[]
		history_all = copy.deepcopy(self.history_all)
		label_all = copy.deepcopy(self.label_all)
		observed_all = copy.deepcopy(self.observed_all)

		#sample from the pool. every class included in each batch
		for _ in range(num_batches):
			num_cls = np.zeros(18).astype(int)
			for _ in range(batch_size):
				while True:
					c = random.choice(list(self.cls_idx_dict.keys()))
					if c != '[]': break

				cls = c.replace('[','').replace(']','').split(', ')
				for cl in cls:
					num_cls[int(cl)] += 1
				# sample from the pool with replacement
				s_idx = random.choice(self.cls_idx_dict[c])

				history.append(history_all[s_idx])
				label.append(label_all[s_idx])
				observed.append(observed_all[s_idx])

			# calculate the max num of classes for each att
			start = end = 0
			maxnum_cls = []
			for n in [2,2,4,4,6]:
				end = start + n
				mn = max(num_cls[start:end])
				for _ in range(n):
					maxnum_cls.append(mn)
				start = end

			# make the num of classes same to avoid skewed prediction
			maxnum_cls = np.asarray(maxnum_cls)

			while True:
				under = num_cls < maxnum_cls
				candi = [i for i, u in enumerate(under) if u]

				while True:
					c = random.choice(list(self.cls_idx_dict.keys()))
					if c != '[]': break

				cls = np.asarray(c.replace('[','').replace(']','').split(', ')).astype(int)
				intersect = np.intersect1d(cls, candi)
				difference = np.setdiff1d(cls, candi)
				s_idx = random.choice(self.cls_idx_dict[c])


				# threshold
				if not intersect.shape[0]: continue

				for cl in intersect:
					num_cls[int(cl)] += 1

				history.append(history_all[s_idx])
				label.append(label_all[s_idx])
				new_ob = copy.deepcopy(np.asarray(observed_all[s_idx]))

				start = end = 0
				for n in [2,2,4,4,6]:
					end = start + n
					for d in difference:
						if d in list(range(start, end)):
							new_ob[start:end] = 0
					start = end
				observed.append(new_ob.tolist())
				if sum(np.equal(num_cls, maxnum_cls).astype(int)) == 18: break

		self.history = history
		self.label = label
		self.observed = observed

	def sample_data_attr(self, sample_attr):
		self.shuffle_data()

		# y_counter : true full label / sampled_counter : sampled y
		# kn_counter : known label
		y_counter = Counter()
		sampled_counter = Counter()
		kn_counter = Counter()
		unk_counter = Counter()

		#for l in self.label:
		#    self.y_counter[str(l)] += 1
		#    self.sampled_counter[str(l)] = 0
		# y_numbering ex : [ 0  1  2  0  0  5  0  0  0  0  0 11  0  0 14  0  0  0]
		y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
								for i, oh in enumerate(self.label_all)])

		y_true = []
		y_known = []
		y_unknown = []
		attr_idx_dict = dict()
		for b_idx, ob in enumerate(self.observed_all):
			true = []
			known = []
			unknown = []
			start = 0
			for a_idx, al in enumerate([2,2,4,4,6]):
				end = start + al
				# only use unknown (to be prediction)
				#if not sum(ob[start:end]):
				  # t = sum(y_numbering[b_idx][start:end])
				   #true.append(t)
				t = sum(y_numbering[b_idx][start:end])
				true.append(t)

				if not sum(ob[start:end]):
					unk = sum(y_numbering[b_idx][start:end])
					unknown.append(unk)
				else:
					kn = sum(y_numbering[b_idx][start:end])
					if kn in attr_idx_dict.keys():
						attr_idx_dict[kn].append(b_idx)
					else:
						attr_idx_dict[kn] = [b_idx]
					known.append(kn)
				start += al

			# true - all true label / known - observed label
			y_true.append(true)
			y_known.append(known)
			y_unknown.append(unknown)

		l_idx_dict = dict()
		total_l = set([str(l) for l in y_true])
		for l in total_l:
			l_idx_dict[l] = []

		for l_idx, l in enumerate(y_true):
			y_counter[str(l)] += 1
			sampled_counter[str(l)] = 0
			l_idx_dict[str(l)].append(l_idx)

		for l in y_known:
			kn_counter[str(l)] += 1
		for l in y_unknown:
			unk_counter[str(l)] += 1

		#torch.save(y_counter, 'y_counter_10')
		#torch.save(kn_counter, 'kn_counter_10')
		#torch.save(unk_counter, 'unk_counter_10')
		#sys.exit()
		def attr_sample_size(attr_idx_dict, att):
			start = 0
			for idx, n_cls in enumerate([2,2,4,4,6]):
				if idx<att:
					start += n_cls
				else:
					end = start+n_cls
					t_cls = [c for c in range(start,end)]
					t_num = [len(attr_idx_dict[c]) for c in range(start, end)]
					#sample_size = int(np.median(np.asarray(t_num)))
					sample_size = 57693//len(t_cls)
					return t_cls, sample_size
		'''
		for i in range(18):
			print(i, len(attr_idx_dict[i]))
		for i in range(5):
			t_cls, sample_size = attr_sample_size(attr_idx_dict, i)
			print(t_cls, sample_size)
		sys.exit()
		'''
		t_cls, sample_size = attr_sample_size(attr_idx_dict, sample_attr)

		history = []
		label = []
		observed = []
		#sample_size = 5000
		for c in t_cls:
			# sample from the pool with replacement
			idx_pool = attr_idx_dict[c]
			# same size sampling
			for _ in range(sample_size - sampled_counter[c]):
				l_idx = idx_pool[randint(0, len(idx_pool)-1)]
				history.append(self.history_all[l_idx])
				label.append(self.label_all[l_idx])
				observed.append(self.observed_all[l_idx])
				sampled_counter[c] += 1

		self.history = history
		self.label = label
		self.observed = observed

	def under_sample(self):
		r = int(len(self.history)*0.85)
		if self.data_type == 'train':
			self.history = self.history[:r]
			self.label = self.label[:r]
			self.observed = self.observed[:r]
		else:
			self.history = self.history[r:]
			self.label = self.label[r:]
			self.observed = self.observed[r:]

		female = male = 0
		for l_idx, l in enumerate(self.label):
			if l[0]: female += 1
			else: male += 1
		print(female, male)

		balancing = False
		if balancing:
			history1 = []
			label1 = []
			observed1 = []
			f = m = 0
			for l_idx, l in enumerate(self.label):
				if f >= male and l[0]: continue
				else:
					label1.append(l)
					history1.append(self.history[l_idx])
					observed1.append(self.observed[l_idx])
					if l[0]: f += 1
					else: m += 1
			self.history = history1
			self.label = label1
			self.observed = observed1

		shuffled_idx = list(range(len(self.history)))
		random.shuffle(shuffled_idx)
		self.history = np.asarray(self.history)[shuffled_idx].tolist()
		self.label = np.asarray(self.label)[shuffled_idx].tolist()
		self.observed = np.asarray(self.observed)[shuffled_idx].tolist()

	def smote(self):
		maxlen_history = 0
		minlen_history = 999
		for h in self.history_all:
			maxlen_history = max(maxlen_history, max(h))
			minlen_history = min(minlen_history, min(h))

		sparse_input = []
		for h in self.history_all:
			onehot = np.zeros(maxlen_history)
			for i in h:
				onehot[i-1] += 1
			sparse_input.append(onehot)

		onehot2label = []
		for l in self.label_all:
			if not l in onehot2label:
				onehot2label.append(l)

		x = np.asarray(sparse_input)
		y = []
		cnt = 0
		for l in self.label_all:
			cnt += 1
			y.append(onehot2label.index(l))
		print('y :', Counter(y))

		sm = SMOTE(random_state=1, n_jobs=1)
		sm.fit(x, y)
		x_res, y_res = sm.sample(x, y)
		print('y_res :', Counter(y_res))
		sys.exit()

	def lengths(self):
		return [len(h) for h in self.history]

def batchify(batch):
	history, label, observed = [],[],[]
	svd_rep = []
	for ex in batch:
		history.append(ex[0])
		label.append(ex[1])
		observed.append(ex[2])
		svd_rep.append(ex[3])

	# padding
	maxlen_history = max([len(h) for h in history])
	maxuniq_history = max(len(set(h)) for h in history)
	x = torch.LongTensor(len(history), maxlen_history).zero_()
	x_uniq = torch.LongTensor(len(history), maxuniq_history).zero_()
	x_mask = torch.ByteTensor(len(history), maxlen_history).zero_()
	x_uniq_mask = torch.ByteTensor(len(history), maxuniq_history).zero_()
	for i, h in enumerate(history):
		x[i, :len(h)].copy_(torch.from_numpy(np.asarray(h)))
		x_uniq[i, :len(set(h))].copy_(torch.from_numpy(np.asarray(list(set(h)))))
		x_mask[i, :len(h)].fill_(1)
		x_uniq_mask[i, :len(set(h))].fill_(1)
	y = np.asarray(label)
	ob = np.asarray(observed)
	svd_rep = np.asarray(svd_rep)
	return x, x_mask, x_uniq, x_uniq_mask, y, ob, svd_rep

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
