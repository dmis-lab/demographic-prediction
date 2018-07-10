# some operations common in different models

import random
import torch

def combinate(list1, list2):
	out = []
	for l1 in list1:
		for l2 in list2:
			out.append(l1+l2)
	return out

def draw_neg_sample(batch_size, label, observed):
	# weight [batch, all_posible]
	# find label index
	val_label = label*observed
	neg_samples = []
	for val_l in val_label:
		neg_idx = []
		val_y = val_l.nonzero()

		for attr_y in val_y:
			start = end = 0
			for n in [2,2,4,4,6]:
				end = start + n
				if start <= attr_y < end:
					candidate = [i for i in range(start,end) if i!=attr_y]
					neg = random.sample(candidate, 1)
					neg_idx.append(neg)
				start += n
		neg_sample = torch.zeros(18)
		for idx in neg_idx:
			neg_sample[idx] = 1
		neg_samples.append(neg_sample)
	return neg_samples

