
import argparse
import copy
from collections import Counter
from functools import reduce
from itertools import chain
import logging
import numpy as np
from operator import mul
import os
from sklearn.metrics import hamming_loss
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from dataset import Dictionary
from model.avg import AvgPooling
from model.tan import TANDemoPredictor


class Experiment:
	def __init__(self, args, logger):

		self.vis =  SummaryWriter(log_dir='save/vis')

		self.args = args
		self.tasks = args.tasks
		self.logger = logger
		#Dict = Dictionary(
		#		args.data_path+'dict_'+args.task_type+args.partial_ratio+'.json')
		Dict = Dictionary(
				args.data_path+args.dataset+'/dict.json')
		self.dict = Dict.dict
		self.attr_len = Dict.attr_len
		self.all_the_poss = reduce(mul, Dict.attr_len, 1)
		self.logger.info("Experiment initializing . . . ")

		# build models
		self.model = []
		def build_models(tasks_list):
			if args.model_type!='TAN':
				for tasks in tasks_list:
					self.model.append(AvgPooling(logger, self.dict.__len__(),
								args.item_emb_size, Dict.attr_len, args.num_negs,
								args.partial_training, args.use_negsample, tasks=tasks).cuda())
			else:
				for tasks in tasks_list:
					self.model.append(TANDemoPredictor(logger, self.dict.__len__(), args.item_emb_size,
									args.attention_layer, Dict.attr_len, args.learning_form,
									args.use_negsample, args.partial_training, args.uniq_input, tasks = tasks).cuda())

		build_models([self.tasks])
		for model in self.model:
			self.select_optimizer(model)
			self.logger.info(model)
		self.step_count = 0


	def select_optimizer(self, model):
		parameters = filter(lambda p: p.requires_grad, model.parameters())
		if(self.args.opt == 'Adam'):
			model.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate,
										weight_decay=self.args.weight_decay
										)
		elif(self.args.opt == 'RMSprop'):
			model.optimizer =  optim.RMSprop(parameters, lr=self.args.learning_rate,
										#weight_decay=self.args.weight_decay,
										momentum=self.args.momentum)
		elif(self.args.opt == 'SGD'):
			model.optimizer =  optim.SGD(parameters, lr=self.args.learning_rate,
										#weight_decay=self.args.weight_decay,
										momentum=self.args.momentum)
		elif(self.args.opt == 'Adagrad'):
			model.optimizer =  optim.Adagrad(parameters, lr=self.args.learning_rate)
		elif(self.args.opt == 'Adadelta'):
			model.optimizer =  optim.Adadelta(parameters, lr=self.args.learning_rate)

	def adjust_lr(self):
		for model in self.model:
			for param_group in model.optimizer.param_groups:
				param_group['lr'] *= self.args.lr_decay

	def run_epoch(self, epoch, data_loader, sample_type, sampling=False, trainable=False):
		num_samples = data_loader.dataset.__len__()
		num_steps = (num_samples // self.args.batch_size) + 1
		self.num_steps = num_steps
		self.logger.info("== {} mode : {} steps for {} samples == \n"
			.format(data_loader.dataset.data_type, num_steps, num_samples))

		self.ytc_counter = []
		self.ypc_counter = []
		for i in range(sum([al for al in self.attr_len])):
			self.ypc_counter.append(0)
			self.ytc_counter.append(0)

		self.y_em_counter = Counter()
		self.yp_counter = Counter()
		self.yt_counter = Counter()
		self.hm_acc = self.num_users = 0
		self.attr_em = np.zeros(len(self.attr_len)).astype(int)
		self.attr_cnt = [0 if i in self.tasks else 1 for i in range(len(self.attr_len))]
		loss_sum = 0
		for i, batch in enumerate(data_loader):
			t0 = time.clock()
			self.step = i+1
			self.step_count += 1
			f_logit = None
			for t_idx, _ in enumerate([self.tasks]):
				model = self.model[t_idx]
				if self.args.model_type == 'POP': break

				# change the mode
				if trainable:
					model.train()
					model.optimizer.zero_grad()
				else:
					model.eval()

				start = 0
				delete_idx = []
				for a_idx, al in enumerate(self.attr_len):
					end = start + al
					if not a_idx in model.tasks:
						delete_idx.extend(list(range(start, end)))
					start += al
				onehot = np.delete(batch[4], delete_idx, 1)
				observed = np.delete(batch[5], delete_idx, 1)
				logit, loss = model((epoch, i+1),
									(batch[0], batch[1], batch[2], batch[3],
									onehot, observed),
									trainable)
				#if self.step_count % self.args.vis_per_step == 0 and not trainable:
				#	self.summary(loss, self.step_count, False)

				if trainable:
					# weight decay for parameters selected by us
					#wd_lambda = torch.tensor(self.args.weight_decay).cuda()
					#wd_l2 = torch.tensor(0.).cuda()
					#for name, param in model.named_parameters():
					#	if name != 'item_emb.weight':
					#		wd_l2 += torch.norm(param)
					#loss += wd_lambda * wd_l2

					loss.backward()
					nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)

					# diminishing the impact of W matrix on model prediction
					#for name, param in model.named_parameters():
					#	if name != 'item_emb.weight':
					#		param.grad *= 1/20

					model.optimizer.step()

					# diminishing the impact of W matrix on model prediction
					#for name, param in model.named_parameters():
					#	if name != 'item_emb.weight':
					#		param.data *= 1/2
				ls = loss.data.cpu().numpy()
				loss_sum += ls


				if f_logit is None:
					f_logit = logit
				else:
					f_logit = np.concatenate((f_logit, logit), 1)

			self.accumulate_score(f_logit, batch[4], batch[5], self.tasks, trainable, sample_type)

			if (i+1) % self.args.print_per_step == 0:
				hm, macP, macR, macF1, wP, wR, wF1 = self.get_score()
				t1 = time.clock()
				self.logger.info("< Step {} > Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
							.format(self.step, loss_sum/self.step, t1-t0, hm))
				self.logger.info("macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
							.format(macP, macR, macF1))
				self.logger.info("weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f}"
							.format(wP, wR, wF1))
				#self.logger.info("Accuracy - gender:{:3.1f}, marital:{:3.1f}, age:{:3.1f}, income:{:3.1f}, edu:{:3.1f} \n"
				self.logger.info("Accuracy - gender:{:3.1f}, age:{:3.1f}, marital:{:3.1f} \n"
									.format(100*self.attr_em[0]/self.attr_cnt[0],
											100*self.attr_em[1]/self.attr_cnt[1],
											100*self.attr_em[2]/self.attr_cnt[2]))
											#100*self.attr_em[3]/self.attr_cnt[3],
											#100*self.attr_em[4]/self.attr_cnt[4]))
		print('pred :', self.ypc_counter)
		print('true :', self.ytc_counter)
		self.logger.info("Accuracy - gender:{:3.1f}, age:{:3.1f}, marital:{:3.1f} \n"
							.format(100*self.attr_em[0]/self.attr_cnt[0],
									100*self.attr_em[1]/self.attr_cnt[1],
									100*self.attr_em[2]/self.attr_cnt[2]))
		#for name, param in model.named_parameters():
		#	print(name, torch.norm(param))
		hm, macP, macR, macF1, wP, wR, wF1 = self.get_score()
		#if not trainable:
		#	self.summary(loss, self.step_count, True)
		return loss_sum / num_steps, hm, macP, macR, macF1, wP, wR, wF1

	def accumulate_score(self, logit, onehot, observed, tasks, trainable, sample_type):
		if not self.args.partial_eval: observed = np.ones_like(observed)

		y_pred, y_true = [],[]
		y_pred_f, y_true_f = [],[]
		y_numbering = np.asarray([[j if l else 0 for j, l in enumerate(oh)] \
								for i, oh in enumerate(onehot)])

		if self.args.model_type == 'POP':
			popular = [[0, 1, 0, 1, 0, 0, 0, 1] \
						for _ in range(y_numbering.shape[0])]
			logit = popular

		for b_idx, ob in enumerate(observed):
			pred, true = [],[]
			start = 0
			for a_idx, al in enumerate(self.attr_len):
				if not a_idx in tasks: continue
				end = start + al

				if sum(ob[start:end]):
					self.attr_cnt[a_idx] += 1
					p = np.argmax(logit[b_idx][start:end], 0) + start
					t = sum(y_numbering[b_idx][start:end])
					if p == t:
						self.attr_em[a_idx] += 1
					pred.append(p)
					true.append(t)
				start += al

			if pred and true:
				y_pred.append(pred)
				y_true.append(true)

		self.num_users += len(y_true)

		##
		for full in y_pred:
			for attr in full:
				self.ypc_counter[attr] += 1
		for full in y_true:
			for attr in full:
				self.ytc_counter[attr] += 1
		##

		for y in zip(y_pred, y_true):
			self.yp_counter[str(y[0])] += 1
			self.yt_counter[str(y[1])] += 1
			if np.array_equal(y[0], y[1]):
				self.y_em_counter[str(y[1])] += 1
			# calculate and accumulate hamming loss
			self.hm_acc += hamming_loss(y[1], y[0])

	def get_score(self):
		# for divide-by-zero exception
		if not self.num_users: num_users = 1
		else: num_users = self.num_users

		hm_loss = self.hm_acc / num_users

		macP = macR = macF1 = wP = wR = wF1 = 0

		# macro and weighted Precision
		for y, cnt in self.yp_counter.items():
			if y in self.y_em_counter.keys():
				macP += (self.y_em_counter[y] / cnt)
				if y in self.yt_counter.keys():
					wP += (self.y_em_counter[y] / cnt) * self.yt_counter[y]
		macP /= len(self.yt_counter)
		wP /= num_users

		# macro and weighted Recall
		for y, cnt in self.yt_counter.items():
			if y in self.y_em_counter.keys():
				wR += self.y_em_counter[y]
				macR += (self.y_em_counter[y] / cnt)
		macR /= len(self.yt_counter)
		wR /= num_users

		# calculate F1 using computed precision and recall.
		# this code includes exception.
		if macP == 0 and macR == 0:
			macP = macR = macF1 = 0
		else:
			macF1 = (2 * macP * macR) / (macP + macR)
		if wP == 0 and wR == 0:
			wP = wR = wF1 = 0
		else:
			wF1 = (2 * wP * wR) / (wP + wR)
		return hm_loss, macP, macR, macF1, wP, wR, wF1


	def summary(self, loss, step, epoch_end):
		hm, macP, macR, macF1, wP, wR, wF1 = self.get_score()
		if epoch_end:
			self.vis.add_scalar('training loss end', loss, step)
			self.vis.add_scalar('hamming loss end', hm, step)
			self.vis.add_scalar('macro precision end', macP, step)
			self.vis.add_scalar('macro recall end', macR, step)
			self.vis.add_scalar('macro f1 end', macF1, step)
			self.vis.add_scalar('weighted precision end', wP, step)
			self.vis.add_scalar('weighted recall end', wR, step)
			self.vis.add_scalar('weighted f1 end', wF1, step)

		else:
			self.vis.add_scalar('training loss', loss, step)
			self.vis.add_scalar('hamming loss', hm, step)
			self.vis.add_scalar('macro precision', macP, step)
			self.vis.add_scalar('macro recall', macR, step)
			self.vis.add_scalar('macro f1', macF1, step)
			self.vis.add_scalar('weighted precision', wP, step)
			self.vis.add_scalar('weighted recall', wR, step)
			self.vis.add_scalar('weighted f1', wF1, step)
		for model in self.model:
			variables = model.named_parameters()
			for i,v in enumerate(variables):
				self.variable_summary(*v,step)


	def variable_summary(self, name, var, step):
		mean = torch.mean(var)
		self.vis.add_scalar(name+'/mean', mean,step)

		stddev = torch.sqrt(torch.mean(torch.pow((var - mean),2)))
		self.vis.add_scalar(name+'/stddev', stddev,step)
		self.vis.add_scalar(name+'/max', torch.max(var),step)
		self.vis.add_scalar(name+'/min', torch.min(var),step)
		self.vis.add_histogram(name+'/histogram', var,step,bins='doane')
