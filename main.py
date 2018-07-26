
import argparse
import logging
import numpy as np
import random
import sys
import time
import torch
from torch.utils.data import DataLoader
import uuid

from dataset import DemoAttrDataset, batchify, SortedBatchSampler
from exp import Experiment

def get_args():
	parser = argparse.ArgumentParser()

# data
	parser.add_argument('--dataset', type=str, default='ocb',
						help="[beiren, ocb]")
	parser.add_argument('--data-path', type=str, default="./data/preprd/",
						help="")
	parser.add_argument('--rand-seed', type=int, default=1)
	parser.add_argument('--data-shuffle', type=int, default=1)
	parser.add_argument('--data-sampling', type=int, default=0)
	parser.add_argument('--sample_type', type=str, default='full')

# task settings
	parser.add_argument('--partial-ratio', type=str, default='50')
	parser.add_argument('--partial-training', type=int, default=1)
	parser.add_argument('--partial-eval', type=int, default=1)
	parser.add_argument('--task-type', type=str, default='partial',
						help="[partial, new_user]")
	parser.add_argument('--tasks', type=int, nargs='+')

# optimizations
	parser.add_argument('--opt', type=str, default='Adam',
						help="Adam / RMSprop / SGD / Adagrad / Adadelta / Adamax")
	parser.add_argument('--amsgrad', type=int, default=0)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--use-negsample', type=int, default=1)

# embeddings
	parser.add_argument('--item-emb-size', type=int, default=100)

# training parameters
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--learning-rate', type=float, default=0.001)
	parser.add_argument('--user_emb_dim', type=int, default=100)
	parser.add_argument('--num_negs', type=int, default=1)
	parser.add_argument('--max-epoch', type=int, default=20000)
	parser.add_argument('--grad-max-norm', type=float, default=5)
	parser.add_argument('--num-batches', type=int, default=20)

# model's parameters
	parser.add_argument('--model-type', type=str, default='Average',
						help="[POP, Average, RNN, TAN]")
	parser.add_argument('--rnn-type', type=str, default='LSTM')
	parser.add_argument('--rnn-size', type=int, default=70)
	parser.add_argument('--rnn-layer', type=int, default=2)
	parser.add_argument('--rnn-drop', type=float, default=0.2)
	parser.add_argument('--attention-layer', type=int, default=1,
						help="you can choose [1 or 2] when using TAN model")
	parser.add_argument('--learning-form', type=str, default='structured',
						help="[separated, structured]")

# debugging and analysis
	parser.add_argument('--save-log', type=int, default=0)
	parser.add_argument('--save-output', type=int, default=0)
	parser.add_argument('--print-per-step', type=int, default=99999)
	parser.add_argument('--vis-per-step', type=int, default=100)
	parser.add_argument('--rand-search', type=int, default=0)

# regularization
	parser.add_argument('--early-stop', type=str, default='va_wF1',
						help="you can combinate two words which are from in each set: [mic, mac, w] and [P, R, F1]")
	parser.add_argument('--weight-decay', type=float, default=0.0001)
	parser.add_argument('--lr-decay', type=float, default=0.0)

	args = parser.parse_args()
	return args

def run_experiment(args, logger):

	# generate a data loader for validation set
	"""
	valid_loader = DataLoader(
					dataset=DemoAttrDataset(
						logger,
						'valid',
						args.data_path+'valid_'+args.task_type+args.partial_ratio+'.json',
						args.data_path+'test_'+args.task_type+args.partial_ratio+'.json',
					),
					batch_size=args.batch_size,
					shuffle=False,
					num_workers=2,
					collate_fn=batchify)
	"""
	test_loader = DataLoader(
	             dataset=DemoAttrDataset(logger, args.task_type, 'test',
								args.data_path+args.dataset+'/test.json'),
	             batch_size=args.batch_size,
	             shuffle=False,
	             num_workers=2,
				 collate_fn=batchify)

	exp = Experiment(args, logger)
	train_dataset = DemoAttrDataset(
						logger,
						args.task_type,
						'train',
						#args.data_path+'train_'+args.task_type+args.partial_ratio+'.json',
						args.data_path+args.dataset+'/train.json',
					)
	train_sampler = SortedBatchSampler(train_dataset.lengths(),
									args.batch_size,
									shuffle=True)
	train_loader = DataLoader(
					dataset=train_dataset,
					batch_size=args.batch_size,
					shuffle=False,
					#sampler=train_sampler,
					num_workers=2,
					collate_fn=batchify)

	max_score = max_loss = stop_cnt = 0
	max_macP = max_macR = max_macF1 = max_wP = max_wR = max_wF1 = 0
	pre_wR = 0
	for epoch in range(args.max_epoch):
		logger.info("++ Epoch : {} ++ \n".format(epoch+1))
		sample_attr = epoch % 5
		if len(args.tasks)==1:
			# if task is for one attr sample with that attr
			sample_attr = args.tasks[0]
		
		if epoch % 2 == 0 and args.data_sampling: 
			train_dataset.sample_data_cls()
			train_dataset.pick_batch_data(args.num_batches, args.batch_size)
		if epoch % 2 == 1 and args.data_sampling:
			#train_dataset.sample_data_cls()
			train_dataset.pick_batch_data(args.num_batches, args.batch_size)
			#train_dataset.shuffle_data()
			#train_dataset.sample_subset(1000)

		tr_t0 = time.clock()
		tr_loss, tr_hm, \
		tr_macP, tr_macR, tr_macF1, tr_wP, tr_wR, tr_wF1 = exp.run_epoch(epoch, train_loader, sample_attr,
														args.data_sampling, trainable=True)
		tr_t1 = time.clock()

		va_t0 = time.clock()
		va_loss, va_hm, \
		va_macP, va_macR, va_macF1, va_wP, va_wR, va_wF1 = exp.run_epoch(epoch, test_loader, sample_attr,
														args.data_sampling, trainable=False)
		va_t1 = time.clock()
		logger.info("### Training # Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
					.format(tr_loss, tr_t1-tr_t0, tr_hm))
		logger.info("# macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
					.format(tr_macP, tr_macR, tr_macF1))
		logger.info("# weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
					.format(tr_wP, tr_wR, tr_wF1))
		logger.info("%%% Validation % Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
					.format(va_loss, va_t1-va_t0, va_hm))
		logger.info("% macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
					.format(va_macP, va_macR, va_macF1))
		logger.info("% weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
					.format(va_wP, va_wR, va_wF1))
		
		# early stop
		if max_score < va_wF1:
			max_epoch = epoch+1
			max_score = va_wF1
			max_loss = va_loss
			max_hm = va_hm
			max_macP = va_macP
			max_macR = va_macR
			max_macF1 = va_macF1
			max_wP = va_wP
			max_wR = va_wR
			max_wF1 = va_wF1
			stop_cnt = 0
		elif max_score == va_wF1:
			# lr decay
			exp.adjust_lr()
			stop_cnt += 1
		if args.model_type == 'POP': break
		if stop_cnt >= 5 and args.early_stop:
			return max_epoch, max_loss, max_hm, \
					max_macP, max_macR, max_macF1, \
					max_wP, max_wR, max_wF1
	return max_epoch, max_loss, max_hm, \
			max_macP, max_macR, max_macF1, \
			max_wP, max_wR, max_wF1


def main():
	# get all arguments
	args = get_args()

	if args.task_type == 'partial':
		args.partial_training = args.partial_eval = 1
	elif args.task_type == 'new_user':
		args.partial_training = args.partial_eval = 0
	if args.model_type == 'TAN' and args.attention_layer == 2:
		args.learning_form = 'seperated'

	if args.tasks == None and args.dataset == 'beiren':
		args.tasks = [0,1,2,3,4]
	elif args.tasks == None and args.dataset == 'ocb':
		args.tasks = [0,1,2]

#if args.model_type == 'TAN' and args.attention_layer == 2:
#args.learning_form = 'seperated'

#run_mfdm_exp(args)

# set random seeds
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.manual_seed(args.rand_seed)

# set a logger
	model_id = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
	formatter = logging.Formatter('%(asctime)s: %(message)s ', '%m/%d/%Y %I:%M:%S %p')
	logger = logging.getLogger(model_id)
	logger.setLevel(logging.INFO)
	streamHandler = logging.StreamHandler()
	streamHandler.setFormatter(formatter)
	logger.addHandler(streamHandler)
	if args.save_log:
		fileHandler = logging.FileHandler('./save/log/'+model_id+'.log')
		fileHandler.setFormatter(formatter)
		logger.addHandler(fileHandler)
		logger.info('log file : ./save/log/'+model_id+'.log')
	logger.info(args)

	ep, loss, hm, macP, macR, macF1, wP, wR, wF1 = run_experiment(args, logger)
	logger.info("[Final score - ep:{}] Loss={:5.3f}, Hamming={:2.3f}"
				.format(ep, loss, hm))
	logger.info("[ macro ] macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
				.format(macP, macR, macF1))
	logger.info("[ weighted ] wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f}"
				.format(wP, wR, wF1))

if __name__ == '__main__':
	main()
