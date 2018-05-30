
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
from mf2demo import *

global label_size
label_size = 18

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data-path', type=str, default="./data/preprd/ordered/",
                        help="")
    parser.add_argument('--rand-seed', type=int, default=1)
    parser.add_argument('--data-shuffle', type=int, default=1)

    # task settings
    parser.add_argument('--partial-ratio', type=str, default='90')
    parser.add_argument('--task', type=str, default='partial',
                        help="[partial, new_user]")

    # optimizations
    parser.add_argument('--opt', type=str, default='SGD',
                        help="Adam / RMSprop / SGD / Adagrad / Adadelta / Adamax")
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)

    # embeddings
    parser.add_argument('--item-emb-size', type=int, default=100)

    # training parameters
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--user_emb_dim', type=int, default=100)
    parser.add_argument('--num_negs', type=int, default=4)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--grad-max-norm', type=float, default=5)

    # model's parameters
    parser.add_argument('--rnn-type', type=str, default='LSTM')
    parser.add_argument('--rnn-size', type=int, default=70)
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--rnn-drop', type=float, default=0.2)
    parser.add_argument('--char-drop', type=float, default=0.2)

    # debugging and analysis
    parser.add_argument('--save-log', type=int, default=0)
    parser.add_argument('--save-output', type=int, default=0)
    parser.add_argument('--print-per-step', type=int, default=100)
    parser.add_argument('--rand-search', type=int, default=0)

    # regularization
    parser.add_argument('--early-stop', type=str, default='f1')
    parser.add_argument('--weight-decay', type=float, default=0.3)

    args = parser.parse_args()
    return args


def run_experiment(args, logger):
    train_dataset = DemoAttrDataset('train',
                    args.data_path+'train_'+args.task+args.partial_ratio+'.json',
                    logger)
    train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                    args.batch_size,
                                    shuffle=True)

    train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=train_sampler,
                    num_workers=2,
                    collate_fn=batchify)

    # generate a data loader for validation set
    valid_loader = DataLoader(
                    dataset=DemoAttrDataset('valid',
                            args.data_path+'valid_'+args.task+args.partial_ratio+'.json',
                            logger),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2,
                    collate_fn=batchify)
    #test_loader = DataLoader(
    #                dataset=DemoAttrDataset(args, logger, 'test'),
    #                batch_size=args.batch_size,
    #                shuffle=False,
    #                num_workers=2)

    exp = Experiment(args, logger, label_size)

    max_loss = max_f1 = max_p = max_r = stop_cnt = 0
    for epoch in range(args.max_epoch):
        logger.info("++++++++++ epoch: {} ++++++++++".format(epoch+1))
        tr_t0 = time.clock()
        tr_loss, tr_hm, tr_p, tr_r, tr_f1 = exp.run_epoch(train_loader,
                                                        trainable=True)
        tr_t1 = time.clock()

        va_t0 = time.clock()
        va_loss, va_hm, va_p, va_r, va_f1 = exp.run_epoch(valid_loader,
                                                        trainable=False)
        va_t1 = time.clock()

        logger.info("[Training] Loss={:5.3f}, time:{:5.2}, Hamming={:4.2f}, P:{:4.2f}, R:{:4.2f}, F1:{:4.2f}"
                            .format(tr_loss, tr_t1-tr_t0, tr_hm, tr_p, tr_r, tr_f1))
        logger.info("[Validation] Loss={:5.3f}, time:{:5.2f}, Hamming={:4.2f}, P:{:4.2f}, R:{:4.2f}, F1:{:4.2f}"
                            .format(va_loss, va_t1-va_t0, va_hm, va_p, va_r, va_f1))
        # early stop
        if max_f1 < va_f1:
            max_f1 = va_f1
            max_loss = va_loss
            max_p = va_p
            max_r = va_r
            stop_cnt = 0
        else: stop_cnt += 1
        if stop_cnt >= 5 and args.early_stop:
            return (epoch+1), max_loss, max_f1, max_p, max_r
    return (epoch+1), max_loss, max_f1, max_p, max_r


def main():
    # get all arguments
    args = get_args()

    run_mfdm_exp(args)

    # set random seeds
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    # set a logger
    model_id = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    formatter = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
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

    ep, loss, hm, f1, p, r = run_experiment(args, logger)


if __name__ == '__main__':
    main()
