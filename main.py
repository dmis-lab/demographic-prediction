
import argparse
import logging
import numpy as np
import random
import sys
import time
import torch
from torch.utils.data import DataLoader
import uuid

from dataset import DemoAttrDataset, batchify
from exp import Experiment

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data-path', type=str, default="./data/preprd/",
                        help="")
    parser.add_argument('--rand-seed', type=int, default=1)
    parser.add_argument('--data-shuffle', type=int, default=1)

    # optimizations
    parser.add_argument('--opt', type=str, default='Adam',
                        help="Adam / RMSprop / SGD / Adagrad / Adadelta / Adamax")
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # embeddings
    parser.add_argument('--item-emb-size', type=int, default=200)
    
    # training parameters
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--max-epoch', type=int, default=20)
    
    # model's parameters
    parser.add_argument('--rnn-size', type=int, default=100)
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--rnn-drop', type=float, default=0.2)
    parser.add_argument('--char-drop', type=float, default=0.2)
  
    # debugging and analysis
    parser.add_argument('--save-log', type=int, default=0)
    parser.add_argument('--save-output', type=int, default=0)
    parser.add_argument('--print-per-step', type=int, default=100)
    parser.add_argument('--rand-search', type=int, default=0)

    # regularization
    parser.add_argument('--early-stop', type=int, default=1)
    parser.add_argument('--var-decay', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    args = parser.parse_args()
    return args


def run_experiment(args, logger):
    train_loader = DataLoader(
                    dataset=DemoAttrDataset(args, logger, 'train'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=batchify)
    valid_loader = DataLoader(
                    dataset=DemoAttrDataset(args, logger, 'valid'),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2,
                    collate_fn=batchify)
    #test_loader = DataLoader(
    #                dataset=DemoAttrDataset(args, logger, 'test'),
    #                batch_size=args.batch_size,
    #                shuffle=False,
    #                num_workers=2)

    exp = Experiment(args, logger)
    
    max_loss = max_f1 = max_p = max_r = stop_cnt = 0
    for epoch in range(args.max_epoch):
        logger.info("====================================")
        logger.info("-- Train mode -- epoch: {}".format(epoch + 1))
        logger.info("====================================")
        t0 = time.clock()
        
        train_result = exp.run_epoch(train_loader, 
                                    (epoch+1), 
                                    trainable=True)
        valid_result = exp.run_epoch(valid_loader,
                                    (epoch+1))
        t1 = time.clock()
        logger.info("\n++++++[epoch {}]++++++".format(epoch + 1))
        logger.info("[Train] Loss={:5.3f}".format(train_loss))
        logger.info("\t(word-level) F1={:4.2f}/{:4.2f}, P:{:4.2f}/{:4.2f}, R:{:4.2f}/{:4.2f}"
                .format(train_f1, train_id_f1, train_p, train_id_p, train_r, train_id_r))
        logger.info("[Valid] Loss={:5.3f}".format(valid_loss))
        logger.info("\t(word-level) F1={:4.2f}/{:4.2f}, P:{:4.2f}/{:4.2f}, R:{:4.2f}/{:4.2f}"
                .format(valid_f1, valid_id_f1, valid_p, valid_id_p, valid_r, valid_id_r))
        logger.info("\t(binary) F1={:4.2f}, P:{:4.2f}, R:{:4.2f}"
                .format(valid_bi_f1, valid_bi_p, valid_bi_r))
        logger.info("+++++ time : {:6.2f} +++++\n".format(t1-t0))
        
        # early stop
        if max_f1 < valid_f1:
            max_f1 = valid_f1
            max_loss = valid_loss
            max_p = valid_p
            max_r = valid_r
            stop_cnt = 0
        else: stop_cnt += 1
        if stop_cnt >= 5 and args.early_stop:
            return (epoch+1), max_loss, max_f1, max_p, max_r
    return (epoch+1), max_loss, max_f1, max_p, max_r


def main():
    # get all arguments
    args = get_args()
    
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

    ep, loss, f1, p, r = run_experiment(args, logger)
    

if __name__ == '__main__':
    main()


