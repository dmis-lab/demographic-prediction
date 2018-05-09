
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

global label_size
label_size = 18

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
    parser.add_argument('--item-emb-size', type=int, default=100)
    
    # training parameters
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--learning-rate', type=float, default=0.0025)
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
    parser.add_argument('--var-decay', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    
    args = parser.parse_args()
    return args


def run_experiment(args, logger):
    train_loader = DataLoader(
                    dataset=DemoAttrDataset('train',
                                    args.data_path+'train.json',
                                    logger),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=batchify)
    valid_loader = DataLoader(
                    dataset=DemoAttrDataset('valid',
                                    args.data_path+'valid.json',
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
    
    max_score = f_hm  = f_p = f_r = f_f1 = patience = 0
    for epoch in range(args.max_epoch):
        t0 = time.clock()
        
        tr_loss, tr_hm, tr_p, tr_r, tr_f1 = exp.run_epoch(train_loader, 
                                                        trainable=True)
        va_loss, va_hm, va_p, va_r, va_f1 = exp.run_epoch(valid_loader, 
                                                        trainable=False)
        t1 = time.clock()

        logger.info("++++++++++ epoch: {} ++++++++++".format(epoch+1))
        logger.info("[Training] Loss={:5.3f}, time:{:5.2f}"
                            .format(tr_loss, t1-t0))
        logger.info("Hamming={:4.2f}, P:{:4.2f}, R:{:4.2f}, F1:{:4.2f}"
                            .format(tr_hm, tr_p, tr_r, tr_f1))
        logger.info("[Validation] Loss={:5.3f}, time:{:5.2f}"
                            .format(va_loss, t1-t0))
        logger.info("Hamming={:4.2f}, P:{:4.2f}, R:{:4.2f}, F1:{:4.2f}"
                            .format(va_hm, va_p, va_r, va_f1))
        # early stop
        if args.early_stop == 'hm': score = va_hm
        elif args.early_stop == 'p': score = va_p
        elif args.early_stop == 'r': score = va_r
        elif args.early_stop == 'f1': score = va_f1
        else: score = va_loss

        if max_score < score:
            max_score = score
            f_loss = va_loss
            f_hm = va_hm
            f_p = va_p
            f_r = va_r
            f_f1 = va_f1
            patience = 0
        else: patience += 1
        if patience >= 5 and args.early_stop:
            return (epoch+1), f_loss, f_hm, f_p, f_r, f_f1
    return (epoch+1), f_loss, f_hm, f_p, f_r, f_f1


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


