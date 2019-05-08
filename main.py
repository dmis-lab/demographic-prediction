
import logging
import numpy as np
import random
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import uuid

from dataset import DemoAttrDataset, batchify
from exp import Experiment
from config import get_args

def run_experiment(args, logger):

    # generate a data loader for validation set
    if args.do_validation:
        eval_loader = DataLoader(
                         dataset=DemoAttrDataset(
                                logger=logger, 
                                data_type='valid',
                                data_path=os.path.join(args.data_path, args.dataset),
                                task_type=args.task_type,
                                model_type=args.model_type),
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=batchify)
    else:
        eval_loader = DataLoader(
                         dataset=DemoAttrDataset(
                                logger=logger, 
                                data_type='test',
                                data_path=os.path.join(args.data_path, args.dataset),
                                task_type=args.task_type,
                                model_type=args.model_type),
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=batchify)

    train_dataset = DemoAttrDataset(
                        logger=logger,
                        data_type='train',
                        data_path=os.path.join(args.data_path, args.dataset),
                        task_type=args.task_type,
                        model_type=args.model_type)
    train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=batchify)

    exp = Experiment(args, logger)

    max_score = max_loss = stop_cnt = 0
    max_macP = max_macR = max_macF1 = max_wP = max_wR = max_wF1 = 0
    pre_wR = 0
    for epoch in range(args.max_epoch):
        logger.info("++ Epoch : {} ++ \n".format(epoch+1))

        tr_t0 = time.clock()
        tr_loss, tr_hm, \
        tr_macP, tr_macR, tr_macF1, tr_wP, tr_wR, tr_wF1 = \
                exp.run_epoch(epoch, train_loader, args.dataset, trainable=True)
        tr_t1 = time.clock()

        eval_t0 = time.clock()
        eval_loss, eval_hm, \
        eval_macP, eval_macR, eval_macF1, eval_wP, eval_wR, eval_wF1 = \
                exp.run_epoch(epoch, eval_loader, args.dataset, trainable=False)
        eval_t1 = time.clock()

        # print training scores
        logger.info("### Training # Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
                    .format(tr_loss, tr_t1-tr_t0, tr_hm))
        logger.info("# macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
                    .format(tr_macP, tr_macR, tr_macF1))
        logger.info("# weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
                    .format(tr_wP, tr_wR, tr_wF1))

        # print val/test scores
        logger.info("%%% Evaluation % Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
                    .format(eval_loss, eval_t1-eval_t0, eval_hm))
        logger.info("% macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
                    .format(eval_macP, eval_macR, eval_macF1))
        logger.info("% weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
                    .format(eval_wP, eval_wR, eval_wF1))

        # early stop
        if max_score < eval_wF1:
            max_epoch = epoch+1
            max_score = eval_wF1
            max_loss = eval_loss
            max_hm = eval_hm
            max_macP = eval_macP
            max_macR = eval_macR
            max_macF1 = eval_macF1
            max_wP = eval_wP
            max_wR = eval_wR
            max_wF1 = eval_wF1
            #model_params = exp.model.item_emb.weight
            stop_cnt = 0
        else:
            # lr decay
            exp.adjust_lr()
            stop_cnt += 1
        if args.model_type == 'POP': break

        if stop_cnt >= 5 and not args.no_early_stop:
            return max_epoch, max_loss, max_hm, \
                    max_macP, max_macR, max_macF1, \
                    max_wP, max_wR, max_wF1
    return max_epoch, max_loss, max_hm, \
            max_macP, max_macR, max_macF1, \
            max_wP, max_wR, max_wF1


def main():
    # get all arguments
    args = get_args()

    # set random seeds
    #np.random.seed(args.rand_seed)
    #random.seed(args.rand_seed)
    #torch.manual_seed(args.rand_seed)

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
    if args.save_log:
        logger.info('log file : ./save/log/'+model_id+'.log')


if __name__ == '__main__':
    main()

