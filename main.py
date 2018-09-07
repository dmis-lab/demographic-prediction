
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
from config import get_args

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
                            args.data_path+args.dataset+'/test_'+args.task_type),
                     batch_size=args.batch_size,
                     shuffle=False,
                     num_workers=2,
                     collate_fn=batchify)

    train_dataset = DemoAttrDataset(logger, args.task_type, 'train',
                                    args.data_path+args.dataset+'/train_'+args.task_type)
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

    exp = Experiment(args, logger)

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
        tr_macPs, tr_macRs, tr_macF1s, tr_wPs, tr_wRs, tr_wF1s = \
                exp.run_epoch(epoch, train_loader, args.dataset, sample_attr,
                            args.data_sampling, trainable=True)
        tr_t1 = time.clock()

        va_t0 = time.clock()
        va_loss, va_hm, \
        va_macPs, va_macRs, va_macF1s, va_wPs, va_wRs, va_wF1s = \
                exp.run_epoch(epoch, test_loader, args.dataset, sample_attr,
                            args.data_sampling, trainable=False)
        va_t1 = time.clock()

        # print training scores
        logger.info("### Training # Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
                    .format(tr_loss, tr_t1-tr_t0, tr_hm))
        for idx, tr_macP, tr_macR, tr_macF1, tr_wP, tr_wR, tr_wF1 \
            in zip(list(range(len(tr_macPs))), tr_macPs, tr_macRs, tr_macF1s, tr_wPs, tr_wRs, tr_wF1s):
            if not args.print_attr_score and idx != 0: continue
            if idx == 0: logger.info("<TOTAL>-----------------------")
            else: logger.info("<attribute {}>".format(idx))
            logger.info("# macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
                        .format(tr_macP, tr_macR, tr_macF1))
            logger.info("# weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
                        .format(tr_wP, tr_wR, tr_wF1))

        # print val/test scores
        logger.info("%%% Validation % Loss={:5.3f}, time:{:5.2}, Hamming={:2.3f}"
                    .format(va_loss, va_t1-va_t0, va_hm))
        for idx, va_macP, va_macR, va_macF1, va_wP, va_wR, va_wF1 \
            in zip(list(range(len(va_macPs))), va_macPs, va_macRs, va_macF1s, va_wPs, va_wRs, va_wF1s):
            if not args.print_attr_score and idx != 0: continue
            if idx == 0: logger.info("<TOTAL>-----------------------")
            else: logger.info("<attribute {}>".format(idx))
            logger.info("% macro - macP:{:2.3f}, macR:{:2.3f}, macF1:{:2.3f}"
                        .format(va_macP, va_macR, va_macF1))
            logger.info("% weighted - wP:{:2.3f}, wR:{:2.3f}, wF1:{:2.3f} \n"
                        .format(va_wP, va_wR, va_wF1))

        # early stop
        if max_score < va_wF1s[0]:
            max_epoch = epoch+1
            max_score = va_wF1s[0]
            max_loss = va_loss
            max_hm = va_hm
            max_macP = va_macPs[0]
            max_macR = va_macRs[0]
            max_macF1 = va_macF1s[0]
            max_wP = va_wPs[0]
            max_wR = va_wRs[0]
            max_wF1 = va_wF1s[0]
            #model_params = exp.model.item_emb.weight
            stop_cnt = 0
        else:
            # lr decay
            exp.adjust_lr()
            stop_cnt += 1
        if args.model_type == 'POP': break

        if stop_cnt >= 5 and args.early_stop:
            #torch.save('./save/emb/avg')
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
    #if args.model_type == 'TAN' and args.attention_layer == 2:
    #	args.learning_form = 'seperated'

    if args.tasks == None and args.dataset == 'beiren':
        args.tasks = [0,1,2,3,4]
    elif args.tasks == None and args.dataset == 'ocb':
        args.tasks = [0,1,2]
    elif args.tasks == None and args.dataset == 'movieLens':
        args.tasks = [0,1,2]
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
    logger.info('log file : ./save/log/'+model_id+'.log')

if __name__ == '__main__':
    main()
