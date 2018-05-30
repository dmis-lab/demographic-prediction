# ---------------------------------------------
# This code is for data preprocessing.

# all attributes and the numbers of possible classes corresponding to the attributes
# gender : male / female
# age :  young / adult / middle-age / old
# marital status : single / married
# income : ultra-low / low / medium / high
# education level : doctor / master / bachelor / college / high school / middle school

# ---------------------------------------------

import argparse
from collections import Counter
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import sys
import time
from tqdm import tqdm

global NULL, UNK
NULL = '<NULL>'
UNK = '<UNK>'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--load-path', type=str, default='./data/raw/',
                        help="the path of data to be loaded")
    parser.add_argument('--save-path', type=str, default='./data/preprd/',
                        help="the path of data to be saved")

    # experimental settings
    parser.add_argument('--data-split', type=str, default='0.8, 0.1, 0.1',
                        help="The ratio of training, validation, test sets, and so on. \
                            You must follow the format of this argument \
                            , which defined as '0.8, 0.1, 0.1' \
                            the ratios of each data set are separated by comma \
                            and the summation of them is 1.")
    parser.add_argument('--task', type=str, default="partial",
                        help="[partial, new_user]")
    parser.add_argument('--partial-ratio', type=float, default=0.1,
                        help="This argument can be used to set the observed ratio of \
                            demographic attributes randomly. \
                            The range of the ratio is [0.1, 0.9] with the step lengh as 0.1. \
                            More details are in the paper \
                            'Your Cart tells You, P.Wang, WSDM 2016'")
    parser.add_argument('--structured', type=int, default=1,
                        help="to concatenate all the one-hot representation of attribute labels \
                                for multi-task multi-class prediction.")

    # hyper-parameters
    parser.add_argument('--rand-seed', type=int, default=1,
                        help="set the random seed of the random module")
    parser.add_argument('--count-th', type=int, default=0,
                        help="")

    args = parser.parse_args()
    return args

def rep_onehot(args, data, attr_cls, idx_gap):
    for i, d in enumerate(data):
        for j, l in enumerate(d[1]):
            onehot = []
            observed = []
            k = -1
            for a, b, c in zip(l, attr_cls, idx_gap):
                k += 1
                oh_tmp = np.zeros(len(b)).astype(int).tolist()
                oh_tmp[a-c] = 1
                onehot.append(oh_tmp)
                
                if d[2][j][k]:
                    ob_tmp = np.ones(len(b)).astype(int).tolist()
                else:
                    ob_tmp = np.zeros(len(b)).astype(int).tolist()
                observed.append(ob_tmp)
            if args.structured:
                data[i][1][j] = list(chain.from_iterable([oh for oh in onehot]))
                data[i][2][j] = list(chain.from_iterable([ob for ob in observed]))
    return data

def item2idx(args, logger, data, dictionary):
    for i, d in enumerate(data):
        logger.info("item2idx processing . . .")
        for j, h in tqdm(enumerate(d[0])):
            for k, item, in enumerate(h):
                if item in dictionary:
                    data[i][0][j][k] = dictionary.index(item)
                else:
                    data[i][0][j][k] = dictionary.index(UNK)
    return data

def build_dict(args, counter):
    dict = [NULL, UNK]
    for i, c in counter.items():
        if c > args.count_th:
            dict.append(i)
    return dict

def build_dataset(args, logger, data):
    # check the number of possible classes for each attribute
    attr_cls = []
    idx_gap = []
    all_y = np.asarray(data[1])
    for j in range(all_y.shape[1]):
        attr_cls.append(np.unique(np.transpose(all_y, (1,0))[j]))
        logger.info("The {}-th attribute has {} possible classes"
                .format((j+1), len(attr_cls[j])))
    for a in attr_cls:
        if min(a) == 0:
            idx_gap.append(0)
        else:
            idx_gap.append(min(a))
    
    # generate partial labels
    def observe_randomly(num_data, num_attr, prob):
        observed = []
        for i in range(num_data):
            tmp = []
            for j in range(num_attr):
                tmp.append(random.random() < prob)
            observed.append(tmp)
        return observed
    observed = observe_randomly(len(data[1]), len(attr_cls), args.partial_ratio)
    
    ###################
    # build data sets #
    ###################

    # shuffle data indices
    total_idx = list(range(len(data[0])))
    if args.task == 'new_user':
        random.shuffle(total_idx)

    spl_ratio = np.asarray(args.data_split.split(',')).astype(float)
    
    whole = []
    if args.task == 'partial':
        tr_idx = total_idx
        va_ratio = spl_ratio[1] / (spl_ratio[1] + spl_ratio[2])
        va_idx = total_idx[0:int(len(total_idx)*va_ratio)]
        te_idx = total_idx[int(len(total_idx)*va_ratio):]
    elif args.task == 'new_user':
        tr_idx = total_idx[:int(len(total_idx)*spl_ratio[0])]
        va_idx = total_idx[int(len(total_idx)*spl_ratio[0]):\
                            int(len(total_idx)*(spl_ratio[0]+spl_ratio[1]))]
        te_idx = total_idx[int(len(total_idx)*(spl_ratio[0]+spl_ratio[1])):]
    else:
        print("Please check the task type. Only two choices are avalible. [partial, new_user]")
        sys.exit()
        
    tr_x = np.asarray(data[0])[tr_idx].tolist()
    tr_y = np.asarray(data[1])[tr_idx].tolist()
    tr_ob = np.asarray(observed)[tr_idx].tolist()
    va_x = np.asarray(data[0])[va_idx].tolist()
    va_y = np.asarray(data[1])[va_idx].tolist()
    va_ob = np.asarray(observed)[va_idx].tolist()
    te_x = np.asarray(data[0])[te_idx].tolist()
    te_y = np.asarray(data[1])[te_idx].tolist()
    te_ob = np.asarray(observed)[te_idx].tolist()
    
    whole.append((tr_x, tr_y, tr_ob))
    whole.append((va_x, va_y, va_ob))
    whole.append((te_x, te_y, te_ob))

    logger.info("Data splitting is done. The number of training samples is '{}'."
            .format(len(whole[0][0])))
    logger.info("The number of validation and test samples is '{}' and '{}'."
            .format(len(whole[1][0]), len(whole[2][0])))

    logger.info("Counting of user histories in the training set start!")
    history_counter = Counter()
    for h in tqdm(whole[0][0]):
        history_counter += Counter(h)
    return whole, history_counter, attr_cls, idx_gap

def read_data(args, logger):
    f = open(os.path.join(args.load_path, 'data.txt'))
    label, history = [], []
    history_cnt = 0
    while True:
        line = f.readline()
        if not line: break
        line = line.split('|')
        l = np.asarray(line[0].split(',')).astype(int).tolist()
        h = np.asarray(line[1].split(',')).astype(int).tolist()

        label.append(l)
        history.append(h)
        history_cnt += len(h)
        if len(label) % 10000 == 0:
            logger.info("{} samples have been read".format(len(label)))
    logger.info("Total number of samples : {}".format(len(label)))
    logger.info("Average of user histories : {:4.3f}"
            .format(history_cnt/len(label)))

    assert len(label) == len(history)
    return history, label

def main():
    # get all arguments
    args = get_args()

    # set a random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    # set a logger
    logger = logging.getLogger('spoilerLogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.info("All arguments : " + str(args))
    logger.info("Data preprocessing starts!")

    # read and get data
    history, label = read_data(args, logger)
    
    # shuffle and split the data into some sets
    splitted_data, history_counter, attr_cls, idx_gap \
            = build_dataset(args, logger, (history, label))

    attr_len = []
    for a in attr_cls:
        attr_len.append(len(a))

    # build a dictionary
    dictionary = build_dict(args, history_counter)

    # item 2 idx
    splitted_data = item2idx(args, logger, splitted_data, dictionary)

    # represent the labels by a one-hot encoding scheme
    # and convert the observed labels into maskings
    splitted_data = rep_onehot(args, splitted_data, attr_cls, idx_gap)

    # save the proprecessed data sets
    json.dump({'dict': dictionary, 'attr_len': attr_len},
            open(os.path.join(args.save_path, 
                'dict_' + args.task + str(int(args.partial_ratio*100)) + '.json'), 'w'),
            cls=NumpyEncoder)
    json.dump({'history': splitted_data[0][0], 
                'label': splitted_data[0][1],
                'observed': splitted_data[0][2]},
            open(os.path.join(args.save_path,
                'train_' + args.task + str(int(args.partial_ratio*100)) + '.json'), 'w'),
            cls=NumpyEncoder)
    json.dump({'history': splitted_data[1][0],
                'label': splitted_data[1][1],
                'observed': splitted_data[1][2]},
            open(os.path.join(args.save_path,
                'valid_' + args.task + str(int(args.partial_ratio*100)) + '.json'), 'w'),
            cls=NumpyEncoder)
    json.dump({'history': splitted_data[2][0],
                'label': splitted_data[2][1],
                'observed': splitted_data[2][2]},
            open(os.path.join(args.save_path,
                'test_' + args.task + str(int(args.partial_ratio*100)) + '.json'), 'w'),
            cls=NumpyEncoder)

    logger.info("Data preprocessing is done successfully!")

if __name__ == '__main__':
    main()
