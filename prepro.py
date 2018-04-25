# This code is for data preprocessing.

import argparse
from collections import Counter
import json
import logging
from multiprocessing import Pool
import numpy as np
import os
import random
import sys
import time
from tqdm import tqdm

# tokens
#w_pad = '@w_p@'
#w_unk = '#w_unk#'



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
    parser.add_argument('--data-split', type=str, default='0.8, 0.1, 0.1',
                        help="The ratio of training, validation, test sets. \
                            You must follow the format of this argument \
                            , which defined as '0.8, 0.1, 0.1' \
                            the ratios of each data set are separated by comma \
                            and the summation of them is 1.")

    # hyper-parameters
    parser.add_argument('--rand-seed', type=int, default=1,
                        help="set the random seed of the random module")
    
    args = parser.parse_args()
    return args

def build_dict(args, logger, counter):
    return

def split_data(args, logger, data):
    total_idx = list(range(len(data[0])))
    random.shuffle(total_idx)
    
    spl_ratio = np.asarray(args.data_split.split(',')).astype(float)
    cum_ratio = np.cumsum(spl_ratio)
    assert cum_ratio[-1] == 1

    whole = []
    for i, r in enumerate(cum_ratio):
        if i == 0:
            idx = total_idx[0:int(len(total_idx)*r)]
        else:
            idx = total_idx[int(len(total_idx)*cum_ratio[i-1]):int(len(total_idx)*r)]
        x = data[0][idx]
        y = data[1][idx]
        whole.append((x,y))
        
    
    logger.info("Data splitting is done. The number of training samples is '{}'"
            .format(len(whole[0][0])))

    logger.info("Counting of user histories in the training set start!")
    history_counter = Counter()
    for h in tqdm(whole[0][0]):
        history_counter += Counter(h)
    return whole, history_counter

def read_data(args, logger):
    f = open(os.path.join(args.load_path, 'data.txt'))
    label, history = [], []
    history_cnt = 0
    while True:
        line = f.readline()
        if not line: break
        line = line.split('|')
        l = np.asarray(line[0].split(',')).astype(int)
        h = np.asarray(line[1].split(',')).astype(int)
        
        label.append(l)
        history.append(h)
        
        history_cnt += len(h)
        if len(label) % 10000 == 0: 
            logger.info("{} samples have been read".format(len(label)))
    logger.info("Total number of samples : {}".format(len(label)))
    logger.info("Average of user histories : {:4.3f}"
            .format(history_cnt/len(label)))
    
    label = np.asarray(label)
    history = np.asarray(history)
    assert len(label) == len(history)
    return history, label

def main():
    # get all arguments
    args = get_args()

    # set random seed
    random.seed(args.rand_seed)

    # set logger
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
    splitted_data, history_counter = split_data(args, logger, (history, label))
    
    # build a dictionary
    history_dict = build_dict(args, logger, history_counter)

    # save the proprecessed data sets
    #pickle.dump(history_dict, 
    #        open(os.path.join(args.save_path, 'dict.json'), 'w'))
    for d_idx, d in enumerate(splitted_data):
        json.dump({'history': d[0], 'label': d[1]}, 
                open(os.path.join(args.save_path, 'data_'+str(d_idx)+'.json'), 'w'), 
                cls=NumpyEncoder)

    logger.info("Data preprocessing is done successfully!")

if __name__ == '__main__':
    main()


