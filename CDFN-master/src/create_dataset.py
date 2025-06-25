import sys
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call, CalledProcessError

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def CMDC_PHQ9_labels(scores):
    labels = []
    for score in scores:
        if score <= 4:
            labels.append(0)  # 正常
        elif 5 <= score <= 9:
            labels.append(1)  # 轻度
        elif 10 <= score <= 14:
            labels.append(2)  # 中度
        elif 15 <= score <= 19:
            labels.append(3)  # 重度
        else:
            labels.append(4)  # 非常严重
    return labels


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK






class MOSI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))  #添加到 Python 解释器的模块搜索路径中
        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # 存储数据
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')  #从指定的train.pkl 文件中加载数据
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:
            print("找不到MOSI数据")
            exit()

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:
            print("找不到MOSEI数据")
            exit()


    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class UR_FUNNY:
    def __init__(self, config):

        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:
            print("找不到UR_FUNNY数据")
            exit()



    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class CMDC:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            data = load_pickle(DATA_PATH + '/' + config.cross_validation + '.pkl')
            describe = 'describe: (words, visual, acoustic, wordtxt), label_PHQ-9, describe)'
            self.train = []
            for (ex_index, example) in enumerate(data["train"]):
                (visual, acoustic, words), (label_id_class, label_id), wordtxt = example
                data_tuple = [((words, visual, acoustic, wordtxt), np.array([[float(label_id)]], dtype=np.float32), describe)]
                self.train.extend(data_tuple)
            self.dev = data["valid"]
            self.test = []
            for (ex_index, example) in enumerate(data["test"]):
                (visual, acoustic, words), (label_id_class, label_id), wordtxt = example
                data_tuple = [((words, visual, acoustic, wordtxt), np.array([[float(label_id)]], dtype=np.float32), describe)]
                self.test.extend(data_tuple)

            self.word2id = None
            self.pretrained_emb = None

            # 继续写，调整数据顺序，和mosi一样的

        except:
            print("N0 CMDC file")

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class IEMOCAP:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)
        # If cached data if already exists
        try:
            data = load_pickle(DATA_PATH + '/'  + 'iemocap_data.pkl')
            describe = 'describe: (txt:s*300, visual:s*35, acoustic:s*74), label:4*2, describe)'
            train_labels = data["train"]["labels"].astype(np.float32)
            train_text = data["train"]["text"].astype(np.float32)
            train_text[train_text == -np.inf] = 0
            train_audio = data["train"]["audio"].astype(np.float32)
            train_audio[train_audio == -np.inf] = 0
            train_vision = data["train"]["vision"].astype(np.float32)
            train_vision[train_vision == -np.inf] = 0
            self.train = []
            for i in range(train_vision.shape[0]):
                label = np.argmax(train_labels[i], -1)
                data_tuple = [((train_text[i], train_vision[i], train_audio[i]), label, describe)]
                self.train.extend(data_tuple)

            valid_labels = data["valid"]["labels"].astype(np.float32)
            valid_text = data["valid"]["text"].astype(np.float32)
            valid_text[valid_text == -np.inf] = 0
            valid_audio = data["valid"]["audio"].astype(np.float32)
            valid_audio[valid_audio == -np.inf] = 0
            valid_vision = data["valid"]["vision"].astype(np.float32)
            valid_vision[valid_vision == -np.inf] = 0
            self.dev = []
            for i in range(valid_vision.shape[0]):
                label = np.argmax(valid_labels[i], -1)
                data_tuple = [((valid_text[i], valid_vision[i], valid_audio[i]), label, describe)]
                self.dev.extend(data_tuple)

            test_labels = data["test"]["labels"].astype(np.float32)
            test_text = data["test"]["text"].astype(np.float32)
            test_text[test_text == -np.inf] = 0
            test_audio = data["test"]["audio"].astype(np.float32)
            test_audio[test_audio == -np.inf] = 0
            test_vision = data["test"]["vision"].astype(np.float32)
            test_vision[test_vision == -np.inf] = 0
            self.test = []
            for i in range(test_vision.shape[0]):
                label = np.argmax(test_labels[i], -1)
                data_tuple = [((test_text[i], test_vision[i], test_audio[i]), label, describe)]
                self.test.extend(data_tuple)

            self.word2id = None
            self.pretrained_emb = None

            # 继续写，调整数据顺序，和mosi一样的

        except:
            print("N0 IEMOCAP file")


    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()