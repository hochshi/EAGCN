from __future__ import division
from __future__ import print_function
import matplotlib; matplotlib.use('agg')
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

import time
import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from models import *
from torch.utils.data import Dataset

from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import os

import matplotlib.pyplot as plt
from time import gmtime, strftime

from rdkit import DataStructs

random_state = 11
nBits = [64, 128, 256, 512, 1024]
batch_size = 100
num_epochs = 80
weight_decay = 0.0001  # L-2 Norm
learning_rate = 0.0005

def ext_jac(x,y):
    abs_val = np.abs(x-y)
    return 1 - np.divide(x+y-abs_val,x+y+abs_val).sum()

def test_model(dist_mat, train_labels, true_labels):
    # We assume that the dist matrix is 4 sub matrices, train vs. train train vs. something else.
    # So the first train_size X train_size entries are the similarity of the train vs. itself.

    #     sub_mat = dist_mat[train_size:, :train_size]
    sub_mat = dist_mat
    correct = [0] * 4
    #     total = dist_mat.shape[0] - train_size
    total = dist_mat.shape[0]
    top_ks = [1, 5, 10, 30]
    #     top_ks = np.array(top_ks) - 1
    for i, topk in enumerate(top_ks):
        nn = np.partition(sub_mat, topk, axis=1)
        nn = sub_mat <= nn[:, topk-1].reshape(-1, 1)
        nn_labels = np.matmul(nn, train_labels)
        correct[i] = (np.multiply(nn_labels, true_labels) > 0).sum()
    return np.true_divide(correct, total).tolist()


data = np.load('../data/outputs.npz')
train_fps, train_fp_labels = np.concatenate(data['train_fps'], axis=0).reshape(-1, 50), np.concatenate(data['train_fp_labels'], axis=0)
train_labels = np.zeros((len(train_fp_labels), max(train_fp_labels)+1), dtype=np.int8)
train_labels[range(len(train_fp_labels)), train_fp_labels] = 1
val_fps, val_fp_labels = np.concatenate(data['val_fps'], axis=0).reshape(-1, 50), np.concatenate(data['val_fp_labels'], axis=0)
val_labels = np.zeros((len(val_fp_labels), max(train_fp_labels)+1), dtype=np.int8)
val_labels[range(len(val_fp_labels)), val_fp_labels] = 1
test_fps, test_fp_labels = np.concatenate(data['test_fps'], axis=0).reshape(-1, 50), np.concatenate(data['test_fp_labels'], axis=0)
test_labels = np.zeros((len(test_fp_labels), max(train_fp_labels)+1), dtype=np.int8)
test_labels[range(len(test_fp_labels)), test_fp_labels] = 1

train_val = cdist(val_fps, train_fps, metric='euclidean')
print(test_model(train_val, train_labels, val_labels))
del train_val
train_test = cdist(test_fps, train_fps, metric='euclidean')
print(test_model(train_test, train_labels, test_labels))
del train_test


# train_val = cdist(val_fps, train_fps, metric=ext_jac)
# print(test_model(train_val, train_labels, val_labels))
# del train_val
# train_test = cdist(test_fps, train_fps, metric=ext_jac)
# print(test_model(train_test, train_labels, test_labels))
# del train_test
