from __future__ import division
from __future__ import print_function

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
from sklearn.metrics import precision_recall_fscore_support
import os

import matplotlib.pyplot as plt
from time import gmtime, strftime

# Training settings
dataset = 'hiv'   # 'tox21', 'hiv'
EAGCN_structure = 'concate' #  'concate', 'weighted_ave'
write_file = True
n_den1, n_den2= 64, 32

if dataset == 'tox21':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 256
    weight_decay = 0.0001  # L-2 Norm
    dropout  = 0.3
    random_state = 2
    num_epochs = 80
    learning_rate = 0.0005
if dataset == 'hiv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 64
    weight_decay = 0.00001  # L-2 Norm
    dropout = 0.3
    random_state = 1
    num_epochs = 50
    learning_rate = 0.0005

# Early Stopping:
early_stop_step_single = 3
early_stop_step_multi = 5
early_stop_required_progress = 0.001
early_stop_diff = 0.11

experiment_date = strftime("%b_%d_%H:%M", gmtime()) +'N'
print(experiment_date)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

# targets for  tox21
if dataset == 'tox21':
    all_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
if dataset == 'hiv':
    all_tasks = ['HIV_active']

def weight_func(BCE_weight, labels):
    minibatch_size = labels.shape[0]
    true_labels = labels.sum().data[0]
    false_labels = minibatch_size - true_labels
    if true_labels > 0:
        return from_numpy(minibatch_size - np.array([false_labels, true_labels])).float()
    else:
        return from_numpy(np.array([1, 1])).float()

def test_model(loader, model, tasks):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    model.eval()

    true_labels = []
    pred_labels = []

    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        true_labels.append(label_batch.squeeze(1).long().data.numpy())
        pred_labels.append(F.log_softmax(outputs, dim=1).max(dim=1)[1].data.numpy())

    model.train()

    return precision_recall_fscore_support(np.concatenate(true_labels), np.concatenate(pred_labels), average='binary')

def train(tasks, EAGCN_structure, n_den1, n_den2, file_name):
    x_all, y_all, target, sizes = load_data(dataset)
    max_size = max(sizes)
    x_all, y_all = data_filter(x_all, y_all, target, sizes, tasks)
    x_all, y_all = shuffle(x_all, y_all, random_state=random_state)

    X, x_test, y, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=random_state)
    del x_all, y_all
    test_loader = construct_loader(x_test, y_test, target, batch_size)
    del x_test, y_test

    n_bfeat = X[0][2].shape[0]

    if EAGCN_structure == 'concate':
        model = Concate_GCN(n_bfeat=n_bfeat, n_afeat=25,
                            n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4,
                            n_sgc1_5=n_sgc1_5,
                            n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4,
                            n_sgc2_5=n_sgc2_5,
                            n_den1=n_den1, n_den2=n_den2,
                            nclass=len(tasks)+1, dropout=dropout)
    else:
        model = Weighted_GCN(n_bfeat=n_bfeat, n_afeat=25,
                             n_sgc1_1 = n_sgc1_1, n_sgc1_2 = n_sgc1_2, n_sgc1_3= n_sgc1_3, n_sgc1_4 = n_sgc1_4, n_sgc1_5 = n_sgc1_5,
                             n_sgc2_1 = n_sgc2_1, n_sgc2_2 = n_sgc2_2, n_sgc2_3= n_sgc2_3, n_sgc2_4 = n_sgc2_4, n_sgc2_5 = n_sgc2_5,
                             n_den1=n_den1, n_den2=n_den2, nclass=len(tasks), dropout=dropout)
    if use_cuda:
        # lgr.info("Using the GPU")
        model.cuda()

    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    validation_acc_history = []
    stop_training = False
    BCE_weight = set_weight(y)

    X = np.array(X)
    y = np.array(y)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
    train_loader = construct_loader(x_train, y_train, target, batch_size)
    validation_loader = construct_loader(x_val, y_val, target, batch_size)
    len_train = len(x_train)
    del x_train, y_train, x_val, y_val

    for epoch in range(num_epochs):
        tot_loss = 0
        for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels) in enumerate(train_loader):
            adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
            orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                aromAtt), Variable(
                conjAtt), Variable(ringAtt)
            optimizer.zero_grad()
            outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                            ringAtt_batch)
            weights = weight_func(BCE_weight, label_batch)
            loss = nn.CrossEntropyLoss(weight=weights)(outputs, label_batch.squeeze(1).long())
            tot_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        # report performance
        if True:
            print("Calculating train precision and recall...")
            tpre, trec, tspe, tacc = test_model(train_loader, model, tasks)
            print("Calculating validation precision and recall...")
            vpre, vrec, vspe, vacc = test_model(validation_loader, model, tasks)
            print(
                'Epoch: [{}/{}], '
                'Step: [{}/{}], '
                'Loss: {},'
                '\n'
                'Train: Precision: {}, Recall: {}, fbeta_score: {}, Support: {}'
                '\n'
                'Validation: Precision: {}, Recall: {}, fbeta_score: {}, Support: {}'.format(
                    epoch + 1, num_epochs, i + 1,
                    math.ceil(len_train / batch_size), tot_loss,
                    tpre, trec, tspe, tacc,
                    vpre, vrec, vspe, vacc
                ))

    print("Calculating train precision and recall...")
    tpre, trec, tspe, tacc = test_model(test_loader, model, tasks)
    print(
        'Test Precision: {}, Recall: {}, fbeta_score: {}, Support: {}'.format(tpre, trec, tspe, tacc)
    )

tasks = all_tasks # [task]
print(' learning_rate: {},\n batch_size: {}, \n '
          'tasks: {},\n random_state: {}, \n EAGCN_structure: {}\n'.format(
        learning_rate, batch_size, tasks, random_state, EAGCN_structure))
print('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))
print('n_den1, nden2: {}, {}'.format(n_den1, n_den2))
if use_cuda:
    position = 'server'
else:
    position = 'local'
if len(tasks) == 1:
    directory = '../experiment_result/{}/{}/{}/'.format(position, dataset, tasks)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}'.format(directory, experiment_date)
else:
    directory = "../experiment_result/{}/{}/['all_tasks']/".format(position, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = '{}{}'.format(directory, experiment_date)


if write_file:
    with open(file_name, 'w') as fp:
        fp.write(' learning_rate: {},\n batch_size: {}, \n '
                     'tasks: {},\n random_state: {} \n,'
                     ' EAGCN_structure: {}\n'.format(learning_rate, batch_size,
                                           tasks, random_state, EAGCN_structure))
        fp.write('early_stop_step_single: {}, early_stop_step_multi: {}, \n'
                     'early_stop_required_progress: {},\n early_stop_diff: {}, \n'
                     'weight_decay: {}, dropout: {}\n'.format(early_stop_step_single, early_stop_step_multi,
                                                             early_stop_required_progress, early_stop_diff,
                                                             weight_decay, dropout))
        fp.write('n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5, '
                 'n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5: '
                 '{}, {} {}, {}, {}, / {}, {}, {}, {}, {}'.format(n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                                                                  n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5))

result = train(tasks, EAGCN_structure, n_den1, n_den2,file_name)
