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
import os

import matplotlib.pyplot as plt
from time import gmtime, strftime

random_state= 13
nbits = 64
batch_size = 100
num_epochs = 80
weight_decay = 0.0001  # L-2 Norm
learning_rate = 0.0005

class ECFPClass(nn.Module):
    def __init__(self, fp_len, nclass):
        super(ECFPClass, self).__init__()

        self.classifier = nn.Linear(fp_len, nclass)

    def forward(self, x):
        return self.classifier(x)


def split_data(data, labels):
    X, x_test, y, y_test = train_test_split(data, labels, test_size=0.1, random_state=random_state, stratify=labels)

    tensor_x_test = torch.from_numpy(x_test).float()
    tensor_y_test = torch.from_numpy(y_test).long()
    test_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state, stratify=y)

    tensor_x_val = torch.from_numpy(x_val).float()
    tensor_y_val = torch.from_numpy(y_val).long()
    val_dataset = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)

    tensor_x_train = torch.from_numpy(x_train).float()
    tensor_y_train = torch.from_numpy(y_train).long()
    train_dataset = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)

    return (train_loader, val_loader, test_loader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

x_all, y_all, target, sizes, mol_to_graph_transform, parameter_holder, edge_vocab, node_vocab = \
    load_data('small_batch_test')

from rdkit import DataStructs
fps = [AllChem.GetMorganFingerprintAsBitVect(MolFromInchi(mol_dat[-1]), 2, nBits=nbits) for mol_dat in x_all]
len_train = len(fps)
labels = y_all
np_fps = []
for fp in fps:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps.append(arr)

np_fps = np.concatenate(np_fps, axis=0)
np_fps = np_fps.reshape(-1, nbits)
np_fps, labels = shuffle(np_fps, labels, random_state=random_state)
# comb_data = np.hstack([np_fps, labels.reshape(-1, 1)])
train_loader, validation_loader, test_loader = split_data(np_fps, labels)
model = ECFPClass(nbits, y_all.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss()
count_parameters(model)


def test_model(loader, model):
    model.eval()
    correct = [0] * 4
    total = Variable(FloatTensor([0]))
    for i, (fps, labels) in enumerate(loader):
        outputs = model(Variable(fps).float())
        smprobs = F.log_softmax(outputs, dim=1)
        labels_pos = torch.sum(smprobs > smprobs[1 == labels].unsqueeze(1).expand(-1, outputs.shape[1]), dim=1)
        top_ks = [1, 5, 10, 30]
        top_ks = np.array(top_ks) - 1
        for i, topk in enumerate(top_ks):
            if (topk + 1) < smprobs.shape[1]:
                correct[i] = correct[i] + sum(labels_pos <= topk).data[0]
            else:
                break

        total += labels.shape[0]
        return np.true_divide(correct, total.data[0]).tolist()

for epoch in range(num_epochs):
    tot_loss = 0
    model.train()
    for i, (fps, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(Variable(fps).float())
        loss = loss_func(outputs, Variable(labels).max(dim=1)[1])
        tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    
    print("Calculating train pos...")
    tpos_0, tpos_5, tpos_10, tpos_30 = test_model(train_loader, model)
    print("Calculating validation pos...")
    vpos_0, vpos_5, vpos_10, vpos_30 = test_model(validation_loader, model)
    print(
        'Epoch: [{}/{}], '
        'Step: [{}/{}], '
        'Loss: {},'
        '\n'
        'Train: 1: {}, 5: {}, 10: {}, 30: {}'
        '\n'
        'Validation: 1: {}, 5: {}, 10: {}, 30: {}'.format(
            epoch + 1, num_epochs, i + 1,
            math.ceil(len_train / batch_size), tot_loss,
            tpos_0, tpos_5, tpos_10, tpos_30,
            vpos_0, vpos_5, vpos_10, vpos_30
        ))
    
tpos_0, tpos_5, tpos_10, tpos_30 = test_model(test_loader, model)
print(
    'Test: 1: {}, 5: {}, 10: {}, 30: {}'.format(
        tpos_0, tpos_5, tpos_10, tpos_30
    ))