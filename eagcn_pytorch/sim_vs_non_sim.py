from __future__ import division
from __future__ import print_function
import matplotlib;

matplotlib.use('agg')
import numpy as np
import pandas as pd

import math
from rdkit import DataStructs
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.spatial.distance import pdist, cdist, squareform

from rdkit.Chem import AllChem
from rdkit import DataStructs, Chem

import tqdm

random_state = 11
nBits = 16384


def convert_inchi_fp(mol_list, fp_len=nBits):
    fps, good_mols = list(), list()
    for molt in mol_list:
        mol = Chem.inchi.MolFromInchi(molt)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len))
            good_mols.append(molt)
    np_fps = []
    for fp in fps:
        l_arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, l_arr)
        np_fps.append(l_arr)

    np_fps = np.concatenate(np_fps, axis=0)
    np_fps = np_fps.reshape(-1, fp_len)
    return np_fps, good_mols


def get_similiarity(dist_mat, mols, index):
    similar_indices = set()
    if index > 0:
        rows, cols = np.unravel_index(np.argsort(dist_mat.reshape(-1))[:index], dist_mat.shape)
    else:
        rows, cols = np.unravel_index(np.argsort(dist_mat.reshape(-1))[index:], dist_mat.shape)
    for row, col in zip(rows, cols):
        similar_indices.add(tuple(sorted((row, col))))
    similar_mols = set()
    for row, col in similar_indices:
        similar_mols.add((mols[row], mols[col], dist_mat[row, col]))
    return list(similar_mols)


def similarity_list_to_matrix(similar_mols):
    labels = []
    fps = []
    for i, target in enumerate(targets):
        pair = similar_mols[target][-1]
        label = np.zeros((len(targets),))
        label[i] = 1
        labels.append(label)
        label = np.zeros((len(targets),))
        label[i] = 1
        labels.append(label)
        fps.append(convert_inchi_fp(pair[0:-1])[0])
    labels = np.concatenate(labels, axis=0).reshape(-1, len(targets))
    fps = np.concatenate(fps, axis=0)
    return labels, fps


def test_model(dist_mat, train_labels, true_labels):
    err = [np.inf]
    top_ks = [1]
    for i, topk in enumerate(top_ks):
        nn = np.partition(dist_mat, topk, axis=1)
        nn = dist_mat <= nn[:, topk - 1].reshape(-1, 1)
        nn_labels = np.matmul(nn, train_labels).clip(max=1)

        err[i] = (
            precision_score(true_labels, nn_labels, average='micro'),
            recall_score(true_labels, nn_labels, average='micro'),
            accuracy_score(true_labels, nn_labels),
            average_precision_score(true_labels, nn_labels, average='micro')
        )
    return err


def test_nn(labels, fps):
    for i in range(3):
        train_fps, test_fps, train_labels, test_labels = train_test_split(fps, labels, test_size=0.5,
                                                                          stratify=labels.argmax(axis=1))
        #         print(auc_test_model(cdist(test_fps, train_fps, metric='jaccard'), train_labels, test_labels))
        print(test_model(cdist(test_fps, train_fps, metric='jaccard'), train_labels, test_labels))



datasets = [{
             'file':'../data/small_batch_test.csv',
             'label': u'HIV_active',
             'smiles': u'InChI'
            }]
df = pd.read_csv(datasets[0]['file'])
df.fillna(0, inplace=True)

targets = df.columns[:-1]
similar_mols = dict()
non_similar_mols = dict()
process_bar = tqdm.tqdm(targets)
for target in process_bar:
    fps, mols = convert_inchi_fp(df['InChI'][df[target] == 1].values)
    dist_mat = squareform(pdist(fps, metric='jaccard'))
    non_similar_mols[target] = get_similiarity(dist_mat, df['InChI'][df[target] == 1].values, -1)
    np.fill_diagonal(dist_mat, np.inf)
    similar_mols[target] = get_similiarity(dist_mat, df['InChI'][df[target] == 1].values, 1)

test_nn(*similarity_list_to_matrix(similar_mols))
test_nn(*similarity_list_to_matrix(non_similar_mols))