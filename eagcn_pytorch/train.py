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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import cPickle as pickle

import os

import matplotlib.pyplot as plt
from time import gmtime, strftime

# Training settings
dataset = 'tox21'   # 'tox21', 'hiv'
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
if dataset == 'affinity':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 256
    weight_decay = 0.0001  # L-2 Norm
    dropout  = 0.3
    random_state = 2
    num_epochs = 80
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
if dataset == 'affinity':
    all_tasks = 'AADAT,ABAT,ABCB1A,ABCB1B,ABCC8,ABCC9,ABHD6,ACACA,ACE2,ACKR3,ACLY,ACOX1,ACP1,ACR,ADA,ADAM10,ADAMTS4,ADCY1,ADCY5,ADM,ADRA1B,ADRA2B,ADRBK1,AGER,AGPAT2,AGTR1A,AHCY,AKP3,AKR1B10,AKR1C1,AKT2,AKT3,ALB,ALDH2,ALDH3A1,ALKBH3,ALPI,ALPPL2,AMD1,AMPD2,AMPD3,ANPEP,AOC3,APAF1,APEX1,APLNR,APOB,ASAH1,ASIC3,ATG4B,ATR,AURKC,AXL,BACE2,BAD,BAP1,BCL2A1,BCL2A1A,BIRC3,BIRC5,BMP4,BRD2,BRD3,BTK,C1R,C1S,C3AR1,CA13,CA14,CA4,CA5A,CA7,CACNA1C,CACNA1I,CACNA1S,CAMK2D,CAPN2,CAR13,CARM1,CASP2,CASP6,CASP8,CBX7,CCL2,CCL5,CD22,CDC25A,CDC25C,CDK5,CDK8,CDK9,CENPE,CES1,CES2,CHAT,CHKA,CHRNA10,CHRNA3,CHRNA6,CHUK,COMT,CPA1,CPB1,CPB2,CREBBP,CSGALNACT1,CSK,CSNK1A1,CSNK1D,CSNK1G1,CSNK1G2,CSNK2A1,CSNK2A2,CTBP1,CTNNB1,CTRB1,CTRC,CTSA,CTSC,CTSE,CTSF,CTSG,CTSV,CX3CR1,CXCL8,CXCR1,CYP1A1,CYP1B1,CYP24A1,CYP26A1,CYP2A6,CYP2J2,CYP51A1,CYSLTR1,DAGLA,DAO,DAPK3,DCK,DDIT3,DDR1,DDR2,DLG4,DNM1,DNMT1,DOT1L,DPEP1,DPP8,DPP9,DRD5,DUSP3,DUT,DYRK1B,DYRK2,EBP,EEF2K,EGLN2,EGLN3,EIF2AK1,EIF2AK2,EIF2AK3,EIF4A1,EIF4E,ELOVL6,ENPEP,EP300,EPAS1,EPHB3,EPHX1,ERAP1,ERBB4,ERN1,ESRRA,EYA2,EZH2,F11,F12,F13A1,F2RL1,F3,F9,FABP3,FABP4,FAP,FAS,FCER2,FFAR2,FFAR4,FGFR2,FLT4,FPGS,FPR2,FSHR,FUCA1,FYN,G6PD,GABRA1,GABRA5,GALK1,GALR2,GALR3,GAPDH,GART,GBA2,GCKR,GGPS1,GHRHR,GHRL,GLI1,GLO1,GLRA1,GPR142,GPR17,GPR183,GRIA1,GRIA2,GRIA4,GRIK2,GRIN2C,GRIN2D,GRK5,GRM3,GRM7,GRM8,GRPR,GSG2,GSR,GSTM1,GSTP1,GUSB,GYS1,GZMB,HAO2,HCAR3,HCK,HCN1,HDAC2,HDAC3,HDAC5,HKDC1,HLA-A,HLA-DRB1,HMOX1,HMOX2,HNF4A,HPGDS,HPRT1,HRAS,HRH2,HSD11B2,HSD17B7,HSPA5,HTR1F,IARS,ICAM1,IDE,IGFBP3,IKBKE,IL5,IMPDH1,INSR,IRAK4,ITGA2B,ITGAV,ITPR1,JMJD7-PLA2G4B,JUN,KARS,KAT2B,KCNJ1,KCNJ11,KCNJ2,KCNK3,KCNN3,KCNN4,KCNQ1,KDM1A,KDM4C,KHK,KLF5,KLK3,KLK5,KLK7,KLKB1,KMO,KPNA2,L3MBTL3,LAP3,LARGE,LDHA,LDLR,LGALS3,LGMN,LHCGR,LIMK1,LIMK2,LIPG,LNPEP,LPAR1,LPAR2,LPAR3,LYN,MAG,MAP2K5,MAP3K11,MAP3K14,MAP3K5,MAP3K7,MAP3K9,MAP4K2,MAP4K4,MAPK11,MAPK13,MAPK7,MAPK9,MAPKAPK5,MARS,MBNL1,MBTPS1,MC3R,MCHR2,MCOLN3,MEN1,METAP1,MGAM,MGAT2,MGMT,MIF,MITF,MKNK1,MLLT3,MMP11,MMP14,MMP7,MOGAT2,MPI,MPO,MRGPRX1,MTAP,MTTP,MYLK,NAAA,NAT1,NAT2,NCEH1,NCF1,NCOA3,NEK2,NFKB1,NIACR1,NISCH,NLRP3,NMBR,NMT1,NOD1,NOD2,NOS3,NOX1,NOX4,NPC1L1,NPFFR1,NPFFR2,NQO1,NR0B1,NR1D1,NR1I2,NR4A1,NR5A2,NRP1,NT5E,NTRK3,NTSR2,OXER1,OXGR1,P2RX1,P2RX2,P2RX4,P2RY14,P2RY4,P2RY6,P4HB,PABPC1,PAK1,PAK4,PAM,PARP2,PCK1,PCNA,PCSK6,PDE11A,PDE2A,PDE3A,PDE3B,PDE6D,PDE8B,PDE9A,PDF,PDGFRA,PFDN6,PGA5,PGC,PGGT1B,PHOSPHO1,PI4KA,PI4KB,PIM3,PIP4K2A,PKLR,PLA2G10,PLA2G1B,PLAT,PLAUR,PLD1,PLD2,PLEC,PLG,PLIN1,PLK2,PLK3,PLK4,PNMT,POLA1,POLK,PORCN,PPIA,PPOX,PPP1CA,PPP5C,PRKACA,PRKCB,PRKCE,PRKCG,PRKCH,PRKCZ,PRKD1,PRKX,PRMT3,PRNP,PROC,PROKR1,PRSS8,PSEN1,PSMB1,PSMB2,PSMB8,PSMD14,PTBP1,PTGDS,PTGER2,PTGES2,PTGFR,PTGIR,PTH1R,PTK2B,PTK6,PTPN11,PTPN2,PTPN22,PTPRB,PTPRC,PYGM,QRFPR,RAC1,RAD51,RAD54L,RAPGEF4,RARA,RARB,RARG,RASGRP3,RBP4,RCE1,RELA,RET,RGS19,RHOA,RIPK1,RORA,ROS1,RPS6KA5,RPS6KB1,RXRB,RXRG,S100A4,S1PR2,S1PR3,S1PR5,SCN10A,SCN2A,SCN4A,SCN5A,SCNN1A,SELE,SELP,SENP1,SENP6,SENP7,SENP8,SFRP1,SGK1,SHBG,SI,SIRT2,SLC10A1,SLC11A2,SLC12A5,SLC16A1,SLC1A2,SLC1A3,SLC22A12,SLC27A1,SLC27A4,SLC5A1,SLC5A4,SLC6A5,SLCO1B1,SLCO1B3,SMG1,SMN2,SMPD1,SOAT2,SORD,SORT1,SPHK2,SQLE,SSTR2,SSTR4,STAT1,STAT6,STK17A,STK33,SUCNR1,SUMO1,TBK1,TDP2,TGFBR2,THPO,THRA,TK1,TK2,TKT,TLR2,TLR4,TLR8,TLR9,TMPRSS11D,TNF,TNFRSF1A,TNIK,TNK2,TOP2A,TPH1,TPP2,TRHR,TRPC6,TRPV4,TSG101,TTK,TTR,TUBB1,TYK2,TYMP,TYR,TYRO3,UBE2N,UGCG,UGT2B7,UPP1,USP1,UTS2,VCAM1,VIPR1,WDR5,WHSC1,WNT3,WNT3A,XBP1,XDH,YARS,YES1,ZAP70'.split(',')

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
    outputs = []
    labels_arr = []

    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        # outputs = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch)
        # pred_labels.append(F.log_softmax(outputs, dim=1).max(dim=1)[1].cpu().data.numpy())
        outputs.append(model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch).cpu().data.numpy())
        labels_arr.append(label_batch.squeeze(1).long().cpu().data.numpy())

    outputs = np.concatenate(outputs)
    labels_arr = np.concatenate(labels_arr)
    outputs = outputs[labels_arr>-1]
    labels_arr = labels_arr[labels_arr>-1]

    precision, recall, _ = precision_recall_curve(labels_arr, outputs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(labels_arr, outputs)
    aps = average_precision_score(labels_arr, outputs)

    model.train()

    return [roc_auc, pr_auc, aps]
    # return precision_recall_fscore_support(np.concatenate(true_labels), np.concatenate(pred_labels), average='binary')

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
                            nclass=len(tasks), dropout=dropout)
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
            # weights = weight_func(BCE_weight, label_batch)
            # loss = nn.CrossEntropyLoss(weight=weights)(outputs, label_batch.squeeze(1).long())
            # loss = F.binary_cross_entropy_with_logits(outputs.view(-1), label_batch.float().view(-1))\
            #     .mul((1 + label_batch.float().view(-1)).clamp(max=1)).mean()
            loss = F.cross_entropy(outputs, label_batch.float().max(dim=1)[1])
            tot_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()

        # report performance
        # if True:
        #     # print("Calculating train precision and recall...")
        #     # tpre, trec, tspe, tacc = test_model(train_loader, model, tasks)
        #     print("Calculating validation precision and recall...")
        #     vpre, vrec, vspe = test_model(validation_loader, model, tasks)
        #     print(
        #         'Epoch: [{}/{}], '
        #         'Step: [{}/{}], '
        #         'Loss: {},'
        #         '\n'
        #         # 'Train: Precision: {}, Recall: {}, fbeta_score: {}, Support: {}'
        #         # '\n'
        #         'Validation: ROC AUC: {}, P-R AUC: {}, PAS: {}'.format(
        #             epoch + 1, num_epochs, i + 1,
        #             math.ceil(len_train / batch_size), tot_loss,
        #             # tpre, trec, tspe, tacc,
        #             vpre, vrec, vspe
        #         ))

    torch.save(model.state_dict(), '{}.pkl'.format(file_name))
    torch.save(model, '{}.pt'.format(file_name))

    # print("Calculating train precision and recall...")
    # tpre, trec, tspe = test_model(test_loader, model, tasks)
    # print(
    #     'Test ROC AUC: {}, P-R AUC: {}, PAS: {}'.format(tpre, trec, tspe)
    # )
    model_params = {'n_bfeat': n_bfeat, 'n_afeat': 25, 'n_sgc1_1': n_sgc1_1, 'n_sgc1_2': n_sgc1_2, 'n_sgc1_3': n_sgc1_3,
            'n_sgc1_4': n_sgc1_4,
            'n_sgc1_5': n_sgc1_5,
            'n_sgc2_1': n_sgc2_1, 'n_sgc2_2': n_sgc2_2, 'n_sgc2_3': n_sgc2_3, 'n_sgc2_4': n_sgc2_4,
            'n_sgc2_5': n_sgc2_5,
            'n_den1': n_den1, 'n_den2': n_den2,
            'nclass': len(tasks), 'dropout': dropout}
    with open('{}_model_arguments'.format(file_name), 'wb') as fp:
        pickle.dump(model_params, fp)


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
