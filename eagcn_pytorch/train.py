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

# Training settings
dataset = 'tox21'  # 'tox21', 'hiv', 'pubchem_chembl'
EAGCN_structure = 'concate'  # 'concate', 'weighted_ave'
write_file = True
n_den1, n_den2 = 64, 32

if dataset == 'tox21':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 10, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 256
    weight_decay = 0.0001  # L-2 Norm
    dropout = 0.3
    random_state = 2
    num_epochs = 80
    learning_rate = 0.0005

    def output_transform(x):
        return x

    def loss_func(outputs, labels, weights):
        return F.binary_cross_entropy_with_logits(outputs.view(-1), labels.float().view(-1), weight=weights,
                                                  size_average=False)

    def weight_func(BCE_weight, labels):
        return Variable(weight_tensor(BCE_weight, labels=labels))

if dataset == 'hiv':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 10, 10, 10, 10, 10
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    batch_size = 64
    weight_decay = 0.00001  # L-2 Norm
    dropout = 0.3
    random_state = 1
    num_epochs = 50
    learning_rate = 0.0005

    def output_transform(x):
        return F.log_softmax(x, dim=1)

    def loss_func(outputs, labels, weights):
        return nn.NLLLoss(weight=weights)(outputs, labels.squeeze(1).long())

    def weight_func(BCE_weight, labels):
        normed_BCE_weight = np.array([val[0] for val in BCE_weight.values()], dtype=np.float32)
        normed_BCE_weight = np.true_divide(normed_BCE_weight, sum(normed_BCE_weight))
        return from_numpy(normed_BCE_weight)

if dataset == 'pubchem_chembl':
    n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 20, 20, 20, 20, 20
    n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20
    # batch_size = 16384
    batch_size = 64
    weight_decay = 0.0001  # L-2 Norm
    dropout = 0.3
    random_state = 11
    num_epochs = 100
    learning_rate = 0.0005

    def output_transform(x):
        return F.log_softmax(x, dim=1)

    def loss_func(outputs, labels, weights):
        return nn.NLLLoss()(outputs, labels.max(dim=1)[1])

    def weight_func(BCE_weight, labels):
        normed_BCE_weight = np.array([val[0] for val in BCE_weight.values()], dtype=np.float32)
        normed_BCE_weight = np.true_divide(normed_BCE_weight, sum(normed_BCE_weight))
        return from_numpy(normed_BCE_weight)

# Early Stopping:
early_stop_step_single = 3
early_stop_step_multi = 5
early_stop_required_progress = 0.001
early_stop_diff = 0.11

experiment_date = strftime("%b_%d_%H:%M", gmtime()) + 'N'
print(experiment_date)
torch.manual_seed(random_state)
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
# DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

calcpos = False

# targets for  tox21
if dataset == 'tox21':
    all_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
if dataset == 'hiv':
    all_tasks = ['HIV_active']
    calcpos = True
if dataset == 'pubchem_chembl':
    #all_tasks = 'AADAT,ABAT,ABCB1,ABCB1A,ABCB1B,ABCC1,ABCC8,ABCC9,ABCG2,ABHD6,ABL1,ACACA,ACACB,ACE,ACE2,ACHE,ACKR3,ACLY,ACOX1,ACP1,ACR,ADA,ADAM10,ADAM17,ADAMTS4,ADAMTS5,ADCY1,ADCY5,ADK,ADM,ADORA1,ADORA2A,ADORA2B,ADORA3,ADRA1A,ADRA1B,ADRA1D,ADRA2A,ADRA2B,ADRA2C,ADRB1,ADRB2,ADRB3,ADRBK1,AGER,AGPAT2,AGTR1,AGTR1A,AGTR1B,AGTR2,AHCY,AHR,AKP3,AKR1B1,AKR1B10,AKR1C1,AKR1C3,AKT1,AKT2,AKT3,ALB,ALDH1A1,ALDH2,ALDH3A1,ALK,ALKBH3,ALOX12,ALOX15,ALOX15B,ALOX5,ALOX5AP,ALPI,ALPL,ALPPL2,AMD1,AMPD2,AMPD3,ANPEP,AOC3,APAF1,APEX1,APLNR,APOB,APOBEC3A,APOBEC3F,APOBEC3G,APP,AR,ASAH1,ASIC3,ATAD5,ATG4B,ATM,ATR,ATXN2,AURKA,AURKB,AURKC,AVPR1A,AVPR1B,AVPR2,AXL,BACE1,BACE2,BAD,BAP1,BAZ2B,BCHE,BCL2,BCL2A1,BCL2A1A,BCL2L1,BDKRB1,BDKRB2,BIRC2,BIRC3,BIRC5,BLM,BMP1,BMP4,BRAF,BRCA1,BRD2,BRD3,BRD4,BRS3,BTK,C1R,C1S,C3AR1,C5AR1,CA1,CA12,CA13,CA14,CA2,CA4,CA5A,CA7,CA9,CACNA1B,CACNA1C,CACNA1D,CACNA1G,CACNA1H,CACNA1I,CACNA1S,CACNA2D1,CALCRL,CAMK2D,CAPN1,CAPN2,CAR13,CARM1,CASP1,CASP2,CASP3,CASP6,CASP7,CASP8,CASR,CBX1,CBX7,CCKAR,CCKBR,CCL2,CCL5,CCR1,CCR2,CCR3,CCR4,CCR5,CCR8,CD22,CDC25A,CDC25B,CDC25C,CDC7,CDK1,CDK2,CDK4,CDK5,CDK8,CDK9,CENPE,CES1,CES2,CETP,CFTR,CGA,CHAT,CHEK1,CHEK2,CHKA,CHRM1,CHRM2,CHRM3,CHRM4,CHRM5,CHRNA10,CHRNA3,CHRNA4,CHRNA6,CHRNA7,CHUK,CLK1,CLK4,CMA1,CNR1,CNR2,COMT,CPA1,CPB1,CPB2,CREBBP,CRHR1,CSF1R,CSGALNACT1,CSK,CSNK1A1,CSNK1D,CSNK1G1,CSNK1G2,CSNK2A1,CSNK2A2,CTBP1,CTDSP1,CTNNB1,CTRB1,CTRC,CTSA,CTSB,CTSC,CTSD,CTSE,CTSF,CTSG,CTSK,CTSL,CTSS,CTSV,CX3CR1,CXCL8,CXCR1,CXCR2,CXCR3,CXCR4,CYP11B1,CYP11B2,CYP17A1,CYP19A1,CYP1A1,CYP1A2,CYP1B1,CYP24A1,CYP26A1,CYP2A6,CYP2C19,CYP2C9,CYP2D6,CYP2J2,CYP3A4,CYP51A1,CYSLTR1,DAGLA,DAO,DAPK3,DCK,DDIT3,DDR1,DDR2,DGAT1,DHFR,DHODH,DLG4,DNM1,DNMT1,DOT1L,DPEP1,DPP4,DPP7,DPP8,DPP9,DRD1,DRD2,DRD3,DRD4,DRD5,DUSP3,DUT,DYRK1A,DYRK1B,DYRK2,EBP,ECE1,EDNRA,EDNRB,EEF2K,EGFR,EGLN1,EGLN2,EGLN3,EHMT2,EIF2AK1,EIF2AK2,EIF2AK3,EIF4A1,EIF4E,ELANE,ELOVL6,ENPEP,ENPP2,EP300,EPAS1,EPHB3,EPHB4,EPHX1,EPHX2,ERAP1,ERBB2,ERBB4,ERCC5,ERG,ERN1,ESR1,ESR2,ESRRA,EYA2,EZH2,F10,F11,F12,F13A1,F2,F2R,F2RL1,F3,F7,F9,FAAH,FABP3,FABP4,FAP,FAS,FASN,FBP1,FCER2,FDFT1,FDPS,FEN1,FFAR1,FFAR2,FFAR4,FGFR1,FGFR2,FKBP1A,FLT1,FLT3,FLT4,FOLH1,FPGS,FPR1,FPR2,FSHR,FUCA1,FURIN,FYN,G6PD,GAA,GABRA1,GABRA5,GALK1,GALR2,GALR3,GAPDH,GART,GBA,GBA2,GCGR,GCK,GCKR,GFER,GGPS1,GHRHR,GHRL,GHSR,GLA,GLI1,GLO1,GLP1R,GLRA1,GLS,GMNN,GNAS,GNRHR,GPBAR1,GPR119,GPR142,GPR17,GPR183,GPR35,GPR55,GRB2,GRIA1,GRIA2,GRIA4,GRIK1,GRIK2,GRIN1,GRIN2B,GRIN2C,GRIN2D,GRK5,GRM1,GRM2,GRM3,GRM4,GRM5,GRM7,GRM8,GRPR,GSG2,GSK3A,GSK3B,GSR,GSTM1,GSTP1,GUSB,GYS1,GZMB,HAO2,HCAR2,HCAR3,HCK,HCN1,HCRTR1,HCRTR2,HDAC1,HDAC2,HDAC3,HDAC4,HDAC5,HDAC6,HDAC8,HIF1A,HKDC1,HLA-A,HLA-DRB1,HMGCR,HMOX1,HMOX2,HNF4A,HPGD,HPGDS,HPRT1,HPSE,HRAS,HRH1,HRH2,HRH3,HRH4,HSD11B1,HSD11B2,HSD17B1,HSD17B10,HSD17B2,HSD17B3,HSD17B7,HSF1,HSP90AA1,HSP90AB1,HSPA5,HTR1A,HTR1B,HTR1D,HTR1F,HTR2A,HTR2B,HTR2C,HTR3A,HTR4,HTR5A,HTR6,HTR7,HTT,IARS,ICAM1,ICMT,IDE,IDH1,IDO1,IGF1R,IGFBP3,IKBKB,IKBKE,IL5,IMPA1,IMPDH1,IMPDH2,INSR,IRAK4,ITGA2B,ITGA4,ITGAL,ITGAV,ITK,ITPR1,JAK1,JAK2,JAK3,JMJD7-PLA2G4B,JUN,KARS,KAT2A,KAT2B,KCNA3,KCNA5,KCNH2,KCNJ1,KCNJ11,KCNJ2,KCNK3,KCNMA1,KCNN3,KCNN4,KCNQ1,KCNQ2,KDM1A,KDM4A,KDM4C,KDM4E,KDR,KHK,KIF11,KISS1R,KIT,KLF5,KLK3,KLK5,KLK7,KLKB1,KMO,KMT2A,KPNA2,L3MBTL1,L3MBTL3,LAP3,LARGE,LCK,LDHA,LDLR,LGALS3,LGMN,LHCGR,LIMK1,LIMK2,LIPE,LIPG,LMNA,LNPEP,LPAR1,LPAR2,LPAR3,LRRK2,LSS,LTA4H,LTB4R,LYN,MAG,MAOA,MAOB,MAP2K1,MAP2K5,MAP3K11,MAP3K14,MAP3K5,MAP3K7,MAP3K8,MAP3K9,MAP4K2,MAP4K4,MAPK1,MAPK10,MAPK11,MAPK13,MAPK14,MAPK7,MAPK8,MAPK9,MAPKAPK2,MAPKAPK5,MAPT,MARS,MBNL1,MBTPS1,MC1R,MC3R,MC4R,MC5R,MCHR1,MCHR2,MCL1,MCOLN3,MDM2,MEN1,MERTK,MET,METAP1,METAP2,MGAM,MGAT2,MGLL,MGMT,MIF,MITF,MKNK1,MLLT3,MLNR,MLYCD,MME,MMP1,MMP11,MMP12,MMP13,MMP14,MMP2,MMP3,MMP7,MMP8,MMP9,MOGAT2,MPHOSPH8,MPI,MPL,MPO,MRGPRX1,MTAP,MTNR1A,MTNR1B,MTOR,MTTP,MYLK,NAAA,NAMPT,NAT1,NAT2,NCEH1,NCF1,NCOA3,NEK2,NFE2L2,NFKB1,NIACR1,NISCH,NLRP3,NMBR,NMT1,NOD1,NOD2,NOS1,NOS2,NOS3,NOX1,NOX4,NPBWR1,NPC1,NPC1L1,NPFFR1,NPFFR2,NPSR1,NPY1R,NPY2R,NPY5R,NQO1,NQO2,NR0B1,NR1D1,NR1H2,NR1H3,NR1H4,NR1I2,NR2E3,NR3C1,NR3C2,NR4A1,NR5A1,NR5A2,NRP1,NT5E,NTRK1,NTRK3,NTSR1,NTSR2,OPRD1,OPRK1,OPRL1,OPRM1,OXER1,OXGR1,OXTR,P2RX1,P2RX2,P2RX3,P2RX4,P2RX7,P2RY1,P2RY12,P2RY14,P2RY2,P2RY4,P2RY6,P4HB,PABPC1,PAK1,PAK4,PAM,PARP1,PARP2,PAX8,PCK1,PCNA,PCSK6,PDE10A,PDE11A,PDE1C,PDE2A,PDE3A,PDE3B,PDE4A,PDE4B,PDE4D,PDE5A,PDE6D,PDE7A,PDE8B,PDE9A,PDF,PDGFRA,PDGFRB,PDPK1,PFDN6,PGA5,PGC,PGGT1B,PGR,PHOSPHO1,PI4KA,PI4KB,PIK3CA,PIK3CB,PIK3CD,PIK3CG,PIM1,PIM2,PIM3,PIN1,PIP4K2A,PKLR,PKM,PLA2G10,PLA2G1B,PLA2G2A,PLA2G4A,PLA2G7,PLAT,PLAU,PLAUR,PLD1,PLD2,PLEC,PLG,PLIN1,PLK1,PLK2,PLK3,PLK4,PMP22,PNMT,PNP,POLA1,POLB,POLH,POLI,POLK,PORCN,PPARA,PPARD,PPARG,PPIA,PPOX,PPP1CA,PPP5C,PRCP,PREP,PRF1,PRKACA,PRKCA,PRKCB,PRKCD,PRKCE,PRKCG,PRKCH,PRKCQ,PRKCZ,PRKD1,PRKDC,PRKX,PRMT3,PRNP,PROC,PROKR1,PRSS1,PRSS8,PSEN1,PSMB1,PSMB2,PSMB5,PSMB8,PSMD14,PTAFR,PTBP1,PTGDR,PTGDR2,PTGDS,PTGER1,PTGER2,PTGER3,PTGER4,PTGES,PTGES2,PTGFR,PTGIR,PTGS1,PTGS2,PTH1R,PTK2,PTK2B,PTK6,PTPN1,PTPN11,PTPN2,PTPN22,PTPN7,PTPRB,PTPRC,PYGL,PYGM,QPCT,QRFPR,RAB9A,RAC1,RAD51,RAD52,RAD54L,RAF1,RAPGEF4,RARA,RARB,RARG,RASGRP3,RBP4,RCE1,RECQL,RELA,REN,RET,RGS19,RGS4,RHOA,RIPK1,ROCK1,ROCK2,RORA,RORC,ROS1,RPS6KA3,RPS6KA5,RPS6KB1,RXFP1,RXRA,RXRB,RXRG,S100A4,S1PR1,S1PR2,S1PR3,S1PR4,S1PR5,SCARB1,SCD,SCD1,SCN10A,SCN2A,SCN4A,SCN5A,SCN9A,SCNN1A,SELE,SELP,SENP1,SENP6,SENP7,SENP8,SERPINE1,SFRP1,SGK1,SHBG,SHH,SI,SIGMAR1,SIRT1,SIRT2,SLC10A1,SLC10A2,SLC11A2,SLC12A5,SLC16A1,SLC18A2,SLC18A3,SLC1A2,SLC1A3,SLC22A12,SLC27A1,SLC27A4,SLC29A1,SLC5A1,SLC5A2,SLC5A4,SLC5A7,SLC6A1,SLC6A2,SLC6A3,SLC6A4,SLC6A5,SLC6A9,SLC9A1,SLCO1B1,SLCO1B3,SMAD3,SMG1,SMN1,SMN2,SMO,SMPD1,SNCA,SOAT1,SOAT2,SORD,SORT1,SPHK1,SPHK2,SQLE,SRC,SRD5A1,SRD5A2,SSTR1,SSTR2,SSTR3,SSTR4,SSTR5,ST14,STAT1,STAT3,STAT6,STK17A,STK33,STS,SUCNR1,SUMO1,SYK,TAAR1,TACR1,TACR2,TACR3,TARDBP,TBK1,TBXA2R,TBXAS1,TDP1,TDP2,TEK,TERT,TGFBR1,TGFBR2,TGM2,THPO,THRA,THRB,TK1,TK2,TKT,TLR2,TLR4,TLR7,TLR8,TLR9,TMPRSS11D,TNF,TNFRSF1A,TNIK,TNK2,TNKS,TNKS2,TOP1,TOP2A,TP53,TPH1,TPP2,TPSAB1,TRHR,TRPA1,TRPC4,TRPC6,TRPM8,TRPV1,TRPV4,TSG101,TSHR,TSPO,TTK,TTR,TUBB1,TXNRD1,TYK2,TYMP,TYMS,TYR,TYRO3,UBE2N,UGCG,UGT2B7,UPP1,USP1,USP2,UTS2,UTS2R,VCAM1,VCP,VDR,VIPR1,WDR5,WEE1,WHSC1,WNT3,WNT3A,WRN,XBP1,XDH,XIAP,YARS,YES1,ZAP70'.split(',')
    all_tasks = 'AADAT,ABAT,ABCB1A,ABCB1B,ABCC8,ABCC9,ABHD6,ACACA,ACE2,ACKR3,ACLY,ACOX1,ACP1,ACR,ADA,ADAM10,ADAMTS4,ADCY1,ADCY5,ADM,ADRA1B,ADRA2B,ADRBK1,AGER,AGPAT2,AGTR1A,AHCY,AKP3,AKR1B10,AKR1C1,AKT2,AKT3,ALB,ALDH2,ALDH3A1,ALKBH3,ALPI,ALPPL2,AMD1,AMPD2,AMPD3,ANPEP,AOC3,APAF1,APEX1,APLNR,APOB,ASAH1,ASIC3,ATG4B,ATR,AURKC,AXL,BACE2,BAD,BAP1,BCL2A1,BCL2A1A,BIRC3,BIRC5,BMP4,BRD2,BRD3,BTK,C1R,C1S,C3AR1,CA13,CA14,CA4,CA5A,CA7,CACNA1C,CACNA1I,CACNA1S,CAMK2D,CAPN2,CAR13,CARM1,CASP2,CASP6,CASP8,CBX7,CCL2,CCL5,CD22,CDC25A,CDC25C,CDK5,CDK8,CDK9,CENPE,CES1,CES2,CHAT,CHKA,CHRNA10,CHRNA3,CHRNA6,CHUK,COMT,CPA1,CPB1,CPB2,CREBBP,CSGALNACT1,CSK,CSNK1A1,CSNK1D,CSNK1G1,CSNK1G2,CSNK2A1,CSNK2A2,CTBP1,CTNNB1,CTRB1,CTRC,CTSA,CTSC,CTSE,CTSF,CTSG,CTSV,CX3CR1,CXCL8,CXCR1,CYP1A1,CYP1B1,CYP24A1,CYP26A1,CYP2A6,CYP2J2,CYP51A1,CYSLTR1,DAGLA,DAO,DAPK3,DCK,DDIT3,DDR1,DDR2,DLG4,DNM1,DNMT1,DOT1L,DPEP1,DPP8,DPP9,DRD5,DUSP3,DUT,DYRK1B,DYRK2,EBP,EEF2K,EGLN2,EGLN3,EIF2AK1,EIF2AK2,EIF2AK3,EIF4A1,EIF4E,ELOVL6,ENPEP,EP300,EPAS1,EPHB3,EPHX1,ERAP1,ERBB4,ERN1,ESRRA,EYA2,EZH2,F11,F12,F13A1,F2RL1,F3,F9,FABP3,FABP4,FAP,FAS,FCER2,FFAR2,FFAR4,FGFR2,FLT4,FPGS,FPR2,FSHR,FUCA1,FYN,G6PD,GABRA1,GABRA5,GALK1,GALR2,GALR3,GAPDH,GART,GBA2,GCKR,GGPS1,GHRHR,GHRL,GLI1,GLO1,GLRA1,GPR142,GPR17,GPR183,GRIA1,GRIA2,GRIA4,GRIK2,GRIN2C,GRIN2D,GRK5,GRM3,GRM7,GRM8,GRPR,GSG2,GSR,GSTM1,GSTP1,GUSB,GYS1,GZMB,HAO2,HCAR3,HCK,HCN1,HDAC2,HDAC3,HDAC5,HKDC1,HLA-A,HLA-DRB1,HMOX1,HMOX2,HNF4A,HPGDS,HPRT1,HRAS,HRH2,HSD11B2,HSD17B7,HSPA5,HTR1F,IARS,ICAM1,IDE,IGFBP3,IKBKE,IL5,IMPDH1,INSR,IRAK4,ITGA2B,ITGAV,ITPR1,JMJD7-PLA2G4B,JUN,KARS,KAT2B,KCNJ1,KCNJ11,KCNJ2,KCNK3,KCNN3,KCNN4,KCNQ1,KDM1A,KDM4C,KHK,KLF5,KLK3,KLK5,KLK7,KLKB1,KMO,KPNA2,L3MBTL3,LAP3,LARGE,LDHA,LDLR,LGALS3,LGMN,LHCGR,LIMK1,LIMK2,LIPG,LNPEP,LPAR1,LPAR2,LPAR3,LYN,MAG,MAP2K5,MAP3K11,MAP3K14,MAP3K5,MAP3K7,MAP3K9,MAP4K2,MAP4K4,MAPK11,MAPK13,MAPK7,MAPK9,MAPKAPK5,MARS,MBNL1,MBTPS1,MC3R,MCHR2,MCOLN3,MEN1,METAP1,MGAM,MGAT2,MGMT,MIF,MITF,MKNK1,MLLT3,MMP11,MMP14,MMP7,MOGAT2,MPI,MPO,MRGPRX1,MTAP,MTTP,MYLK,NAAA,NAT1,NAT2,NCEH1,NCF1,NCOA3,NEK2,NFKB1,NIACR1,NISCH,NLRP3,NMBR,NMT1,NOD1,NOD2,NOS3,NOX1,NOX4,NPC1L1,NPFFR1,NPFFR2,NQO1,NR0B1,NR1D1,NR1I2,NR4A1,NR5A2,NRP1,NT5E,NTRK3,NTSR2,OXER1,OXGR1,P2RX1,P2RX2,P2RX4,P2RY14,P2RY4,P2RY6,P4HB,PABPC1,PAK1,PAK4,PAM,PARP2,PCK1,PCNA,PCSK6,PDE11A,PDE2A,PDE3A,PDE3B,PDE6D,PDE8B,PDE9A,PDF,PDGFRA,PFDN6,PGA5,PGC,PGGT1B,PHOSPHO1,PI4KA,PI4KB,PIM3,PIP4K2A,PKLR,PLA2G10,PLA2G1B,PLAT,PLAUR,PLD1,PLD2,PLEC,PLG,PLIN1,PLK2,PLK3,PLK4,PNMT,POLA1,POLK,PORCN,PPIA,PPOX,PPP1CA,PPP5C,PRKACA,PRKCB,PRKCE,PRKCG,PRKCH,PRKCZ,PRKD1,PRKX,PRMT3,PRNP,PROC,PROKR1,PRSS8,PSEN1,PSMB1,PSMB2,PSMB8,PSMD14,PTBP1,PTGDS,PTGER2,PTGES2,PTGFR,PTGIR,PTH1R,PTK2B,PTK6,PTPN11,PTPN2,PTPN22,PTPRB,PTPRC,PYGM,QRFPR,RAC1,RAD51,RAD54L,RAPGEF4,RARA,RARB,RARG,RASGRP3,RBP4,RCE1,RELA,RET,RGS19,RHOA,RIPK1,RORA,ROS1,RPS6KA5,RPS6KB1,RXRB,RXRG,S100A4,S1PR2,S1PR3,S1PR5,SCN10A,SCN2A,SCN4A,SCN5A,SCNN1A,SELE,SELP,SENP1,SENP6,SENP7,SENP8,SFRP1,SGK1,SHBG,SI,SIRT2,SLC10A1,SLC11A2,SLC12A5,SLC16A1,SLC1A2,SLC1A3,SLC22A12,SLC27A1,SLC27A4,SLC5A1,SLC5A4,SLC6A5,SLCO1B1,SLCO1B3,SMG1,SMN2,SMPD1,SOAT2,SORD,SORT1,SPHK2,SQLE,SSTR2,SSTR4,STAT1,STAT6,STK17A,STK33,SUCNR1,SUMO1,TBK1,TDP2,TGFBR2,THPO,THRA,TK1,TK2,TKT,TLR2,TLR4,TLR8,TLR9,TMPRSS11D,TNF,TNFRSF1A,TNIK,TNK2,TOP2A,TPH1,TPP2,TRHR,TRPC6,TRPV4,TSG101,TTK,TTR,TUBB1,TYK2,TYMP,TYR,TYRO3,UBE2N,UGCG,UGT2B7,UPP1,USP1,UTS2,VCAM1,VIPR1,WDR5,WHSC1,WNT3,WNT3A,XBP1,XDH,YARS,YES1,ZAP70'.split(',')
    calcpos = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def embed_afm(afm, node_to_ix, node_word_len):
    nodes = afm[:, 0:node_word_len].astype(np.int8).tolist()
    nodes = ["".join([str(i) for i in word]) for word in nodes]
    nodes = np.array([[node_to_ix[key]] for key in nodes])
    remaining_node_attributes = afm[:, node_word_len:]
    return np.concatenate((nodes, remaining_node_attributes), axis=1)

def embed_bonds(adj, orderAtt, aromAtt, conjAtt, ringAtt, edge_to_ix, edge_word_len):
    nz = (0 != adj.reshape(-1))
    edge_data = np.concatenate((orderAtt, aromAtt, conjAtt, ringAtt), axis=0).transpose(1, 2, 0)
    edges = edge_data.astype(np.int8).reshape(-1, edge_word_len)[nz].tolist()
    edges = ["".join([str(i) for i in word]) for word in edges]
    edges = np.array([edge_to_ix[key] for key in edges])
    new_edges = np.zeros(nz.shape, dtype=np.float32)
    new_edges[nz] = edges
    return new_edges.reshape(adj.shape)


def embed_data(x_all, edge_vocab, node_vocab):
    edge_to_ix = {edge: i for i, edge in enumerate(edge_vocab)}
    edge_word_len = len(list(edge_to_ix.keys())[0])

    node_to_ix = {node: i for i, node in enumerate(node_vocab)}
    node_word_len = len(list(node_to_ix.keys())[0])


    afm_pos, adj_pos, bfm_pos, orderAtts_pos, aromAtts_pos, conjAtts_pos, ringAtts_pos = 0, 1, 2, 3, 4, 5, 6

    for x in x_all:
        new_bfm = embed_bonds(x[adj_pos], x[orderAtts_pos], x[aromAtts_pos], x[conjAtts_pos], x[ringAtts_pos],
                              edge_to_ix, edge_word_len)
        new_afm = embed_afm(x[afm_pos], node_to_ix, node_word_len)
        x[afm_pos] = new_afm
        x[bfm_pos] = new_bfm

    return (x_all, edge_to_ix, edge_word_len, node_to_ix, node_word_len)


def embed_edges(adjs, bfts):
    nz = adjs.view(-1).byte()
    return torch.masked_select(bfts.view(-1), nz).long()


def embed_nodes(adjs, afms):
    nz, _ = adjs.max(dim=2)
    nz = nz.view(-1).byte()
    return (
        afms.view(-1, afms.shape[-1])[nz.unsqueeze(1).expand(-1, afms.shape[-1])].view(-1, afms.shape[-1])[:,0].long(),
        afms.view(-1, afms.shape[-1])[nz.unsqueeze(1).expand(-1, afms.shape[-1])].view(-1, afms.shape[-1])[:, 1:],
    )


def split_data(x_all, y_all, target, mol_to_graph_transform, random_state=random_state):
    try:
        X, x_test, y, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=random_state, stratify=y_all)
    except:
        X, x_test, y, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=random_state)

    if mol_to_graph_transform is None:
        test_loader = construct_loader(x_test, y_test, target, batch_size)
    else:
        test_loader = construct_pubchem_loader(x_test, y_test, batch_size, mol_to_graph_transform)
    del x_test, y_test
    BCE_weight = set_weight(y)
    try:
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state, stratify=y)
    except:
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
    del X, y
    if mol_to_graph_transform is None:
        train_loader = construct_loader(x_train, y_train, target, batch_size)
        validation_loader = construct_loader(x_val, y_val, target, batch_size)
    else:
        train_loader = construct_pubchem_loader(x_train, y_train, batch_size, mol_to_graph_transform)
        validation_loader = construct_pubchem_loader(x_val, y_val, batch_size, mol_to_graph_transform)
    len_train = len(x_train)
    del x_train, y_train, x_val, y_val
    return (train_loader, validation_loader, test_loader, BCE_weight, len_train)


def test_model(loader, model, tasks, calcpos=False):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    true_value = []
    all_out = []
    model.eval()
    out_value_dic = {}
    true_value_dic = {}
    correct = [0]*4
    total = 0
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels in loader:
        afm, axfm = embed_nodes(adj, afm)
        btf = embed_edges(adj, btf)
        outputs = model(Variable(adj), Variable(afm), Variable(axfm), Variable(btf))
        if calcpos:
            smprobs = output_transform(outputs)
            labels_pos = torch.sum(smprobs > smprobs[1 == labels].unsqueeze(1).expand(-1, outputs.shape[1]), dim=1)
            top_ks = [1, 5, 10, 30]
            top_ks = np.array(top_ks) - 1
            for i, topk in enumerate(top_ks):
                if (topk+1) < smprobs.shape[1]:
                    correct[i] = correct[i] + sum(labels_pos <= topk).data[0]
                else:
                    break

            total += labels.shape[0]
        else:
            probs = F.sigmoid(outputs)
            if use_cuda:
                out_list = probs.cpu().data.view(-1).numpy().tolist()
                all_out.extend(out_list)
                label_list = labels.tolist()
                true_value.extend([item for sublist in label_list for item in sublist])
                out_sep_list = probs.cpu().data.view(-1, len(tasks)).numpy().tolist()
            else:
                label_list = labels.tolist()
                out_sep_list = probs.data.view(-1, len(tasks)).numpy().tolist()

            for i in range(0, len(out_sep_list)):
                for j in list(range(0, len(tasks))):
                    if label_list[i][j] == -1:
                        continue
                    if j not in true_value_dic.keys():
                        out_value_dic[j] = [out_sep_list[i][j]]
                        true_value_dic[j] = [int(label_list[i][j])]
                    else:
                        out_value_dic[j].extend([out_sep_list[i][j]])
                        true_value_dic[j].extend([int(label_list[i][j])])
    model.train()

    if calcpos:
        return tuple(np.true_divide(correct, total).tolist())
    else:
        aucs = []
        for key in list(range(0, len(tasks))):
            fpr, tpr, threshold = metrics.roc_curve(true_value_dic[key], out_value_dic[key], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            if math.isnan(auc):
                print('the {}th label has no postive samples, max value {}'.format(key, max(true_value_dic[key])))
            aucs.append(auc)
        return (aucs, sum(aucs) / len(aucs))


def train(tasks, EAGCN_structure, n_den1, n_den2, file_name):
    x_all, y_all, target, sizes, mol_to_graph_transform, parameter_holder, edge_vocab, node_vocab = load_data(dataset)
    max_size = max(sizes)
    x_all, y_all = data_filter(x_all, y_all, target, sizes, tasks)
    x_all, y_all = shuffle(x_all, y_all, random_state=random_state)

    if mol_to_graph_transform is None:
        n_afeat = x_all[0][0].shape[1]
        n_bfeat = 0
    else:
        n_afeat = mol_to_graph_transform.afnorm.feature_num
        n_bfeat = 0 # parameter_holder.n_bfeat


    x_all, edge_to_ix, edge_word_len, node_to_ix, node_word_len = embed_data(x_all, edge_vocab, node_vocab)

    # if 'hiv' == dataset:
    #     model = Shi_GCN(n_bfeat=n_bfeat, n_afeat=n_afeat,
    #                     n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4,
    #                     n_sgc1_5=n_sgc1_5,
    #                     n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4,
    #                     n_sgc2_5=n_sgc2_5,
    #                     n_den1=n_den1, n_den2=n_den2,
    #                     nclass=len(tasks)+1, dropout=dropout,
    #                     edge_to_ix=edge_to_ix, edge_word_len=edge_word_len, node_to_ix=node_to_ix,
    #                     node_word_len=node_word_len)
    # else:
    #     model = Shi_GCN(n_bfeat=n_bfeat, n_afeat=n_afeat,
    #                     n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4,
    #                     n_sgc1_5=n_sgc1_5,
    #                     n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4,
    #                     n_sgc2_5=n_sgc2_5,
    #                     n_den1=n_den1, n_den2=n_den2,
    #                     nclass=len(tasks), dropout=dropout,
    #                     edge_to_ix=edge_to_ix, edge_word_len=edge_word_len, node_to_ix=node_to_ix,
    #                     node_word_len=node_word_len, use_att=False)

    # model = MolGraph(n_afeat, edge_to_ix, edge_word_len, node_to_ix, node_word_len)
    model = MolGCN(n_afeat, edge_to_ix, edge_word_len, node_to_ix, node_word_len, len(tasks))


    print("model has {} parameters".format(count_parameters(model)))
    if use_cuda:
        # lgr.info("Using the GPU")
        model.cuda()

    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    validation_acc_history = []
    acc_history = np.empty([4, 2, num_epochs])
    stop_training = False
    train_loader, validation_loader, test_loader, BCE_weight, len_train = split_data(x_all, y_all, target,
                                                                                     mol_to_graph_transform,
                                                                                     random_state=random_state)
    del x_all, y_all, target

    for epoch in range(num_epochs):
        print("Epoch: [{}/{}]".format(epoch + 1, num_epochs))
        tot_loss = 0
        for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            afm, axfm = embed_nodes(adj, afm)
            btf = embed_edges(adj, btf)
            outputs = model(Variable(adj), Variable(afm), Variable(axfm), Variable(btf))
            labels = Variable(labels.float())
            non_nan_num = ((labels == 1).sum() + (labels == 0).sum()).float()
            weights = weight_func(BCE_weight, labels)
            loss = loss_func(output_transform(outputs),labels, weights)
            tot_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        # report performance
        if True:
            if calcpos:
                print("Calculating train pos...")
                tpos_0, tpos_5, tpos_10, tpos_30 = test_model(train_loader, model, tasks, calcpos=True)
                acc_history[:, 0, epoch] = [tpos_0, tpos_5, tpos_10, tpos_30]
                print("Calculating validation pos...")
                vpos_0, vpos_5, vpos_10, vpos_30 = test_model(validation_loader, model, tasks, calcpos=True)
                acc_history[:, 1, epoch] = [vpos_0, vpos_5, vpos_10, vpos_30]
                print(
                    'Epoch: [{}/{}], '
                    'Step: [{}/{}], '
                    'Loss: {},'
                    '\n'
                    'Train: 0: {}, 5: {}, 10: {}, 30: {}'
                    '\n'
                    'Validation: 0: {}, 5: {}, 10: {}, 30: {}'.format(
                        epoch + 1, num_epochs, i + 1,
                        math.ceil(len_train / batch_size), tot_loss,
                        tpos_0, tpos_5, tpos_10, tpos_30,
                        vpos_0, vpos_5, vpos_10, vpos_30
                    ))
            else:
                print("Calculating train auc...")
                train_acc_sep, train_acc_tot = test_model(train_loader, model, tasks)
                print("Calculating validation auc...")
                val_acc_sep, val_acc_tot = test_model(validation_loader, model, tasks)
                print(
                    'Epoch: [{}/{}], '
                    'Step: [{}/{}], '
                    'Loss: {}, \n'
                    'Train AUC seperate: {}, \n'
                    'Train AUC total: {}, \n'
                    'Validation AUC seperate: {}, \n'
                    'Validation AUC total: {} \n'.format(
                        epoch + 1, num_epochs, i + 1,
                        math.ceil(len_train / batch_size), tot_loss, \
                        train_acc_sep, train_acc_tot, val_acc_sep,
                        val_acc_tot))
                if write_file:
                    with open(file_name, 'a') as fp:
                        fp.write(
                            'Epoch: [{}/{}], '
                            'Step: [{}/{}], '
                            'Loss: {}, \n'
                            'Train AUC seperate: {}, \n'
                            'Train AUC total: {}, \n'
                            'Validation AUC seperate: {}, \n'
                            'Validation AUC total: {} \n'.format(
                                epoch + 1, num_epochs, i + 1,
                                math.ceil(len_train / batch_size),
                                tot_loss, \
                                train_acc_sep, train_acc_tot, val_acc_sep,
                                val_acc_tot))
                validation_acc_history.append(val_acc_tot)
                # check if we need to earily stop the model
                stop_training = earily_stop(validation_acc_history, tasks, early_stop_step_single,
                                            early_stop_step_multi, early_stop_required_progress) and (
                                train_acc_tot > 0.99)
                if stop_training:  # early stopping
                    print("{}th epoch: earily stop triggered".format(epoch))
                    if write_file:
                        with open(file_name, 'a') as fp:
                            fp.write("{}th epoch: earily stop triggered".format(epoch))
                    break

        # because of the the nested loop
        if stop_training:
            break

    if calcpos:
        tpos_0, tpos_5, tpos_10, tpos_30 = test_model(test_loader, model, tasks, calcpos=True)
        print(
            'Test: 1: {}, 5: {}, 10: {}, 30: {}'.format(
                tpos_0, tpos_5, tpos_10, tpos_30
            ))
        torch.save(model.state_dict(), '{}.pkl'.format(file_name))
        torch.save(model, '{}.pt'.format(file_name))

        if write_file:
            with open(file_name, 'a') as fp:
                fp.write('Test: 1: {}, 5: {}, 10: {}, 30: {}'.format(
                tpos_0, tpos_5, tpos_10, tpos_30
            ))
            np.savez('{}_acc_history'.format(file_name), acc_history=acc_history)
        return (tpos_0, tpos_5, tpos_10, tpos_30)
    else:
        test_auc_sep, test_auc_tot = test_model(test_loader, model, tasks)
        torch.save(model.state_dict(), '{}.pkl'.format(file_name))
        torch.save(model, '{}.pt'.format(file_name))

        print('AUC of the model on the test set for single task: {}\n'
              'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))
        if write_file:
            with open(file_name, 'a') as fp:
                fp.write('AUC of the model on the test set for single task: {}\n'
                         'AUC of the model on the test set for all tasks: {}'.format(test_auc_sep, test_auc_tot))

        return (test_auc_tot)


tasks = all_tasks  # [task]
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

result = train(tasks, EAGCN_structure, n_den1, n_den2, file_name)
