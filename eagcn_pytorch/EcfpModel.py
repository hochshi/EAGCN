from torch import nn
from SkipGramModel import SkipGramModel
from utils import *

def construct_ecfp_loader(x, y, target, batch_size, shuffle=True):
    data_set = EcfpDataSet(x, y)
    return torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=EcfpDataSet.collate, shuffle=shuffle)

def prep_input(ecfps, labels):
    return Variable(ecfps), Variable(labels)


class EcfpDataSet(Dataset):
    def __init__(self, data_list, labels):
        super(EcfpDataSet, self).__init__()
        self.data_list = data_list
        self.labels_list = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item], self.labels_list[item]

    @staticmethod
    def collate(batch):
        ecfps = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        return from_numpy(np.vstack(ecfps)).float(), from_numpy(np.vstack(labels)).float()

class EcfpModel(nn.Module):

    def __init__(self, ecfp_len, fp_len, batch_size):

        super(EcfpModel, self).__init__()
        self.fp_output = nn.Linear(ecfp_len, fp_len)
        self.bn = nn.BatchNorm1d(batch_size-1, affine=False)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, ecfps, labels):
        fps = self.fp_output(ecfps)
        dists = self.bn(SkipGramModel.remove_diag(SkipGramModel.euclidean_dist(fps, fps)))
        labels = 1 - SkipGramModel.remove_diag(labels.float().matmul(labels.float().t()))
        return self.loss(dists, labels)
