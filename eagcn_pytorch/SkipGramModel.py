from utils import *
from collections import OrderedDict
from fractions import gcd
from sympy.ntheory import factorint


def construct_sgm_loader(x, y, target, batch_size, shuffle=True):
    data_set, labels, label_dict = construct_dataset(x, y, target)
    data_set = SkipGramModelDataset(data_set, labels, label_dict)
    return torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=SkipGramModelDataset.collate, shuffle=shuffle)


def lid(n):
    factors_dict = factorint(n)
    factors = sorted(list(factors_dict.keys()), reverse=True)
    ld = 1
    for factor in factors[:-1]:
        ld = ld * (factor ** factors_dict[factor])
    ld = ld * (factors[-1] ** (factors_dict[factors[-1]] - 1))
    return ld

class NNMolEmbed(nn.Module):
    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius):

        super(NNMolEmbed, self).__init__()

        self.radius = radius
        self.fp_len = fp_len
        self.edge_to_ix = edge_to_ix
        self.edge_word_len = edge_word_len
        self.node_to_ix = node_to_ix
        self.node_word_len = node_word_len
        self.edge_embeddings = nn.Embedding(len(edge_to_ix), fp_len)
        self.node_embeddings = nn.Embedding(len(node_to_ix), fp_len)

        self.summarize = nn.Conv2d(self.radius, 1, 1, bias=False)

        self.conv = {}

        for rad in range(radius):
            setattr(self, '_'.join(('conv', 'node', str(rad))),
                    nn.Conv1d(fp_len, fp_len, kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=lid(fp_len),
                              bias=False))
            self.conv[('node', rad)] = getattr(self, '_'.join(('conv', 'node', str(rad))))

            setattr(self, '_'.join(('conv', 'out', str(rad))),
                    nn.Conv1d(fp_len, fp_len, kernel_size=1,
                              stride=1, padding=0, dilation=1,
                              groups=lid(fp_len),
                              bias=False))
            self.conv[('out', rad)] = getattr(self, '_'.join(('conv', 'out', str(rad))))

            setattr(self, '_'.join(('conv', 'neighbor', 'att', str(rad))),
                    nn.Conv2d(2 * fp_len, fp_len, kernel_size=1,
                              stride=1,
                              padding=0, dilation=1, groups=gcd(2 * fp_len, fp_len), bias=False))
            self.conv[('neighbor', 'att', rad)] = getattr(self, '_'.join(('conv', 'neighbor', 'att', str(rad))))

            setattr(self, '_'.join(('conv', 'neighbor', 'act', str(rad))),
                    nn.Conv2d(2 * fp_len, fp_len, kernel_size=1, stride=1,
                              padding=0, dilation=1, groups=gcd(2 * fp_len, fp_len), bias=False))
            self.conv[('neighbor', 'act', rad)] = getattr(self, '_'.join(('conv', 'neighbor', 'act', str(rad))))

            setattr(self, '_'.join(('conv', 'edge', str(rad))),
                    nn.Conv2d(fp_len, fp_len,
                              kernel_size=1, stride=1, padding=0,
                              dilation=1,
                              groups=lid(fp_len), bias=False))
            self.conv[('edge', rad)] = getattr(self, '_'.join(('conv', 'edge', str(rad))))

    def embed_edges(self, adjs, bfts):
        return self.edge_embeddings(bfts.view(-1)).mul(adjs.view(-1).unsqueeze(1)).view(bfts.shape + (-1,))

    def embed_nodes(self, adjs, afms):
        nz = adjs.max(dim=2)[0]
        return self.node_embeddings(afms.view(-1)).mul(nz.view(-1).unsqueeze(1)).view(nz.shape + (-1,))

    def get_node_activation(self, node_data, layer):
        return self.conv[('out', layer)](node_data).sum(dim=-1)

    def get_next_node(self, node_data, layer):
        return self.conv[('node', layer)](node_data) + node_data

    def get_next_edge(self, edge_data, layer):
        return self.conv[('edge', layer)](edge_data) + edge_data

    def get_neighbor_act(self, adj_mat, node_data, edge_data, layer):
        lna = torch.mul(adj_mat.unsqueeze(1).expand(-1, self.fp_len, -1, -1), node_data.unsqueeze(3))
        lnb = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.fp_len, -1, -1)), edge_data)
        ln = torch.cat((lna, lnb), dim=1)
        att = self.conv[('neighbor', 'att', layer)](ln).sum(dim=2)
        act = self.conv[('neighbor', 'act', layer)](ln).sum(dim=2)
        return (att, act)

    @staticmethod
    def next_radius_adj_mat(adj_mat, adj_mats):
        next_adj_mat = torch.bmm(adj_mat, adj_mats[0])
        next_adj_mat[next_adj_mat!=0] = 1
        next_adj_mat = next_adj_mat - sum(adj_mats)
        next_adj_mat = torch.clamp(next_adj_mat - Variable(from_numpy(np.eye(next_adj_mat.size()[1])*next_adj_mat.size()[1]).float()), min=0)
        return next_adj_mat

    @staticmethod
    def concat_fps(inputs):
        return torch.cat(inputs, dim=1).view(inputs[0].shape[0], len(inputs), inputs[0].shape[1], -1)

    def forward(self, adjs, afms, bfts):  # bfts
        adjs = adjs.float()
        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).float()), min=0)
        edge_data = self.embed_edges(adjs_no_diag, bfts).permute(0, 3, 1, 2)
        node_data = self.embed_nodes(adjs, afms).permute(0, 2, 1)

        nz, _ = adjs.max(dim=2)

        fps = []

        node_current = node_data
        edge_current = edge_data
        adj_mat = adjs_no_diag
        adj_mats = []

        for radius in range(self.radius):
            fp = self.get_node_activation(node_current, radius)
            fps.append(fp)

            node_next = self.get_next_node(node_current, radius) # node_current
            neighbor_att, neighbor_act = self.get_neighbor_act(adj_mat, node_current, edge_current, radius) # get neighbor activation
            node_next = torch.mul(node_next, neighbor_att) + neighbor_act

            edge_next = torch.matmul(adj_mat.unsqueeze(1).expand((-1, self.fp_len, -1, -1)), edge_current)
            adj_mats.append(adj_mat)
            adj_mat = self.next_radius_adj_mat(adj_mat, adj_mats)
            edge_next = self.get_next_edge(edge_next, radius).mul(adj_mat.unsqueeze(1))

            node_current, edge_current = node_next, edge_next

        return torch.cat(fps, dim=-1)
        # return self.summarize(self.concat_fps(fps))

class SkipGramMolEmbed(nn.Module):
    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius):

        super(SkipGramMolEmbed, self).__init__()

        self.radius = radius
        self.fp_len = fp_len
        self.edge_word_len = edge_word_len
        self.node_word_len = node_word_len
        self.edge_embeddings = nn.Embedding(len(edge_to_ix), fp_len, sparse=False)
        self.node_embeddings = nn.Embedding(len(node_to_ix), fp_len, sparse=False)

    def embed_edges(self, adjs, bfts):
        return self.edge_embeddings(bfts.view(-1)).mul(adjs.view(-1).unsqueeze(1).float()).view(bfts.shape + (-1,))

    def embed_nodes(self, adjs, afms):
        nz = adjs.max(dim=2)[0].float()
        return self.node_embeddings(afms.view(-1)).mul(nz.view(-1).unsqueeze(1)).view(nz.shape + (-1,))

    def get_next_node(self, adj_mat, node_data, edge_data):
        # Must remember to sum over the rows!!!
        return edge_data.mul(node_data.unsqueeze(1)).sum(dim=-3)


    def batch_norm_nodes(self, node_data, adjs_diag_only, eps=1e-6):
        nz = adjs_diag_only.max(dim=2)[0].float().unsqueeze(2).expand(-1, -1, self.fp_len)
        mean = node_data.sum().div(nz.float().sum())
        var = node_data.pow(2).sum().div(nz.float().sum()) - mean.pow(2) + eps
        return (node_data-mean).div(var.sqrt()).mul(nz)


    def batch_norm_edges(self, edge_data, adjs_no_diag, eps=1e-6):
        adjs_no_diag = adjs_no_diag.unsqueeze(3).expand(-1, -1, -1, self.fp_len).float()
        mean = edge_data.sum().div(adjs_no_diag.sum())
        var = edge_data.pow(2).sum().div(adjs_no_diag.sum()) - mean.pow(2) + eps
        return (edge_data-mean).div(var.sqrt()).mul(adjs_no_diag)

    def forward(self, adjs, afms, bfts):
        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).long()), min=0)

        edge_data = self.batch_norm_edges(self.embed_edges(adjs_no_diag, bfts), adjs_no_diag)
        node_data = self.batch_norm_nodes(self.embed_nodes(adjs, afms), adjs - adjs_no_diag)



        node_current = node_data
        adj_mat = adjs_no_diag
        node_next = node_current
        fps = list()
        fps.append(node_data)

        # Very important sum over ROWS!!!
        r1 = edge_data.mul(adjs_no_diag.unsqueeze(3).float()).mul(node_data.unsqueeze(1))
        fps.append(self.batch_norm_edges(r1, adjs_no_diag).sum(dim=-3))
        t1 = adjs_no_diag.bmm(adjs_no_diag).clamp(max=1) - Variable(from_numpy(np.eye(adjs.size()[1])).long())
        r2 = r1.permute(0, 3, 1, 2).matmul(edge_data.permute(0, 3, 2, 1)).permute(0, 2, 3, 1).mul(
            t1.float().unsqueeze(3)).mul(node_data.unsqueeze(1))
        fps.append(self.batch_norm_edges(r2, t1).sum(dim=-3))

        # fps = list()
        # fps.append(node_next)
        #
        # for radius in range(self.radius):
        #     node_next = self.get_next_node(adj_mat, node_next, edge_data)
        #     fps.append(node_next)
        return torch.cat(fps, dim=-1).sum(dim=-2)

class SkipGramModel(nn.Module):
    """
    Attributes:
        fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius - passed down to SimpleMolEmbed
        w_embedding: Embedding for center word
        c_embedding: Embedding for context words
    """

    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius, batch_size):
        super(SkipGramModel, self).__init__()
        self.w_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius+1)
        # self.c_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius)
        # self.w_embedding = NNMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius+1)
        self.init_emb()
        self.loss = nn.BCEWithLogitsLoss()
        # self.bn = nn.BatchNorm1d(batch_size-1, affine=False)

    def init_emb(self):
        initrange = 0.5 / self.w_embedding.fp_len
        # self.w_embedding.edge_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.w_embedding.node_embeddings.weight.data.uniform_(-initrange, initrange)

        self.w_embedding.edge_embeddings.weight.data.normal_(0, initrange)
        self.w_embedding.node_embeddings.weight.data.normal_(0, initrange)

        # self.c_embedding.edge_embeddings.weight.data.uniform_(-0, 0)
        # self.c_embedding.node_embeddings.weight.data.uniform_(-0, 0)

    @staticmethod
    def euclidean_dist(x,y):
        X = x.pow(2).sum(dim=-1).view(x.shape[-2], -1)
        Y = y.pow(2).sum(dim=-1).view(-1, y.shape[-2])
        return (X + Y - 2 * x.matmul(y.t()))
        # return (X + Y -2*x.matmul(y.t())).sqrt()

    @staticmethod
    def remove_diag(x):
        return x[1 - from_numpy(np.eye(x.shape[0])).byte()].view(x.shape[-2], -1)

    @staticmethod
    def cosine_sim(A, B, eps=1e-8):
        dist_mat = torch.matmul(A, B.t())
        w1 = torch.norm(A, 2, 1).unsqueeze(1).expand(-1, B.shape[0])
        w2 = torch.norm(B, 2, 1).unsqueeze(0).expand(A.shape[0], -1)
        return dist_mat / (w1 * w2).clamp(min=eps)

    @staticmethod
    def norm_dists(dists, eps=1e-6):
        return (dists-dists.mean()).div((dists.var() + eps).sqrt())

    # def forward(self, pos_context, neg_context, sizes, padding):
    def forward(self, mols):

        fps = self.w_embedding(*mols[0:-1])
        # dists = self.remove_diag(self.euclidean_dist(fps, fps)).exp().pow(-1).clamp(max=1)
        dists = self.norm_dists(self.remove_diag(self.euclidean_dist(fps, fps)))
        # dists = self.bn(dists)
        labels = 1 - self.remove_diag(mols[-1].float().matmul(mols[-1].float().t()))
        # return (dists.mul(labels).neg() + dists.mul(1-labels).exp().add(-1).log() - dists.mul(1-labels)).neg().mean()
        return self.loss(dists, labels)

        # pos_fps = self.w_embedding(*pos[0:-1])
        # neg_fps = self.w_embedding(*neg[0:-1])
        # pos_dists = self.remove_diag(self.euclidean_dist(pos_fps, pos_fps))
        # mins = pos_dists.min(dim=-1)[0].unsqueeze(1)
        # pos_labels = Variable(from_numpy(np.ones(mins.shape)).float())
        # neg_dists = self.euclidean_dist(pos_fps, neg_fps)
        # neg_labels = Variable(from_numpy(np.zeros(neg_dists.shape)).float())
        # dists = torch.cat([mins, neg_dists], dim=-1).exp().pow(-1).clamp(max=1)
        # labels = torch.cat([pos_labels, neg_labels], dim=-1)
        # return self.loss(dists, labels.detach())


        # pos_labels = pos_dists <= pos_dists.min(dim=-1)[0].unsqueeze(1)
        # neg_dists = self.euclidean_dist(pos_fps, neg_fps)
        # neg_labels = Variable(from_numpy(np.zeros(neg_dists.shape)).byte())
        # dists = torch.cat([pos_dists, neg_dists], dim=-1).exp().pow(-1).clamp(max=1)
        # labels = torch.cat([pos_labels, neg_labels], dim=-1)
        # return self.loss(dists, labels.float().detach())
        # correct_labels = pos_context[-1].max(dim=-1)[1]
        # correct_label = sizes[0]
        # dists = []
        # for i, neg in enumerate(neg_context):
        #     neg_fps = self.w_embedding(*neg[0:-1])
        #     if correct_label == i:
        #         dist = self.remove_diag(self.euclidean_dist(word_fps, neg_fps))
        #     else:
        #         dist = self.euclidean_dist(word_fps, neg_fps)
        #         #     # dists.append(dist)
        #         #     # dist = self.euclidean_dist(word_fps, neg_fps)
        #         #     # dist = self.cosine_sim(word_fps, neg_fps)
        #     dists.append(dist.min(dim=-1)[0])
        # #
        # dists = torch.cat(dists, dim=-1).view(correct_labels.shape + (-1,))
        # dists = self.remove_diag(self.euclidean_dist(pos_fps, pos_fps))
        # correct_labels = dists.min(dim=-1)[1]
        # dists = torch.cat([dists, self.euclidean_dist(pos_fps, neg_fps)], dim=-1)
        # dists = F.softmin(dists, dim=-1).log()
        # return F.nll_loss(dists, correct_labels)

        # word_fps = self.w_embedding(*pos_context[0:-1])
        # neg_fps = self.w_embedding(torch.cat([neg[0] for neg in neg_context], dim=0),
        #                            torch.cat([neg[1] for neg in neg_context], dim=0),
        #                            torch.cat([neg[2] for neg in neg_context], dim=0))
        #
        # dists = self.euclidean_dist(word_fps, neg_fps)
        # self_dist = self.remove_diag(self.euclidean_dist(word_fps, word_fps))
        # correct_labels = self_dist.min(dim=-1)[1] + dists.shape[1]
        # dists = torch.cat([dists, self_dist], dim=-1)
        #
        # return F.nll_loss(F.softmin(dists, dim=-1).log(), correct_labels)


        # word_self = (torch.cat([neg[-1].max(dim=-1)[1] for neg in neg_context]) == sizes[0]).view(-1, dists.shape[1]).expand(dists.shape[0], -1)
        #
        # word_fps = self.w_embedding(*pos_context[0:-1])
        # correct_labels = pos_context[-1].max(dim=-1)[1]
        # correct_label = sizes[0]
        # dists = []
        # for i, neg in enumerate(neg_context):
        #     neg_fps = self.w_embedding(*neg[0:-1])
        #     if correct_label == i:
        #         dist = self.remove_diag(self.euclidean_dist(word_fps, neg_fps))
        #     else:
        #         dist = self.euclidean_dist(word_fps, neg_fps)
        # #     # dists.append(dist)
        # #     # dist = self.euclidean_dist(word_fps, neg_fps)
        # #     # dist = self.cosine_sim(word_fps, neg_fps)
        #     dists.append(dist.min(dim=-1)[0])
        # #
        # dists = torch.cat(dists, dim=-1).view(correct_labels.shape + (-1,))
        # dists = F.softmin(dists, dim=-1).log()
        # return F.nll_loss(dists, correct_labels)

        # pos_scores =
        # neg_scores = self.euclidean_dist(word_fps, neg_fps)
        #
        # scores = F.softmin(torch.cat([pos_scores, neg_scores], dim=-1), dim=-1).log()
        #
        # pos_labels = self.remove_diag(word_labels.matmul(word_labels.t()))
        # neg_labels = word_labels.matmul(neg_labels.t())
        # labels = torch.cat([pos_labels, neg_labels], dim=-1).max(dim=-1)[1]
        #
        # return F.nll_loss(scores, labels)

        # context_fps = self.c_embedding(*pos_context[0:-1])
        # neg_fps = self.c_embedding(*neg_context[0:-1])

        # iscore = F.logsigmoid(context_fps.matmul(word_fps.t()))
        # iscore = iscore.sum() - iscore.diag().sum()
        #
        # oscore = F.logsigmoid(neg_fps.neg().matmul(word_fps.t())).sum()

        # mask = np.ones((len(sizes), sizes.max()))
        # for i, size in enumerate(sizes):
        #     mask[i, size:] = 0
        # mask = Variable(from_numpy(mask))
        #
        # # iscore = F.logsigmoid(torch.matmul(pos_fps, mol_fps.t()))
        # iscore = F.logsigmoid(torch.bmm(pos_fps.view(mask.shape + (-1,)), mol_fps.unsqueeze(1).permute(0, 2, 1)).squeeze())
        # # oscore = F.logsigmoid(torch.matmul(neg_fps.neg(), mol_fps.t()))
        # oscore = F.logsigmoid(torch.bmm(neg_fps.view(mask.shape + (-1,)), mol_fps.unsqueeze(1).permute(0, 2, 1)).squeeze())
        # iscore = torch.div(torch.mul(iscore, mask.float()).sum(dim=1), Variable(from_numpy(sizes)).float())
        # oscore = torch.div(torch.mul(oscore, mask.float()).sum(dim=1), Variable(from_numpy(sizes)).float())
        # return -1 * (iscore + oscore)


class SkipGramModelDataset(Dataset):

    def __init__(self, data_list, labels, label_data_dict):
        super(SkipGramModelDataset, self).__init__()
        self.data_list = data_list
        self.labels = labels
        self.label_data_dict = label_data_dict

    # def __len__(self):
    #     return len(self.label_data_dict)

    # def __getitem__(self, item):
    #     label = item
    #     context = []
    #     for mol in self.label_data_dict[label]:
    #         context.append(mol.to_collate())
    #     mask = self.labels != item
    #     neg = [mol.to_collate() for mol in np.random.choice(self.data_list[mask], size=len(context))]
    #     # neg = [
    #     #     [mol.to_collate() for mol in np.random.choice(mols, size=np.max((1, len(mols)/10)))]
    #     #     for mols in self.label_data_dict.values()
    #     # ]
    #     # neg[item] = context
    #     return OrderedDict([
    #         ('context', context),
    #         ('neg', neg)
    #         # ('context_size', [item])
    #     ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item].to_collate()

    def getall(self):
        return mol_collate_func_class([mol.to_collate() for mol in self.data_list])

    @staticmethod
    def collate(batch):
        return mol_collate_func_class(batch)

    # @staticmethod
    # def collate(batch):
    #     # return mol_collate_func_class(batch[0]['context'])
    #     # neg_sizes = [[datum[0].shape[0] for datum in minibatch] for minibatch in batch[0]['neg']]
    #     # max_size = np.concatenate(neg_sizes).max()
    #     return (
    #         mol_collate_func_class(batch[0]['context']),
    #         mol_collate_func_class(batch[0]['neg'])
    #         # [mol_collate_func_class(minibatch, max_size) for minibatch in batch[0]['neg']],
    #         # np.array(batch[0]['context_size']),
    #         # np.array([0])
    #     )
