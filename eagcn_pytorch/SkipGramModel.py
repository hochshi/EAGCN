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

        # self.summarize = nn.Conv2d(self.radius, 1, 1, bias=False)

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
        # lna = torch.mul(adj_mat.unsqueeze(1).expand(-1, self.fp_len, -1, -1), node_data.unsqueeze(3))
        # lnb = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.fp_len, -1, -1)), edge_data)
        # ln = torch.cat((lna, lnb), dim=1)
        ln = torch.cat((
            torch.mul(adj_mat.unsqueeze(1).expand(-1, self.fp_len, -1, -1), node_data.unsqueeze(3)),
            torch.mul(adj_mat.unsqueeze(1).expand((-1, self.fp_len, -1, -1)), edge_data)
        ), dim=1)
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
    def __init__(self, e_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius, out_len):

        super(SkipGramMolEmbed, self).__init__()

        self.radius = radius
        self.fp_len = e_len
        self.edge_word_len = edge_word_len
        self.node_word_len = node_word_len
        self.edge_embeddings = nn.Embedding(len(edge_to_ix), e_len, sparse=False)
        self.node_embeddings = nn.Embedding(len(node_to_ix), e_len, sparse=False)
        self.bl1_edge = nn.Bilinear(e_len, e_len, e_len, bias=False)
        self.bl1_node = nn.Bilinear(e_len, e_len, e_len, bias=False)
        self.bl2_edge = nn.Bilinear(e_len, e_len, e_len, bias=False)
        self.bl2_node = nn.Bilinear(e_len, e_len, e_len, bias=False)
        # self.reduceconv = nn.Conv1d(e_len ** (radius * 2 + 1), out_len, kernel_size=1, groups=gcd(e_len ** (radius * 2 + 1), out_len))


    def embed_edges(self, adjs, bfts):
        # return self.edge_embeddings(bfts.view(-1)).mul(adjs.view(-1).unsqueeze(1).float())\
        #     .view(bfts.shape + (-1,)).permute(0, 3, 2, 1).contiguous()
        return self.edge_embeddings(bfts.view(-1)).mul(adjs.view(-1).unsqueeze(1).float()) \
                .view(bfts.shape + (-1,)).contiguous()

    def embed_nodes(self, adjs, afms):
        nz = adjs.max(dim=2)[0].float()
        # return self.node_embeddings(afms.view(-1)).mul(nz.view(-1).unsqueeze(1))\
        #     .view(nz.shape + (-1,)).permute(0, 2, 1).contiguous()
        return self.node_embeddings(afms.view(-1)).mul(nz.view(-1).unsqueeze(1)) \
            .view(nz.shape + (-1,)).contiguous()

    def get_next_node(self, adj_mat, node_data, edge_data):
        # Must remember to sum over the rows!!!
        return edge_data.mul(node_data.unsqueeze(1)).sum(dim=-3)


    def batch_norm_nodes(self, node_data, adjs_diag_only, eps=1e-6):
        nz = adjs_diag_only.max(dim=2)[0].float().unsqueeze(2)
        mean = node_data.sum(dim=-2).sum(dim=-2).div(nz.sum())
        var = node_data.pow(2).sum(dim=-2).sum(dim=-2).div(nz.sum()) - mean.pow(2) + eps
        return (node_data - mean).mul(nz).div(var.sqrt())
        # nz = adjs_diag_only.max(dim=2)[0].float().unsqueeze(2).expand(-1, -1, self.fp_len)
        # mean = node_data.sum().div(nz.float().sum())
        # var = node_data.pow(2).sum().div(nz.float().sum()) - mean.pow(2) + eps
        # return (node_data-mean).div(var.sqrt()).mul(nz)


    def batch_norm_edges(self, edge_data, adjs_no_diag, eps=1e-6):
        adjs_no_diag = adjs_no_diag.unsqueeze(3).expand(-1, -1, -1, self.fp_len).float()
        mean = edge_data.sum().div(adjs_no_diag.sum())
        var = edge_data.pow(2).sum().div(adjs_no_diag.sum()) - mean.pow(2) + eps
        return (edge_data-mean).div(var.sqrt()).mul(adjs_no_diag)

    def forward2(self, adjs, afms, bfts):
        adjs_no_diag = torch.clamp(adjs - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(False), min=0)

        edge_data = self.embed_edges(adjs_no_diag, bfts)
        node_data = self.embed_nodes(adjs, afms)

        fps = node_data.unsqueeze(2).expand(-1, -1, node_data.shape[1], -1)
        nodes = node_data.unsqueeze(1).expand(-1, node_data.shape[1], -1, -1)

        for i in range(self.radius):
            fps = torch.einsum('abcd,abcz->abcdz',(fps.clone(), edge_data.clone())).view(edge_data.shape[:-1] + (-1,))
            fps = torch.einsum('abcd,abcz->abcdz',(fps.clone(), nodes.clone())).view(edge_data.shape[:-1]+(-1,))
            if i+1 < self.radius:
                fps = fps.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1))\
                    .mul((1 - adjs).unsqueeze(1).clamp(min=0)).permute(0, 2, 3, 1)
                edge_data = edge_data.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1))\
                    .mul((1 - adjs).unsqueeze(1).clamp(min=0)).permute(0, 2, 3, 1)


        # return self.reduceconv(F.normalize(fps.sum(dim=2).sum(dim=1), dim=1).unsqueeze(-1)).squeeze()
        adjs.requires_grad_(True)
        adjs_no_diag.requires_grad_(True)
        return torch.cat([self.reduceconv(F.normalize(fps.sum(dim=2).sum(dim=1), dim=1).unsqueeze(-1)).squeeze(),
                   adjs_no_diag.sum(dim=-1).sum(dim=-1).unsqueeze(-1) / 2,
                   (adjs - adjs_no_diag).sum(dim=-1).sum(dim=-1).unsqueeze(-1)], dim=-1)


        # Create outer product of each node with its edges - this is done row wise - i.e
        # row 0 has the edges coming out of node 0 with outer product of node 0
        # r0 = torch.einsum('abcd,abcz->abcdz',
        #                   (node_data.unsqueeze(2).expand(-1, -1, node_data.shape[1], -1), edge_data))\
        #     .view(edge_data.shape[:-1]+(-1,))
        # # Create next neighbourhood edge
        # t1 = adjs_no_diag.bmm(adjs_no_diag).clamp(max=1) - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(
        #     False)
        # # Move the outer product to the location first layer neighbor
        # r1 = r0.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1)).mul(t1.unsqueeze(1)).permute(0, 2, 3, 1)
        # # Create outer product of the previous with the node connected (the neighbor) by that edge
        # r2 = torch.einsum('abcd,abcz->abcdz',
        #                   (node_data.unsqueeze(1).expand(-1, node_data.shape[1], -1, -1), r1))\
        #     .view(edge_data.shape[:-1]+(-1,))
        #
        #
        # r3 = torch.einsum('abcd,abcz->abcdz', (edge_data, r2)).view(edge_data.shape[:-1]+(-1,))
        #
        # t2 = (t1.bmm(adjs_no_diag) - adjs - t1).clamp(min=0)
        # # Move the outer product to the location of the second layer node
        # r3 = r2.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1)).mul(t2.unsqueeze(1)).permute(0, 2, 3, 1)
        # r4 = torch.einsum('abcd,abcz->abcdz',
        #                   (node_data.unsqueeze(1).expand(-1, node_data.shape[1], -1, -1), r3)) \
        #     .view(edge_data.shape[:-1] + (-1,))


    def forward(self, adjs, afms, bfts):
        adjs_no_diag = torch.clamp(adjs - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(False), min=0)
        mols_sizes = (adjs - adjs_no_diag).sum(dim=-2).sum(dim=-1).unsqueeze(-1)

        edge_data = F.normalize(self.embed_edges(adjs_no_diag, bfts), dim=-1)
        node_data = F.normalize(self.embed_nodes(adjs, afms), dim=-1)

        out = list()
        out.append(F.normalize(node_data.sum(dim=-2).div(mols_sizes), dim=-1))

        fps = node_data.unsqueeze(2).expand(-1, -1, node_data.shape[1], -1).contiguous()
        nodes = node_data.unsqueeze(1).expand(-1, node_data.shape[1], -1, -1).contiguous()


        for i in range(self.radius):
            # fps = self.bl1_edge(fps, edge_data)
            # fps = self.bl1_node(fps, nodes)
            bl_node = getattr(self, 'bl{}_node'.format(i+1))
            bl_edge = getattr(self, 'bl{}_edge'.format(i+1))
            fps = F.normalize(bl_node(F.normalize(bl_edge(fps, edge_data), dim=-1), nodes), dim=-1)
            if 0 == i:
                out.append(F.normalize(fps.sum(dim=-2).div(adjs_no_diag.sum(dim=-2).unsqueeze(-1) + 1e-6).sum(dim=-2).div(mols_sizes), dim=-1))
            else:
                out.append(F.normalize(fps.sum(dim=-2).div((adjs_no_diag.bmm(adjs_no_diag).clamp(max=1) - adjs).clamp(min=0)
                                               .sum(dim=-2).unsqueeze(-1) + 1e-6).sum(dim=-2).div(mols_sizes), dim=-1))
            if i+1 < self.radius:
                fps = fps.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1))\
                    .mul((1 - adjs).unsqueeze(1).clamp(min=0)).permute(0, 2, 3, 1).contiguous()
                edge_data = edge_data.permute(0, 3, 1, 2).matmul(adjs_no_diag.unsqueeze(1))\
                    .mul((1 - adjs).unsqueeze(1).clamp(min=0)).permute(0, 2, 3, 1).contiguous()
        return torch.cat([
            F.normalize(torch.cat(out, dim=-1), dim=-1),
            adjs_no_diag.sum(dim=-1).sum(dim=-1).unsqueeze(-1) / 2,
            mols_sizes
        ], dim=-1)


    def forward3(self, adjs, afms, bfts):
        adjs_no_diag = torch.clamp(adjs - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(False), min=0)

        edge_data = F.normalize(self.embed_edges(adjs_no_diag, bfts), dim=-1)
        node_data = F.normalize(self.embed_nodes(adjs, afms), dim=-1)

        fps = list()
        fps.append(node_data)

        node_data = node_data.unsqueeze(2).expand(-1, -1, edge_data.shape[2], -1).contiguous()

        r1 = F.normalize(self.bl1(edge_data, node_data), dim=-1)
        fps.append(self.bl0_1(fps[0],r1.sum(dim=-3)))
        t1 = adjs_no_diag.bmm(adjs_no_diag).clamp(max=1) - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(
            False)
        # r2 = F.normalize(self.bl3(F.normalize(self.bl2(edge_data, r1), dim=-1), node_data), dim=-1)\
        #     .mul(t1.float().unsqueeze(-1))
        r2 = F.normalize(self.bl2(F.normalize(r1.permute(0, 3, 1, 2).matmul(edge_data.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).mul(t1.unsqueeze(-1)), dim=-1), node_data), dim=-1)
        # fps.append(r2.sum(dim=-3))
        return self.bl1_2(fps[1], r2.sum(dim=-3)).sum(dim=-2)
        # return torch.cat(fps, dim=-2).sum(dim=-1)

        """
        This is working quite well - 
        trying to improve by using bilinear layers instead of multiplying directly the node and edge data
        # Very important sum the rows!!!
        # TODO: Should I try a different mechanism in which edge data and node data are multiplied
        # but next level data is added? n1-(e1)-n2-(e2)-n3 turns into: (e1*n2)+(e2+n3)
        # or the other way around? (e1+n2)*(e2+n3)?
        r1 = edge_data.mul(adjs_no_diag.unsqueeze(1).float()).mul(node_data.unsqueeze(-1))
        fps.append(r1.sum(dim=-2))
        t1 = adjs_no_diag.bmm(adjs_no_diag).clamp(max=1) - from_numpy(np.eye(adjs.size()[1])).float().requires_grad_(False)
        # r2 = r1.matmul(edge_data).mul(t1.float().unsqueeze(1)).mul(node_data.unsqueeze(-1))
        fps.append(r1.matmul(edge_data).mul(t1.float().unsqueeze(1)).mul(node_data.unsqueeze(-1)).sum(dim=-2))
        # fps = list()
        # fps.append(node_next)
        #
        # for radius in range(self.radius):
        #     node_next = self.get_next_node(adj_mat, node_next, edge_data)
        #     fps.append(node_next)
        return torch.cat(fps, dim=-2).sum(dim=-1)
        """

class SkipGramModel(nn.Module):
    """
    Attributes:
        fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius - passed down to SimpleMolEmbed
        w_embedding: Embedding for center word
        c_embedding: Embedding for context words
    """

    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius, nclass):
        super(SkipGramModel, self).__init__()
        self.w_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius, 50)
        # self.c_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius)
        # self.w_embedding = NNMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius+1)
        # self.loss = nn.BCEWithLogitsLoss()
        # self.bn = nn.BatchNorm1d(fp_len*(radius+1), affine=True)
        # self.classifier = nn.Linear(fp_len*(radius+1), nclass)
        self.bn = nn.BatchNorm1d(fp_len*(radius+1)+2, affine=True)
        self.classifier = nn.Linear(fp_len*(radius+1)+2, nclass)
        self.init_emb()

    def init_emb(self):
        # initrange = 0.5 / self.w_embedding.fp_len
        initrange = 0.25
        self.w_embedding.edge_embeddings.weight.data.uniform_(-initrange, initrange)
        self.w_embedding.node_embeddings.weight.data.uniform_(-initrange, initrange)
        #
        self.w_embedding.bl1_node.weight.data.uniform_(-initrange, initrange)
        self.w_embedding.bl1_edge.weight.data.uniform_(-initrange, initrange)
        self.w_embedding.bl2_edge.weight.data.uniform_(-initrange, initrange)
        self.w_embedding.bl2_node.weight.data.uniform_(-initrange, initrange)

        self.bn.weight.data.normal_(1, initrange)
        self.bn.bias.data.fill_(0)

        # self.w_embedding.edge_embeddings.weight.data.normal_(0, 0.05)
        # self.w_embedding.node_embeddings.weight.data.normal_(0, 0.05)
        # self.w_embedding.bl1_node.weight.data.normal_(0, 0.05)
        # self.w_embedding.bl1_edge.weight.data.normal_(0, 0.05)
        # self.w_embedding.bl2_edge.weight.data.normal_(0, 0.05)
        # self.w_embedding.bl2_node.weight.data.normal_(0, 0.05)

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
        return self.classifier(self.bn(fps))

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
