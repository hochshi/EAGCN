from utils import *
from collections import OrderedDict

def construct_sgm_loader(x, y, target, batch_size, shuffle=True):
    data_set, label_dict = construct_dataset(x, y, target)
    data_set = SkipGramModelDataset(data_set, label_dict)
    return torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=SkipGramModelDataset.collate, shuffle=shuffle)

class SkipGramMolEmbed(nn.Module):
    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius):

        super(SkipGramMolEmbed, self).__init__()

        self.radius = radius
        self.fp_len = fp_len
        self.edge_word_len = edge_word_len
        self.node_word_len = node_word_len
        self.edge_embeddings = nn.Embedding(len(edge_to_ix), fp_len, sparse=True)
        self.node_embeddings = nn.Embedding(len(node_to_ix), fp_len, sparse=True)

    def embed_edges(self, adjs, bfts):
        nz = adjs.view(-1).byte()
        new_edges = Variable(from_numpy(np.zeros(nz.shape + (self.fp_len,))).float())
        new_edges[nz.unsqueeze(1).expand(-1, self.fp_len)] = self.edge_embeddings(bfts).view(-1)
        return new_edges.view(adjs.shape + (-1,)).permute(0, 3, 1, 2).contiguous()

    def embed_nodes(self, adjs, afms):
        nz, _ = adjs.max(dim=2)
        nz = nz.view(-1).byte()
        new_nodes = Variable(from_numpy(np.zeros(nz.shape + (self.fp_len,))).float())
        new_nodes[nz.unsqueeze(1).expand(-1, self.fp_len)] = self.node_embeddings(afms).view(-1)
        return new_nodes.view(adjs.shape[0:-1] + (-1,)).permute(0, 2, 1).contiguous()

    def get_next_node(self, adj_mat, node_data, edge_data):
        lna = torch.mul(adj_mat.unsqueeze(1).expand(-1, self.fp_len, -1, -1).float(), node_data.unsqueeze(3))
        lnb = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.fp_len, -1, -1)).float(), edge_data)
        return torch.mul(lna, lnb).sum(dim=3)

    def forward(self, adjs, afms, bfts):
        edge_data = self.embed_edges(adjs, bfts)
        node_data = self.embed_nodes(adjs, afms)

        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).long()), min=0)

        node_current = node_data
        adj_mat = adjs_no_diag
        node_next = node_current

        fps = list()
        fps.append(node_next)

        for radius in range(self.radius):
            node_next = self.get_next_node(adj_mat, node_next, edge_data)
            fps.append(node_next)
        fps = torch.cat(fps, dim=1).sum(dim=-1)
        return fps

class SkipGramModel(nn.Module):
    """
    Attributes:
        fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius - passed down to SimpleMolEmbed
        w_embedding: Embedding for center word
        c_embedding: Embedding for context words
    """

    def __init__(self, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius):
        super(SkipGramModel, self).__init__()
        self.w_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius)
        self.c_embedding = SkipGramMolEmbed(fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.w_embedding.fp_len
        self.w_embedding.edge_embeddings.weight.data.uniform_(-initrange, initrange)
        self.w_embedding.node_embeddings.weight.data.uniform_(-initrange, initrange)

        self.c_embedding.edge_embeddings.weight.data.uniform_(-0, 0)
        self.c_embedding.node_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, mol, pos_context, neg_context, sizes, padding):
        mol_fps = self.w_embedding(*mol)
        pos_fps = self.c_embedding(*pos_context)
        neg_fps = self.c_embedding(*neg_context)
        mask = np.ones((len(sizes), sizes.max()))
        for i, size in enumerate(sizes):
            mask[i, size:] = 0
        mask = Variable(from_numpy(mask))
        # iscore = F.logsigmoid(torch.matmul(pos_fps, mol_fps.t()))
        iscore = F.logsigmoid(torch.bmm(pos_fps.view(mask.shape + (-1,)), mol_fps.unsqueeze(1).permute(0, 2, 1)).squeeze())
        # oscore = F.logsigmoid(torch.matmul(neg_fps.neg(), mol_fps.t()))
        oscore = F.logsigmoid(torch.bmm(neg_fps.view(mask.shape + (-1,)), mol_fps.unsqueeze(1).permute(0, 2, 1)).squeeze())
        iscore = torch.div(torch.mul(iscore, mask.float()).sum(dim=1), Variable(from_numpy(sizes)).float())
        oscore = torch.div(torch.mul(oscore, mask.float()).sum(dim=1), Variable(from_numpy(sizes)).float())
        return -1 * (sum(iscore)+sum(oscore))


class SkipGramModelDataset(Dataset):

    def __init__(self, data_list, label_data_dict):
        super(SkipGramModelDataset, self).__init__()
        self.data_list = data_list
        self.label_data_dict = label_data_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        label = self.data_list[item].label
        context = []
        for mol in self.label_data_dict[np.argmax(label)]:
            if self.data_list[item].index != mol.index:
                context.append(mol.to_collate())
        return OrderedDict([
            ('word', [self.data_list[item].to_collate()]),
            ('context', context),
            ('neg', [mol.to_collate() for mol in np.random.choice(self.data_list, size=len(context))]),
            ('context_size', [len(context)])
        ])
        # (
        #     [self.data_list[item].to_collate()],
        #     context,
        #     [mol.to_collate() for mol in np.random.choice(self.data_list, size=len(context))],
        #     [len(context)]
        # )

    def getall(self):
        return mol_collate_func_class([mol.to_collate() for mol in self.data_list])

    @staticmethod
    def collate(batch):
        words = []
        context = []
        neg = []
        sizes = []
        for datum in batch:
            sizes += datum['context_size']
        padding = np.max(sizes) - np.array(sizes)
        for datum, pad in zip(batch, padding):
            words += datum['word']
            context += datum['context'] + datum['word'] * pad
            neg += datum['neg'] + datum['word'] * pad
        return (
            mol_collate_func_class(words),
            mol_collate_func_class(context),
            mol_collate_func_class(neg),
            np.array(sizes),
            padding
        )