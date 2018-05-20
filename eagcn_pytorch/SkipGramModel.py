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
        return self.edge_embeddings(bfts.view(-1)).mul(adjs.view(-1).unsqueeze(1).float()).view(bfts.shape + (-1,))

    def embed_nodes(self, adjs, afms):
        nz = adjs.max(dim=2)[0].float()
        return self.node_embeddings(afms.view(-1)).mul(nz.view(-1).unsqueeze(1)).view(nz.shape + (-1,))

    def get_next_node(self, adj_mat, node_data, edge_data):
        return edge_data.mul(node_data.unsqueeze(1)).sum(dim=-2)

    def forward(self, adjs, afms, bfts):
        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).long()), min=0)

        edge_data = self.embed_edges(adjs_no_diag, bfts)
        node_data = self.embed_nodes(adjs, afms)



        node_current = node_data
        adj_mat = adjs_no_diag
        node_next = node_current

        fps = list()
        fps.append(node_next)

        for radius in range(self.radius):
            node_next = self.get_next_node(adj_mat, node_next, edge_data)
            fps.append(node_next)
        return torch.cat(fps, dim=-1).sum(dim=-2)

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

    def forward(self, pos_context, neg_context, sizes, padding):

        word_fps = self.w_embedding(*pos_context)
        context_fps = self.c_embedding(*pos_context)
        neg_fps = self.c_embedding(*neg_context)

        iscore = F.logsigmoid(context_fps.matmul(word_fps.t()))
        iscore = iscore.sum() - iscore.diag().sum()

        oscore = F.logsigmoid(neg_fps.neg().matmul(word_fps.t())).sum()

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
        return -1 * (iscore + oscore)


class SkipGramModelDataset(Dataset):

    def __init__(self, data_list, label_data_dict):
        super(SkipGramModelDataset, self).__init__()
        self.data_list = data_list
        self.label_data_dict = label_data_dict

    def __len__(self):
        return len(self.label_data_dict)

    def __getitem__(self, item):
        label = item
        context = []
        for mol in self.label_data_dict[label]:
            context.append(mol.to_collate())
        return OrderedDict([
            ('context', context),
            ('neg', [mol.to_collate() for mol in np.random.choice(self.data_list, size=len(context)-1)]),
            ('context_size', [len(context)])
        ])

    def getall(self):
        return mol_collate_func_class([mol.to_collate() for mol in self.data_list])

    @staticmethod
    def collate(batch):
        return (
            mol_collate_func_class(batch[0]['context']),
            mol_collate_func_class(batch[0]['neg']),
            np.array(batch[0]['context_size']),
            np.array([0])
        )