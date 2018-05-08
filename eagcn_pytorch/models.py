import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *
import torch
from torch.autograd import Variable
from fractions import gcd
from sympy.ntheory import factorint

# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
# DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


class MolGraph(nn.Module):
    def __init__(self, n_afeat, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, radius, edge_embedding_dim, node_embedding_dim, use_att):
        super(MolGraph, self).__init__()

        self.act_att = Parameter(FloatTensor([0.5]))

        self.radius = radius
        self.use_att = use_att
        self.fp_len = fp_len
        self.edge_to_ix = edge_to_ix
        self.edge_word_len = edge_word_len
        self.edge_embed_len = edge_embedding_dim
        self.node_to_ix = node_to_ix
        self.node_word_len = node_word_len
        self.node_embed_len = node_embedding_dim
        self.node_attr_len = n_afeat - self.node_word_len + self.node_embed_len
        self.edge_embeddings = nn.Embedding(len(edge_to_ix), edge_embedding_dim)
        self.node_embeddings = nn.Embedding(len(node_to_ix), node_embedding_dim)

        self.conv = {}
        self.bn = {}

        self.conv_out = nn.Conv1d(self.node_attr_len, fp_len, kernel_size=1, stride=1, padding=0, dilation=1,
                                  groups=gcd(self.node_attr_len, fp_len), bias=False)
        self.conv[('out', -1)] = getattr(self, 'conv_out')

        for rad in range(radius):
            setattr(self, '_'.join(('conv', 'node', str(rad))),
                    to_hw(nn.Conv1d(self.node_attr_len, self.node_attr_len, kernel_size=1,
                                    stride=1, padding=0, dilation=1, groups=Shi_GCN.largest_divisor(self.node_attr_len),
                                    bias=False)))
            self.conv[('node', rad)] = getattr(self, '_'.join(('conv', 'node', str(rad))))

            setattr(self, '_'.join(('conv', 'out', str(rad))), to_hw(nn.Conv1d(self.node_attr_len, fp_len, kernel_size=1,
                                                                               stride=1, padding=0, dilation=1,
                                                                               groups=gcd(self.node_attr_len, fp_len),
                                                                               bias=False)))
            self.conv[('out', rad)] = getattr(self, '_'.join(('conv', 'out', str(rad))))

            setattr(self, '_'.join(('bn', 'out', str(rad))), to_hw(nn.BatchNorm1d(fp_len, affine=False)))
            self.bn[('out', rad)] = getattr(self, '_'.join(('bn', 'out', str(rad))))

            setattr(self, '_'.join(('conv', 'neighbor', 'att', str(rad))),
                    to_hw(nn.Conv2d(self.node_attr_len + self.edge_embed_len,
                                    self.node_attr_len, kernel_size=1,
                                    stride=1,
                                    padding=0, dilation=1, groups=gcd(
                            self.node_attr_len + self.edge_embed_len, self.node_attr_len), bias=False)))
            self.conv[('neighbor', 'att', rad)] = getattr(self, '_'.join(('conv', 'neighbor', 'att', str(rad))))

            setattr(self, '_'.join(('conv', 'neighbor', 'act', str(rad))),
                    to_hw(nn.Conv2d(self.node_attr_len + self.edge_embed_len,
                                    self.node_attr_len, kernel_size=1,
                                    stride=1,
                                    padding=0, dilation=1, groups=gcd(
                            self.node_attr_len + self.edge_embed_len, self.node_attr_len), bias=False)))
            self.conv[('neighbor', 'act', rad)] = getattr(self, '_'.join(('conv', 'neighbor', 'act', str(rad))))

            setattr(self, '_'.join(('conv', 'edge', str(rad))),
                    to_hw(nn.Conv2d(self.edge_embed_len, self.edge_embed_len,
                                    kernel_size=1, stride=1, padding=0,
                                    dilation=1,
                                    groups=Shi_GCN.largest_divisor(
                                        self.edge_embed_len), bias=False)))
            self.conv[('edge', rad)] = getattr(self, '_'.join(('conv', 'edge', str(rad))))

    def embed_edges(self, adjs, bfts):
        nz = adjs.view(-1).byte()
        new_edges = Variable(from_numpy(np.zeros(nz.shape + (self.edge_embed_len,))).float())
        new_edges[nz.unsqueeze(1).expand(-1, self.edge_embed_len)] = self.edge_embeddings(bfts).view(-1)
        return new_edges.view(adjs.shape + (-1,)).permute(0, 3, 1, 2).contiguous()
        # return torch.mul(new_edges, adjs.unsqueeze(1).expand(-1, self.edge_embed_len, -1, -1))

    def embed_nodes(self, adjs, afms, axfms):
        nz, _ = adjs.max(dim=2)
        nz = nz.view(-1).byte()
        new_nodes = Variable(from_numpy(np.zeros(nz.shape + (self.node_attr_len,))).float())
        new_nodes[nz.unsqueeze(1).expand(-1, self.node_attr_len)] = torch.cat((self.node_embeddings(afms), axfms), dim=1).view(-1) # self.node_embeddings(afms).view(-1)

        # new_nodes = new_nodes.view(adjs.shape[0:-1] + (-1,)).permute(0, 2, 1).contiguous()
        return new_nodes.view(adjs.shape[0:-1] + (-1,)).permute(0, 2, 1).contiguous()
        # return torch.mul(new_nodes, nz.view(adjs.shape[0:-1]).unsqueeze(1).expand(-1, self.node_attr_len, -1).float())

    def get_node_activation(self, nz, node_data, layer):
        out = self.conv[('out', layer)](node_data)
        # out = torch.mul(out, nz.unsqueeze(1).expand(-1, self.fp_len, -1))
        # TODO: Should we average the atoms or sum them? summing weights bigger molecules
        out = torch.sum(out, dim=2)
        # F.elu seems to work better here? is it because I don't normalize the the node and edges?
        # Should I drop the dropout? It seems to cause issues
        # F.dropout is bad?
        # return F.dropout(F.relu(self.bn[('out', layer)](out)), p=self.dropout, training=self.training)
        # return out
        return out.div(nz.sum(dim=1).unsqueeze(1))

    def get_next_node(self, nz, node_data, layer):
        out = self.conv[('node', layer)](node_data)
        # out = torch.mul(out, nz.unsqueeze(1).expand(-1, self.node_attr_len, -1))
        return out+node_data

    def get_next_edge(self, adj, edge_data, layer):
        out = self.conv[('edge', layer)](edge_data)
        # out = torch.mul(adj.unsqueeze(1).expand(-1, self.edge_embed_len, -1, -1), out)
        return out+edge_data

    def get_neighbor_act(self, adj_mat, node_data, edge_data, layer):
        lna = torch.mul(adj_mat.unsqueeze(1).expand(-1, self.node_attr_len, -1, -1), node_data.unsqueeze(3))
        lnb = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_data)
        ln = torch.cat((lna, lnb), dim=1)
        att = self.conv[('neighbor', 'att', layer)](ln)
        act = self.conv[('neighbor', 'act', layer)](ln)
        # ln = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.node_attr_len, -1, -1)), ln)
        # TODO: We Should let the machine decide how it would like to summarize the neighbor data
        # TODO: Should we average over neighbors or sum them?
        # ln = ln.sum(dim=2)
        # nz, _ = adj_mat.max(dim=2)
        # nz = nz.view(-1)
        # ln = torch.mul(nz.view(adj_mat.shape[0:-1]).unsqueeze(1).expand(-1, self.node_attr_len, -1), ln)
        avg = adj_mat.sum(dim=2).unsqueeze(1)
        return (att.sum(dim=2).div(avg+1e-6), act.sum(dim=2).div(avg+1e-6))
        # return (att.sum(dim=2), act.sum(dim=2))

    def next_radius_adj_mat(self, adj_mat, adj_mats):
        next_adj_mat = torch.bmm(adj_mat, adj_mats[0])
        next_adj_mat[next_adj_mat!=0] = 1
        next_adj_mat = next_adj_mat - sum(adj_mats)
        next_adj_mat = torch.clamp(next_adj_mat - Variable(from_numpy(np.eye(next_adj_mat.size()[1])*next_adj_mat.size()[1]).float()), min=0)
        return next_adj_mat

    def forward(self, adjs, afms, axfms, bfts):
        if self.use_att:
            return self.att_forward(adjs, afms, axfms, bfts)
        else:
            return self.act_forward(adjs, afms, axfms, bfts)

    def act_forward(self, adjs, afms, axfms, bfts):
        edge_data = self.embed_edges(adjs, bfts)
        node_data = self.embed_nodes(adjs, afms, axfms)

        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).float()), min=0)

        nz, _ = adjs.max(dim=2)

        fps = []

        node_current = node_data
        edge_current = edge_data
        adj_mat = adjs_no_diag
        adj_mats = []

        # TODO: Should we conv edge_data (edge_current) for next iteration?
        # TODO: Should we conv node_data (node_current) for next iteration?

        for radius in range(self.radius):
            # fp = self.get_node_activation(nz, node_current, radius)
            # fps.append(fp)

            node_next = self.get_next_node(nz, node_current, radius)
            neighbor_next = self.get_neighbor_act(adj_mat, node_current, edge_current,
                                                  radius)  # get neighbor activation
            node_next = node_next + neighbor_next

            edge_next = torch.matmul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_data)
            adj_mats.append(adj_mat)
            adj_mat = self.next_radius_adj_mat(adj_mat, adj_mats)
            edge_next = self.get_next_edge(adj_mat, edge_next, radius)

            node_current, edge_current = node_next, edge_next


        # return fps
        return self.get_node_activation(nz, node_current, -1)

    def att_forward(self, adjs, afms, axfms, bfts):  # bfts
        edge_data = self.embed_edges(adjs, bfts)
        node_data = self.embed_nodes(adjs, afms, axfms)

        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).float()), min=0)

        nz, _ = adjs.max(dim=2)

        fps = []

        node_current = node_data
        edge_current = edge_data
        adj_mat = adjs_no_diag
        adj_mats = []

        # TODO: Should we conv edge_data (edge_current) for next iteration? - Yes for now
        # TODO: Should we conv node_data (node_current) for next iteration? - Yes for now

        for radius in range(self.radius):
            fp = self.get_node_activation(nz, node_current, radius)
            fps.append(fp)

            node_next = self.get_next_node(nz, node_current, radius) # node_current
            neighbor_att, neighbor_act = self.get_neighbor_act(adj_mat, node_current, edge_current, radius) # get neighbor activation
            # TODO: Should let the machine decide how to take the neighbor data into account
            # TODO: by summing activation? by multiplying (attention)?
            node_next = self.act_att*torch.mul(node_next, neighbor_att) + (1-self.act_att)*neighbor_act

            edge_next = torch.matmul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_current)
            adj_mats.append(adj_mat)
            adj_mat = self.next_radius_adj_mat(adj_mat, adj_mats)
            edge_next = self.get_next_edge(adj_mat, edge_next, radius)

            node_current, edge_current = node_next, edge_next

        return fps

class ConcatModule(nn.Module):
    def __init__(self):
        super(ConcatModule, self).__init__()

    def forward(self, inputs):
        return torch.cat(inputs, dim=1).view(inputs[0].shape[0], len(inputs), inputs[0].shape[1], -1)
        # return torch.cat(inputs, dim=1).view(inputs[0].shape[0], len(inputs), inputs[0].shape[1], -1).permute(0, 3, 1, 2).unsqueeze(1)

class MolGCN(nn.Module):
    def __init__(self, n_afeat, edge_to_ix, edge_word_len, node_to_ix, node_word_len, nclass, edge_embedding_dim=5,
                 node_embedding_dim=5, radius=3, use_att=True, fp_len=50):
        super(MolGCN, self).__init__()
        self.radius = radius + 1

        self.molgraph = MolGraph(n_afeat, fp_len, edge_to_ix, edge_word_len, node_to_ix, node_word_len, self.radius, edge_embedding_dim, node_embedding_dim, use_att)
        # self.molgraph = Shi_GCN(0, n_afeat, 10, 10, 10, 10, 10, 0, 0, 0, 0, 1, 1, 1, 1, 0.3, edge_to_ix, edge_word_len, node_to_ix, node_word_len)
        self.concat = ConcatModule()

        # no_stages = np.ceil(np.log2(n_afeat - node_word_len + node_embedding_dim))
        # stages_dim = np.power(2, [0] + (np.arange(no_stages) + 4).tolist()).astype(int)
        # stages_dim[0] = self.radius
        # stages_dim_tup = zip(stages_dim[:-1], stages_dim[1:])
        #
        # stages = [self._make_layer(*dims) for dims in stages_dim_tup]
        # self.stages = nn.Sequential(*stages)

        self.stages = nn.Conv2d(self.radius, 1, 1, bias=False)

        self.bn = nn.BatchNorm1d(fp_len, affine=False)

        self.classifier = nn.Linear(fp_len, nclass)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.ReLU(),
            # nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=(1, 2, 2), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=(2, 1), bias=False),
            nn.ReLU()
        )

    def forward(self, adjs, afms, axfms, bfts):
        x = self.molgraph(adjs, afms, axfms, bfts)
        x = self.concat(x)
        x = self.stages(x)
        # x = x.squeeze(-1).squeeze(-1).sum(dim=2)
        # x = x.sum(dim=-1).view(x.shape[0], -1)
        # x = self.bn(x.view(x.shape[0], -1))
        return self.classifier(x.view(x.shape[0], -1)), x

class NodeBatchNorm(nn.Module):
    def __init__(self, attr_len):
        super(NodeBatchNorm, self).__init__()
        self.attr_len = attr_len

    def forward(self, x, nz):
        xo = x.permute(1, 0, 2).contiguous().view(self.attr_len, -1)
        xo_mean = (xo.sum(dim=1) / nz.sum()).unsqueeze(1)
        xo_mask = nz.unsqueeze(0).expand(self.attr_len, -1, -1).contiguous().view(self.attr_len, -1)
        xo_std = (torch.mul((xo - xo_mean), xo_mask).sum(dim=1) / nz.sum() + (10 ** -6)).sqrt().unsqueeze(1)
        xo = torch.mul(((xo - xo_mean)/xo_std), xo_mask)
        return xo.view(self.attr_len, x.shape[0], -1).permute(1, 0, 2).contiguous()

class EdgeBatchNorm(nn.Module):
    def __init__(self, attr_len):
        super(EdgeBatchNorm, self).__init__()
        self.attr_len = attr_len

    def forward(self, x, adj):
        xo = x.permute(1, 0, 2, 3).contiguous().view(self.attr_len, -1)
        xo_mask = adj.unsqueeze(0).expand(self.attr_len, -1, -1, -1).contiguous().view(self.attr_len, -1)
        xo_mean = (xo.sum(dim=1) / adj.sum()).unsqueeze(1)
        xo_std = (torch.mul((xo - xo_mean), xo_mask).sum(dim=1) / adj.sum() + (10 ** -6)).sqrt().unsqueeze(1)
        xo = torch.mul(((xo - xo_mean) / xo_std), xo_mask)
        return xo.view(self.attr_len, x.shape[0], x.shape[2], -1).permute(1, 0, 2, 3).contiguous()

class Shi_GCN(nn.Module):
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, edge_to_ix, edge_word_len, node_to_ix, node_word_len, use_att=True, molfp_mode='sum',
                 hidden_size=50, radius=3,
                 edge_embedding_dim=5, node_embedding_dim=5):
        super(Shi_GCN, self).__init__()

        self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
        ngc1 = self.ngc1
        self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        ngc2 = self.ngc2
        self.hidden_size = ngc1
        hidden_size = self.hidden_size
        self.radius = radius+1

        self.use_att = use_att
        self.edge_to_ix = edge_to_ix
        self.edge_word_len = edge_word_len
        self.edge_embed_len = edge_embedding_dim
        self.node_to_ix = node_to_ix
        self.node_word_len = node_word_len
        self.node_embed_len = node_embedding_dim
        self.node_attr_len = n_afeat - self.node_word_len + self.node_embed_len
        self.edge_embeddings = to_hw(nn.Embedding(len(edge_to_ix), edge_embedding_dim))
        self.node_embeddings = to_hw(nn.Embedding(len(node_to_ix), node_embedding_dim))
        self.edge_bn = EdgeBatchNorm(edge_embedding_dim)
        self.node_bn = NodeBatchNorm(self.node_attr_len)

        self.conv = {}
        self.bn = {}

        for rad in range(self.radius):
            setattr(self, '_'.join(('conv', 'node', str(rad))),
                    to_hw(nn.Conv1d(self.node_attr_len, self.node_attr_len, kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=Shi_GCN.largest_divisor(self.node_attr_len),
                              bias=False)))
            self.conv[('node', rad)] = getattr(self, '_'.join(('conv', 'node', str(rad))))

            setattr(self, '_'.join(('conv', 'out', str(rad))), to_hw(nn.Conv1d(self.node_attr_len, ngc1, kernel_size=1,
                                                                         stride=1, padding=0, dilation=1,
                                                                         groups=gcd(self.node_attr_len, ngc1),
                                                                         bias=False)))
            self.conv[('out', rad)] = getattr(self, '_'.join(('conv', 'out', str(rad))))

            setattr(self, '_'.join(('bn', 'out', str(rad))), to_hw(nn.BatchNorm1d(ngc1, affine=False)))
            self.bn[('out', rad)] = getattr(self, '_'.join(('bn', 'out',str(rad))))

            setattr(self, '_'.join(('conv', 'neighbor', str(rad))), to_hw(nn.Conv2d(self.node_attr_len + self.edge_embed_len,
                                                                              self.node_attr_len, kernel_size=1,
                                                                              stride=1,
                                                                              padding=0, dilation=1, groups=gcd(
                    self.node_attr_len + self.edge_embed_len, self.node_attr_len), bias=False)))
            self.conv[('neighbor', rad)] = getattr(self, '_'.join(('conv', 'neighbor', str(rad))))

            setattr(self, '_'.join(('conv', 'edge', str(rad))), to_hw(nn.Conv2d(self.edge_embed_len, self.edge_embed_len,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          dilation=1,
                                                                          groups=Shi_GCN.largest_divisor(
                                                                              self.edge_embed_len), bias=False)))
            self.conv[('edge', rad)] = getattr(self, '_'.join(('conv', 'edge', str(rad))))

            # setattr(self, '_'.join(('bn', str(rad))), to_hw(nn.BatchNorm1d(self.node_attr_len)))
            # self.bn[rad] = getattr(self, '_'.join(('bn', str(rad))))

        # self.molfp_mode = molfp_mode
        # if 'sum' == molfp_mode:
        #     self.out_bn = to_hw(nn.BatchNorm1d(ngc1))
        #     self.dense = to_hw(nn.Linear(ngc1, nclass))
        # else:
        #     self.out_bn = to_hw(nn.BatchNorm1d(self.radius*ngc1))
        #     self.dense = to_hw(nn.Linear(self.radius*ngc1, nclass))
        #
        # self.out_softmax = to_hw(nn.Softmax(dim=1))
        self.dropout = dropout

    @staticmethod
    def largest_divisor(n):
        factors_dict = factorint(n)
        factors = sorted(list(factors_dict.keys()), reverse=True)
        ld = 1
        for factor in factors[:-1]:
            ld = ld * (factor ** factors_dict[factor])
        ld = ld * (factors[-1] ** (factors_dict[factors[-1]] - 1))
        return ld

    def fp_func(self, x):
        if 'sum' == self.molfp_mode:
            return sum(x)
        return torch.cat(x, dim=1)

    def embed_edges(self, adjs, bfts):
        nz = adjs.view(-1).byte()
        new_edges = Variable(from_numpy(np.zeros(nz.shape + (self.edge_embed_len,))).float())
        new_edges[nz.unsqueeze(1).expand(-1, self.edge_embed_len)] = self.edge_embeddings(bfts).view(-1)
        new_edges = new_edges.view(adjs.shape + (-1,)).permute(0, 3, 1, 2).contiguous()
        # return F.dropout(
        #     torch.mul(new_edges, adjs.unsqueeze(1).expand(-1, self.edge_embed_len, -1, -1)),
        #     p=self.dropout,
        #     training=self.training
        # )
        return torch.mul(new_edges, adjs.unsqueeze(1).expand(-1, self.edge_embed_len, -1, -1))

    def embed_nodes(self, adjs, afms, axfms):
        nz, _ = adjs.max(dim=2)
        nz = nz.view(-1).byte()
        new_nodes = Variable(from_numpy(np.zeros(nz.shape + (self.node_attr_len,))).float())
        new_nodes[nz.unsqueeze(1).expand(-1, self.node_attr_len)] = torch.cat((self.node_embeddings(afms), axfms), dim=1).view(-1) # self.node_embeddings(afms).view(-1)

        # new_nodes = torch.cat((new_nodes, axfms), dim=1)
        new_nodes = new_nodes.view(adjs.shape[0:-1] + (-1,)).permute(0, 2, 1).contiguous()

        # return F.dropout(
        #     torch.mul(new_nodes, nz.view(adjs.shape[0:-1]).unsqueeze(1).expand(-1, self.node_attr_len, -1).float()),
        #     p=self.dropout,
        #     training=self.training
        # )
        return torch.mul(new_nodes, nz.view(adjs.shape[0:-1]).unsqueeze(1).expand(-1, self.node_attr_len, -1).float())

    def get_self(self, nz, node_data, layer):
        return (
            self.get_self_out(nz, node_data, layer),
            node_data # self.get_self_act(nz, node_data, layer)
        )

    def get_node_activation(self, nz, node_data, layer):
        out = self.conv[('out', layer)](node_data)
        out = torch.mul(out, nz.unsqueeze(1).expand(-1, self.ngc1, -1))
        out = torch.sum(out, dim=2)
        # F.elu seems to work better here? is it because I don't normalize the the node and edges?
        # Should I drop the dropout? It seems to cause issues
        # F.dropout is bad?
        # return F.dropout(F.relu(self.bn[('out', layer)](out)), p=self.dropout, training=self.training)
        return out

    def get_next_node(self, nz, node_data, layer):
        out = self.conv[('node', layer)](node_data)
        out = torch.mul(out, nz.unsqueeze(1).expand(-1, self.node_attr_len, -1))
        # return out
        return out

    def get_next_edge(self, adj, edge_data, layer):
        out = self.conv[('edge', layer)](edge_data)
        out = torch.mul(adj.unsqueeze(1).expand(-1, self.edge_embed_len, -1, -1), out)
        # return out
        return out

    def get_neighbor_act(self, adj_mat, node_data, edge_data, layer):
        lna = torch.mul(adj_mat.unsqueeze(1).expand(-1, self.node_attr_len, -1, -1), node_data.unsqueeze(3))
        lnb = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_data)
        ln = torch.cat((lna, lnb), dim=1)
        ln = self.conv[('neighbor', layer)](ln)
        ln = torch.mul(adj_mat.unsqueeze(1).expand((-1, self.node_attr_len, -1, -1)), ln)
        ln = ln.sum(dim=3)
        nz, _ = adj_mat.max(dim=2)
        nz = nz.view(-1)
        ln = torch.mul(nz.view(adj_mat.shape[0:-1]).unsqueeze(1).expand(-1, self.node_attr_len, -1),
                       ln)
                  # self.out_softmax(ln))
        # return ln
        return ln

    def next_radius_adj_mat(self, adj_mat, adj_mats):
        next_adj_mat = torch.bmm(adj_mat, adj_mats[0])
        next_adj_mat[next_adj_mat!=0] = 1
        next_adj_mat = next_adj_mat - sum(adj_mats)
        next_adj_mat = torch.clamp(next_adj_mat - Variable(from_numpy(np.eye(next_adj_mat.size()[1])*next_adj_mat.size()[1]).float()), min=0)
        return next_adj_mat

    def print_embed_weights(self):
        print("Edge embeddings:")
        print(self.edge_embeddings.weight)
        print("Node embeddings:")
        print(self.node_embeddings.weight)

    def train(self, mode=True):
        super(Shi_GCN, self).train(mode=mode)

    def forward(self, adjs, afms, axfms, bfts):
        if self.use_att:
            return self.att_forward(adjs, afms, axfms, bfts)
        else:
            return self.act_forward(adjs, afms, axfms, bfts)

    def act_forward(self, adjs, afms, axfms, bfts):
        edge_data = self.embed_edges(adjs, bfts)
        node_data = self.embed_nodes(adjs, afms, axfms)

        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).float()), min=0)

        nz, _ = adjs.max(dim=2)

        fps = []
        nodes = []

        node_current = node_data
        edge_current = edge_data
        adj_mat = adjs_no_diag
        adj_mats = []

        # TODO: Should we conv edge_data (edge_current) for next iteration?
        # TODO: Should we conv node_data (node_current) for next iteration?

        # node_rep = node_current
        for radius in range(self.radius):
            fp = self.get_node_activation(nz, node_current, radius)
            fps.append(fp)
            # nodes.append(node_current)

            node_next = self.get_next_node(nz, node_current, radius)
            neighbor_next = self.get_neighbor_act(adj_mat, node_current, edge_current,
                                                  radius)  # get neighbor activation
            node_next = node_next + neighbor_next

            edge_next = torch.matmul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_data)
            adj_mats.append(adj_mat)
            adj_mat = self.next_radius_adj_mat(adj_mat, adj_mats)
            edge_next = self.get_next_edge(adj_mat, edge_next, radius)

            node_current, edge_current = node_next, edge_next

        # fps = self.fp_func(fps)
        # fps = self.out_bn(fps)

        # return self.dense(fps)
        return fps

    def att_forward(self, adjs, afms, axfms, bfts):  # bfts
        edge_data = self.embed_edges(adjs, bfts)
        node_data = self.embed_nodes(adjs, afms, axfms)

        adjs_no_diag = torch.clamp(adjs - Variable(from_numpy(np.eye(adjs.size()[1])).float()), min=0)

        nz, _ = adjs.max(dim=2)

        fps = []
        nodes = []

        node_current = node_data
        edge_current = edge_data
        adj_mat = adjs_no_diag
        adj_mats = []

        # TODO: Should we conv edge_data (edge_current) for next iteration?
        # TODO: Should we conv node_data (node_current) for next iteration?

        # node_rep = node_current
        for radius in range(self.radius):
            fp = self.get_node_activation(nz, node_current, radius)
            fps.append(fp)
            # nodes.append(node_current)

            node_next = self.get_next_node(nz, node_current, radius) # node_current
            neighbor_next = self.get_neighbor_act(adj_mat, node_current, edge_current, radius) # get neighbor activation
            node_next = torch.mul(node_next, neighbor_next)

            # edge_next = self.conv[('edge', radius)](edge_current)
            # edge_next = edge_current
            # multiply by the adj. mat to get the edges which are
            edge_next = torch.matmul(adj_mat.unsqueeze(1).expand((-1, self.edge_embed_len, -1, -1)), edge_data)
            adj_mats.append(adj_mat)
            adj_mat = self.next_radius_adj_mat(adj_mat, adj_mats)
            edge_next = self.get_next_edge(adj_mat, edge_next, radius)

            node_current, edge_current = node_next, edge_next
            # node_rep = node_current + neighbor_current
            # node_rep = torch.mul(node_current, neighbor_current)

        # fps = self.fp_func(fps)
        #
        # fps = self.out_bn(fps)
        #
        # return self.dense(fps).mul(-1).exp().add(1).mul(-1).exp()
        # return self.dense(fps)
        return fps

        # adjs_afms = torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        #     0, 3, 1, 2)
        # adjs_att = torch.cat((adjs_afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1)
        # att = self.soft(self.att0(adjs_att))
        # att = torch.sum(att, dim=3)
        # fp0 = torch.sum(torch.bmm(att, afms), dim=2)
        # fp0, hidden = self.rnn_forward(fp0, self.init_hidden().expand(fp0.size()[0], -1))
        #
        # fp0 = F.dropout(F.relu(self.bn(fp0)), p=self.dropout, training=self.training)
        #
        # # adjs_afms = torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2)
        # # adjs_att = torch.cat((torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1)
        # # att = self.soft(self.att0(torch.cat((torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1)))
        # # att = torch.sum(self.soft(self.att0(torch.cat((torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1))), dim=3)
        # # fp0 = torch.sum(torch.bmm(torch.sum(self.soft(self.att0(torch.cat((torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1))), dim=3), afms), dim=2)
        # # fp0, hidden = self.rnn_forward(torch.sum(torch.bmm(torch.sum(self.soft(self.att0(torch.cat((torch.mul(adjs_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #     0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1))), dim=3), afms), dim=2), self.init_hidden().expand(afms.size()[0], -1))
        # # fp0 = F.dropout(F.relu(self.bn(fp0)), p=self.dropout, training=self.training)
        #
        #
        # adjs_no_diag = torch.clamp(adjs - Variable(torch.eye(adjs.size()[1]).float()), min=0)
        # adjs_afms = torch.mul(adjs_no_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(0, 3, 1, 2)
        # adjs_att = torch.cat((adjs_afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1)
        # att = self.soft(self.att1(adjs_att))
        # att = torch.sum(att, dim=3)
        # fp1 = torch.sum(torch.bmm(att, afms), dim=2)
        # fp1, hidden = self.rnn_forward(fp1, hidden)
        # fp1 = F.dropout(F.relu(self.bn(fp1)), p=self.dropout, training=self.training)
        #
        # # fp1, hidden = self.rnn_forward(torch.sum(torch.bmm(torch.sum(self.soft(self.att0(
        # #     torch.cat((torch.mul(adjs_no_diag.unsqueeze(3).expand(-1, -1, -1, afms.size()[2]), afms.unsqueeze(1)).permute(
        # #         0, 3, 1, 2), bfts, OrderAtt, AromAtt, ConjAtt, RingAtt), dim=1))), dim=3), afms), dim=2),
        # #     self.init_hidden().expand(afms.size()[0], -1))
        # # fp1 = F.dropout(F.relu(self.bn(fp1)), p=self.dropout, training=self.training)
        #
        # # fp = self.bn(torch.cat((fp0, fp1), dim=1))
        #
        # fp = torch.cat((fp0, fp1), dim=1)
        #
        # x = self.den1(fp)
        # x = F.relu(self.bn_den1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.den2(x)
        # x = F.relu(self.bn_den2(x))
        # x = self.den3(x)
        # return x


class Concate_GCN(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, use_att = True, molfp_mode = 'sum'):
        super(Concate_GCN, self).__init__()

        self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
        ngc1 = self.ngc1
        self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        ngc2 = self.ngc2

        self.att1_1 = nn.Conv2d(n_bfeat, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_2 = nn.Conv2d(5, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.soft = nn.Softmax(dim=2)

        self.att2_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        # Compress atom representations to mol representation
        if molfp_mode =='sum':
            self.molfp1 = MolFP(self.ngc2)
            self.molfp2 = MolFP(self.ngc2)
            self.molfp = MolFP(self.ngc2)

        # Weighted average the mol representations to final mol representation.
        self.gc1_1 = GraphConvolution(n_afeat, n_sgc1_1, bias=True)
        self.gc1_2 = GraphConvolution(n_afeat, n_sgc1_2, bias=True)
        self.gc1_3 = GraphConvolution(n_afeat, n_sgc1_3, bias=True)
        self.gc1_4 = GraphConvolution(n_afeat, n_sgc1_4, bias=True)
        self.gc1_5 = GraphConvolution(n_afeat, n_sgc1_5, bias=True)

        self.gc2_1 = GraphConvolution(ngc1, n_sgc2_1, bias=True)
        self.gc2_2 = GraphConvolution(ngc1, n_sgc2_2, bias=True)
        self.gc2_3 = GraphConvolution(ngc1, n_sgc2_3, bias=True)
        self.gc2_4 = GraphConvolution(ngc1, n_sgc2_4, bias=True)
        self.gc2_5 = GraphConvolution(ngc1, n_sgc2_5, bias=True)


        self.den1 = Dense(self.ngc2, n_den1)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if use_cuda:
            self.att_bn1_1 = AFM_BatchNorm(n_sgc1_1).cuda()
            self.att_bn1_2 = AFM_BatchNorm(n_sgc1_2).cuda()
            self.att_bn1_3 = AFM_BatchNorm(n_sgc1_3).cuda()
            self.att_bn1_4 = AFM_BatchNorm(n_sgc1_4).cuda()
            self.att_bn1_5 = AFM_BatchNorm(n_sgc1_5).cuda()

            self.att_bn2_1 = AFM_BatchNorm(n_sgc2_1).cuda()
            self.att_bn2_2 = AFM_BatchNorm(n_sgc2_2).cuda()
            self.att_bn2_3 = AFM_BatchNorm(n_sgc2_3).cuda()
            self.att_bn2_4 = AFM_BatchNorm(n_sgc2_4).cuda()
            self.att_bn2_5 = AFM_BatchNorm(n_sgc2_5).cuda()

            self.molfp_bn = nn.BatchNorm1d(self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            self.att_bn1_1 = AFM_BatchNorm(n_sgc1_1)
            self.att_bn1_2 = AFM_BatchNorm(n_sgc1_2)
            self.att_bn1_3 = AFM_BatchNorm(n_sgc1_3)
            self.att_bn1_4 = AFM_BatchNorm(n_sgc1_4)
            self.att_bn1_5 = AFM_BatchNorm(n_sgc1_5)

            self.att_bn2_1 = AFM_BatchNorm(n_sgc2_1)
            self.att_bn2_2 = AFM_BatchNorm(n_sgc2_2)
            self.att_bn2_3 = AFM_BatchNorm(n_sgc2_3)
            self.att_bn2_4 = AFM_BatchNorm(n_sgc2_4)
            self.att_bn2_5 = AFM_BatchNorm(n_sgc2_5)

            self.molfp_bn = nn.BatchNorm1d(self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)


        self.dropout = dropout
        self.use_att = use_att

    def forward(self, adjs, afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt): #bfts
        mask = (1. - adjs) * -1e9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc1)
        mask4 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc2)

        A1_1 = self.att1_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_1 = self.soft(A1_1 + mask) * mask2
        x1_1 = self.gc1_1(A1_1, afms)
        x1_1 = F.relu(self.att_bn1_1(x1_1))
        x1_1 = F.dropout(x1_1, p=self.dropout, training=self.training)

        A1_2 = self.att1_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_2 = F.softmax(A1_2 + mask, 2) * mask2
        x1_2 = self.gc1_2(A1_2, afms)
        x1_2 = F.relu(self.att_bn1_2(x1_2))
        x1_2 = F.dropout(x1_2, p=self.dropout, training=self.training)

        A1_3 = self.att1_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_3 = F.softmax(A1_3 + mask, 2) * mask2
        x1_3 = self.gc1_3(A1_3, afms)
        x1_3 = F.relu(self.att_bn1_3(x1_3))
        x1_3 = F.dropout(x1_3, p=self.dropout, training=self.training)

        A1_4 = self.att1_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_4 = F.softmax(A1_4 + mask, 2) * mask2
        x1_4 = self.gc1_4(A1_4, afms)
        x1_4 = F.relu(self.att_bn1_4(x1_4))
        x1_4 = F.dropout(x1_4, p=self.dropout, training=self.training)

        A1_5 = self.soft(self.att1_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask) * mask2
        A1_5 = F.softmax(A1_5 + mask, 2) * mask2
        x1_5 = self.gc1_5(A1_5, afms)
        x1_5 = F.relu(self.att_bn1_5(x1_5))
        x1_5 = F.dropout(x1_5, p=self.dropout, training=self.training)

        x1 = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5), dim=2) * mask3

        A2_1 = F.softmax(self.att2_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask, 2) * mask2
        x2_1 = self.gc2_1(A2_1, x1)
        x2_1 = F.relu(self.att_bn2_1(x2_1))
        x2_1 = F.dropout(x2_1, p=self.dropout, training=self.training)

        A2_2 = F.softmax(self.att2_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_2 = self.gc2_2(A2_2, x1)
        x2_2 = F.relu(self.att_bn2_2(x2_2))
        x2_2 = F.dropout(x2_2, p=self.dropout, training=self.training)

        A2_3 = self.att2_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A2_3 = F.softmax(A2_3 + mask, 2) * mask2
        x2_3 = self.gc2_3(A2_3, x1)
        x2_3 = F.relu(self.att_bn2_3(x2_3))
        x2_3 = F.dropout(x2_3, p=self.dropout, training=self.training)

        A2_4 = F.softmax(self.att2_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_4 = self.gc2_4(A2_4, x1)
        x2_4 = F.relu(self.att_bn2_4(x2_4))
        x2_4 = F.dropout(x2_4, p=self.dropout, training=self.training)

        A2_5 = F.softmax(self.att2_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_5 = self.gc2_5(A2_5, x1)
        x2_5 = F.relu(self.att_bn2_5(x2_5))
        x2_5 = F.dropout(x2_5, p=self.dropout, training=self.training)

        x2 = torch.cat((x2_1, x2_2, x2_3, x2_4, x2_5), dim=2) * mask4

        x = self.molfp(x2)
        x = self.molfp_bn(x)

        x = self.den1(x)
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x

class Weighted_GCN(nn.Module):
    """
    @ The model used to train weighted sum structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    sum of n_sgc{i}_{j} over j: length of atom features updated by in {i}th layer for each relation.
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """
    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, use_att = True, molfp_mode = 'sum'):
        super(Weighted_GCN, self).__init__()

        self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
        ngc1 = self.ngc1
        self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        ngc2 = self.ngc2

        self.att1_1 = nn.Conv2d(n_bfeat, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_2 = nn.Conv2d(5, 1, kernel_size = 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.soft = nn.Softmax(dim=2)

        self.att2_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        # Compress atom representations to mol representation
        self.molfp1 = MolFP(self.ngc2)  # 132 is the molsize after padding.
        self.molfp2 = MolFP(self.ngc2)
        self.molfp = MolFP(self.ngc2)

        # Weighted average the mol representations to final mol representation.
        self.gc1_1 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_2 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_3 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_4 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_5 = GraphConvolution(n_afeat, ngc1, bias=True)

        self.gc2_1 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_2 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_3 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_4 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_5 = GraphConvolution(ngc1, ngc2, bias=True)

        self.den1 = Dense(self.ngc2, n_den1)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if use_cuda:
            self.att_bn1_1 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_2 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_3 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_4 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_5 = AFM_BatchNorm(ngc1).cuda()

            self.att_bn2_1 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_2 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_3 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_4 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_5 = AFM_BatchNorm(ngc2).cuda()

            self.molfp_bn = nn.BatchNorm1d(self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            self.att_bn1_1 = AFM_BatchNorm(ngc1)
            self.att_bn1_2 = AFM_BatchNorm(ngc1)
            self.att_bn1_3 = AFM_BatchNorm(ngc1)
            self.att_bn1_4 = AFM_BatchNorm(ngc1)
            self.att_bn1_5 = AFM_BatchNorm(ngc1)

            self.att_bn2_1 = AFM_BatchNorm(ngc2)
            self.att_bn2_2 = AFM_BatchNorm(ngc2)
            self.att_bn2_3 = AFM_BatchNorm(ngc2)
            self.att_bn2_4 = AFM_BatchNorm(ngc2)
            self.att_bn2_5 = AFM_BatchNorm(ngc2)

            self.molfp_bn = nn.BatchNorm1d(self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        self.ave1 = Ave_multi_view(5, self.ngc1)
        self.ave2 = Ave_multi_view(5, self.ngc2)

        self.dropout = dropout
        self.use_att = use_att

    def forward(self, adjs, afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt): #bfts
        mask = (1. - adjs) * -1e9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc1)
        mask4 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc2)

        A1_1 = self.att1_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_1 = self.soft(A1_1 + mask) * mask2
        x1_1 = self.gc1_1(A1_1, afms)
        x1_1 = F.relu(self.att_bn1_1(x1_1))
        x1_1 = F.dropout(x1_1, p=self.dropout, training=self.training) * mask3

        A1_2 = self.att1_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_2 = F.softmax(A1_2 + mask, 2) * mask2
        x1_2 = self.gc1_2(A1_2, afms)
        x1_2 = F.relu(self.att_bn1_2(x1_2))
        x1_2 = F.dropout(x1_2, p=self.dropout, training=self.training) * mask3

        A1_3 = self.att1_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_3 = F.softmax(A1_3 + mask, 2) * mask2
        x1_3 = self.gc1_3(A1_3, afms)
        x1_3 = F.relu(self.att_bn1_3(x1_3))
        x1_3 = F.dropout(x1_3, p=self.dropout, training=self.training) * mask3

        A1_4 = self.att1_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_4 = F.softmax(A1_4 + mask, 2) * mask2
        x1_4 = self.gc1_4(A1_4, afms)
        x1_4 = F.relu(self.att_bn1_4(x1_4))
        x1_4 = F.dropout(x1_4, p=self.dropout, training=self.training) * mask3

        A1_5 = self.soft(self.att1_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask) * mask2
        A1_5 = F.softmax(A1_5 + mask, 2) * mask2
        x1_5 = self.gc1_5(A1_5, afms)
        x1_5 = F.relu(self.att_bn1_5(x1_5))
        x1_5 = F.dropout(x1_5, p=self.dropout, training=self.training) * mask3

        x1 = torch.stack((x1_1, x1_2, x1_3, x1_4, x1_5), dim=0)
        x1 = self.ave1(x1)

        A2_1 = F.softmax(self.att2_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask, 2) * mask2
        x2_1 = self.gc2_1(A2_1, x1)
        x2_1 = F.relu(self.att_bn2_1(x2_1))
        x2_1 = F.dropout(x2_1, p=self.dropout, training=self.training) * mask4

        A2_2 = F.softmax(self.att2_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_2 = self.gc2_2(A2_2, x1)
        x2_2 = F.relu(self.att_bn2_2(x2_2))
        x2_2 = F.dropout(x2_2, p=self.dropout, training=self.training) * mask4

        A2_3 = self.att2_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A2_3 = F.softmax(A2_3 + mask, 2) * mask2
        x2_3 = self.gc2_3(A2_3, x1)
        x2_3 = F.relu(self.att_bn2_3(x2_3))
        x2_3 = F.dropout(x2_3, p=self.dropout, training=self.training) * mask4

        A2_4 = F.softmax(self.att2_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_4 = self.gc2_4(A2_4, x1)
        x2_4 = F.relu(self.att_bn2_4(x2_4))
        x2_4 = F.dropout(x2_4, p=self.dropout, training=self.training) * mask4

        A2_5 = F.softmax(self.att2_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_5 = self.gc2_5(A2_5, x1)
        x2_5 = F.relu(self.att_bn2_5(x2_5))
        x2_5 = F.dropout(x2_5, p=self.dropout, training=self.training) * mask4

        x2 = torch.stack((x2_1, x2_2, x2_3, x2_4, x2_5), dim=0)
        x2 = self.ave2(x2)

        x = self.molfp(x2)
        x = self.molfp_bn(x)

        x = self.den1(x)
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p =self.dropout, training=self.training)
        x = self.den2(x)
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x
