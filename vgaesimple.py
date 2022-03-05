import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class VGAE(nn.Module):
    def __init__(self, args, input_dim, dropout=0.5):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_dim, input_dim, dropout, act=F.relu)
        self.gc2 = GraphConvolution(input_dim, input_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(input_dim, input_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)


        self.device = args.device

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar+(1e-5))
            eps = torch.randn_like(std)
            # import pdb;pdb.set_trace()
            after = eps.mul(std).add_(mu)

            return after
            # return eps * std + mu
        else:
            return mu

    def _normalize_input(self, x):
        'normalzied input x'
        'before move to gcn, we normalized x'
        'normalize into [-1,1]'
        min = torch.min(x)
        max = torch.max(x)
        x = (((x - min) / (max - min)) - 0.5) * 2

        return x


    def _distribution_loss(self,mu,logvar):
        # TODO: mu, logvar too large
        mu = self._normalize_input(mu)
        logvar = self._normalize_input(logvar)

        KLD = -0.5 / self.n_nodes * torch.mean(torch.sum( 1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        return KLD

    def forward(self, feature, edge_index):
        self.n_nodes = feature.shape[0]
        values = torch.ones(edge_index.shape[1])
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=values,size=[feature.shape[0], feature.shape[0]])
        adj = adjacency.to(self.device)


        feature = self._normalize_input(feature)
        mu, logvar = self.encode(feature, adj)
        z = self.reparameterize(mu, logvar)


        z[z == float('-inf')] = 0
        z[z != z] = 0

        # z is the hidden value
        # return self.dc(z), mu, logvar, z

        kld = self._distribution_loss(mu,logvar)

        # print (kld)

        return self.dc(z), mu, logvar, mu, kld


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj