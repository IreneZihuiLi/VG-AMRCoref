import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# class GraphConvolution(nn.Module):
#     def __init__(self, input_dim, output_dim,use_bias=True):
#         super(GraphConvolution, self).__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.use_bias = use_bias
#         self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, adjacency, input_feature):
#
#         support = torch.mm(input_feature, self.weight)
#         output = torch.sparse.mm(adjacency, support)
#
#         return output
#
#     def __repr__(self):
#         # return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
#         return ''


class GCN(nn.Module):
    def __init__(self,args, input_dim, dropout=0.3):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, input_dim)
        self.gcn2 = GraphConvolution(input_dim, input_dim)
        self.gcn3 = GraphConvolution(input_dim, input_dim)

        self.device = args.device
        self.dropout = dropout

        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, feature, edge_index):


        values = torch.ones(edge_index.shape[1]).to(self.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=values, size=[feature.shape[0], feature.shape[0]])
        adjacency = adjacency.to(self.device)


        # import pdb;pdb.set_trace()

        hidden = F.relu(self.gcn1(feature, adjacency))
        # hidden = F.dropout(hidden, self.dropout)
        hidden = F.relu(self.gcn2(hidden, adjacency))

        hidden = F.relu(self.gcn3(hidden, adjacency))

        # torch.sum(concept_graph_reps[0],1)

        return hidden