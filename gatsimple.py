# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)
        self.alpha = alpha

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.atten_fact = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        # self.output_proj = nn.Linear(input_size, hidden_size)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_params()

    def reset_params(self):
        torch.nn.init.xavier_uniform_(self.atten_fact)

    # def reset_params(self):
    #     nn.init.xavier_normal_(self.atten_fact, gain=1.414)

    def forward(self, input, adj_identity):
        # input size (batch_size, N, input_size)
        input = self.input_proj(input) # (batch_size, N, hidden_size)
        batch_size, domain_num = input.size(0), input.size(1)
        # print(input.size())


        atten_input = torch.cat([input.repeat(1, 1, domain_num).view(batch_size, domain_num * domain_num, -1), input.repeat(1, domain_num, 1)], dim=-1)
        atten_input = atten_input.view(batch_size, domain_num, domain_num, self.hidden_size * 2)
        atten_scores = self.leakyrelu(torch.matmul(atten_input, self.atten_fact).squeeze(-1)) #(batch_size, domain_num, domain_num)

        # import pdb;
        # pdb.set_trace()

        atten_scores = atten_scores.squeeze(0)

        # atten_scores = atten_scores.masked_fill_(adj_identity.bool(), float('-inf'))
        atten_scores = F.softmax(atten_scores, dim=-1)
        ## add self-loop
        atten_scores = atten_scores + adj_identity
        atten_scores = self.drop(atten_scores)

        output = torch.bmm(atten_scores.unsqueeze(0), input) # (batch_size, domain_num, hidden_size)
        output = F.elu_(output) # (batch_size, domain_num, hidden_size)

        return output


class GAT(nn.Module):
    def __init__(self, args, input_size, hidden_size, dropout=0.0, heads_num=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = args.device
        self.heads_num = heads_num
        assert(hidden_size % self.heads_num == 0)
        self.layer_num = 2

        for layer_idx in range(self.layer_num):
            for head_idx in range(self.heads_num):
                if layer_idx == 0:
                    attention = GraphAttention(input_size, hidden_size//heads_num, dropout=self.dropout)
                else:
                    attention = GraphAttention(hidden_size, hidden_size//heads_num, dropout=self.dropout)
                self.add_module('layer_{}_attention_{}'.format(layer_idx, head_idx), attention)


    def forward(self, feature, edge_index):


        # input size (batch_size, N, input_size)
        values = torch.ones(edge_index.shape[1])
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=values,
                                            size=[feature.shape[1], feature.shape[1]])
        adjacency = adjacency.to(self.device)


        for layer_idx in range(self.layer_num):
            x = torch.cat([self._modules['layer_{}_attention_{}'.format(layer_idx, head_idx)](feature, adjacency) for head_idx in range(self.heads_num)], dim=-1)

        return x

if __name__ == '__main__':
    # test modules
    bs = 2
    node_num = 4
    node_vec_size = 6
    gnn_hidden_size = 8
    x = torch.ones((bs, node_num, node_vec_size))
    edges = []
    gat = GAT(node_vec_size, gnn_hidden_size)
    out = gat(x, edges)
    print(out)

    # x torch.Size([1, 229, 256])
