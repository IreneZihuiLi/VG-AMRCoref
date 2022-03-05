import torch
import torch.nn as nn
#from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from grn import *
from gcn import *
from gat import *
from vgae import *




class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.gnn_type = self.args.gnn_type
        self.node_dim = self.args.concept_dim
        self.gnn_layers = nn.ModuleList()
        self.num_layers = self.args.gnn_layer_num


        if self.gnn_type == 'gcn':
            for i in range(self.num_layers):
                # self.gnn_layers.append(GCN(self.args, self.node_dim))
                self.gnn_layers = GCN(self.args)

        elif self.gnn_type == 'vgae':
            for i in range(self.num_layers):
                self.gnn_layers = VGAE(self.args)

                # self.vgae = VGAE(self.args, self.node_dim)

        elif self.gnn_type == 'gat':
            for i in range(self.num_layers):
                self.gnn_layers = GAT(self.args)
            # self.hidden_dim = self.node_dim
            # # hidden layers
            # for i in range(self.num_layers):
            #     # due to multi-head, the in_dim = num_hidden * num_heads
            #     self.gnn_layers.append(GAT(self.args,self.node_dim, self.hidden_dim))
            #     # self.gnn_layers.append(GAT(self.node_dim, self.node_dim, self.args.heads))
        elif self.gnn_type == 'grn':
            self.gnn_layers = GRN(self.args)
        self.dropout = nn.Dropout(self.args.gnn_dropout)

    def forward(self, data, pretrain):

        if self.gnn_type == 'gcn':
            x = self.gnn_layers(data)

            # x, edge_index = data[0], data[8]
            # for i in range(self.num_layers):
            #     x = self.gnn_layers[i](x.squeeze(0), edge_index)
            # return x.unsqueeze(0), None
            return x, None
        elif self.gnn_type == 'vgae':


            for i in range(self.num_layers):
                x,vgae_loss = self.gnn_layers(data,pretrain)


            return x,vgae_loss
        elif self.gnn_type == 'gat':

            for i in range(self.num_layers):

                x = self.gnn_layers(data)

            # x, edge_index = data[0], data[8]
            #
            # for i in range(self.num_layers):
            #     # edge_index is not used
            #     x = F.elu(self.gnn_layers[i](x, edge_index)) # torch.Size([1, 481, 256])

            return x, None
        elif self.gnn_type == 'grn':
            for i in range(self.num_layers):
                _, x, _ = self.gnn_layers(data) # torch.Size([1, 45, 256])


            return x, _



'''
('concept', 6926, 1.0)
('token', 13883, 1.0)
('concept_char', 39, 1.0)
('token_char', 110, 1.0)
('relation', 86, 1.0) <----




'''