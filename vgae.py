import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from config import parse_config



class MLPreadout(nn.Module):
    def __init__(self, embed_dim):
        super(MLPreadout, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/4)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/4), 2)
        )
        self.softmax=nn.Softmax()

    def forward(self, x):
        # x shape: [1, node_num, embed_dim]
        x = self.layers(x[0])
        x = torch.sum(x,dim=0)
        return self.softmax(x)


class MLPAggregation(nn.Module):
    def __init__(self, embed_dim):
        super(MLPAggregation, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/4)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/4), embed_dim),
        )
        self.softmax=nn.Softmax()

    def forward(self, x):
        # x shape: [1, node_num, embed_dim]
        x = self.layers(x[0])
        return x

class GCNConv(nn.Module):
    def __init__(self, args):
        super(GCNConv, self).__init__()
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)

        'combine neighbours'
        self.neighbors = nn.Linear(self.edge_dim+self.node_dim,self.node_dim)

        'readout using MLP'
        self.MLPreadout = MLPreadout(self.node_dim)

        # 'aggregation using MLP'
        # self.MLPAggregation = MLPAggregation(self.node_dim)


    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max # in_indices
        # edges shapes : batch_size, node_num, edge_labels # in_edges, out_edges



        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative = batch_data
        node_reps = self.dropout(node_reps)

        # ==== input from in neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        in_edge_reps = self.edge_embedding(in_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        in_node_reps = self.collect_neighbors(node_reps, in_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        in_reps = torch.cat([in_node_reps, in_edge_reps], 3)

        in_reps = in_reps.mul(in_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        in_reps = in_reps.sum(dim=2)

        # ==== input from out neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        out_edge_reps = self.edge_embedding(out_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        out_node_reps = self.collect_neighbors(node_reps, out_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        out_reps = torch.cat([out_node_reps, out_edge_reps], 3)


        out_reps = out_reps.mul(out_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        out_reps = out_reps.sum(2)

        out_nodes = self.neighbors(out_reps.squeeze(0))
        in_nodes = self.neighbors(in_reps.squeeze(0))




        'this works the best'
        node_hidden = node_reps + out_nodes + in_nodes
        # node_hidden = self.MLPAggregation(node_hidden).unsqueeze(0)

        # repack for the next layer
        batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative]



        'add a readout function to capture graph-level reps, shape: [2]'
        readout = self.MLPreadout(node_hidden)


        'return node representation: shape [1, node_num, 256]'
        return node_hidden, batch_data, readout


    def collect_neighbors(self, node_reps, index):
        # node_rep: [batch_size, node_num, node_dim]
        # index: [batch_size, node_num, neighbors_num]
        batch_size = index.size(0)
        node_num = index.size(1)
        neighbor_num = index.size(2)
        rids = torch.arange(0, batch_size).to(self.args.device)  # [batch]
        rids = rids.reshape([-1, 1, 1])  # [batch, 1, 1]
        rids = rids.repeat(1, node_num, neighbor_num)  # [batch, nodes, neighbors]
        indices = torch.stack((rids, index), 3)  # [batch, nodes, neighbors, 2]
        return node_reps[indices[:, :, :, 0], indices[:, :, :, 1], :]


class GRNConv(nn.Module):
    def __init__(self, args):
        super(GRNConv, self).__init__()
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)
        # input gate
        self.W_ig_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_ig_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_ig_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_ig_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # forget gate
        self.W_fg_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_fg_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_fg_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_fg_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # output gate
        self.W_og_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_og_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_og_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_og_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # cell
        self.W_cell_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_cell_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_cell_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_cell_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        'readout using MLP'
        self.MLPreadout = MLPreadout(self.node_dim)

    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max # in_indices
        # edges shapes : batch_size, node_num, edge_labels # in_edges, out_edges

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative = batch_data

        # node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, _ = batch_data[:-1]
        node_reps = self.dropout(node_reps)
        batch_size = node_reps.size(0)
        node_num_max = node_reps.size(1)


        # ==== input from in neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        in_edge_reps = self.edge_embedding(in_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        in_node_reps = self.collect_neighbors(node_reps, in_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        in_reps = torch.cat([in_node_reps, in_edge_reps], 3)

        in_reps = in_reps.mul(in_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        in_reps = in_reps.sum(dim=2)
        in_reps = in_reps.reshape([-1, self.node_dim + self.edge_dim])

        # ==== input from out neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        out_edge_reps = self.edge_embedding(out_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        out_node_reps = self.collect_neighbors(node_reps, out_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        out_reps = torch.cat([out_node_reps, out_edge_reps], 3)




        out_reps = out_reps.mul(out_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        out_reps = out_reps.sum(2)
        out_reps = out_reps.reshape([-1, self.node_dim + self.edge_dim])

        node_hidden = node_reps
        node_cell = torch.zeros(batch_size, node_num_max, self.hidden_dim).to(self.args.device)

        # node_reps = node_reps.reshape([-1, self.word_dim])

        graph_representations = []
        for i in range(self.gnn_layers):
            # in neighbor hidden
            # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
            in_pre_hidden = self.collect_neighbors(node_hidden, in_indices)
            in_pre_hidden = torch.cat([in_pre_hidden, in_edge_reps], 3)
            in_pre_hidden = in_pre_hidden.mul(in_mask.unsqueeze(-1))
            # [batch_size, node_num, u_input_dim]
            in_pre_hidden = in_pre_hidden.sum(2)
            in_pre_hidden = in_pre_hidden.reshape([-1, self.node_dim + self.edge_dim])

            # out neighbor hidden
            # [batch_size, node_num, neighbors_size_max, node_dim + edge_dim]
            out_pre_hidden = self.collect_neighbors(node_hidden, out_indices)
            out_pre_hidden = torch.cat([out_pre_hidden, out_edge_reps], 3)
            out_pre_hidden = out_pre_hidden.mul(out_mask.unsqueeze(-1))
            # [batch_size, node_num, node_dim + edge_dim]
            out_pre_hidden = out_pre_hidden.sum(2)
            out_pre_hidden = out_pre_hidden.reshape([-1, self.node_dim + self.edge_dim])

            # in gate
            edge_ig = torch.sigmoid(self.W_ig_in(in_reps)
                                    + self.U_ig_in(in_pre_hidden)
                                    + self.W_ig_out(out_reps)
                                    + self.U_ig_out(out_pre_hidden))
            edge_ig = edge_ig.reshape([batch_size, node_num_max, self.hidden_dim])

            # forget gate
            edge_fg = torch.sigmoid(self.W_fg_in(in_reps)
                                    + self.U_fg_in(in_pre_hidden)
                                    + self.W_fg_out(out_reps)
                                    + self.U_fg_out(out_pre_hidden))
            edge_fg = edge_fg.reshape([batch_size, node_num_max, self.hidden_dim])

            # out gate
            edge_og = torch.sigmoid(self.W_og_in(in_reps)
                                    + self.U_og_in(in_pre_hidden)
                                    + self.W_og_out(out_reps)
                                    + self.U_og_out(out_pre_hidden))
            edge_og = edge_og.reshape([batch_size, node_num_max, self.hidden_dim])

            # input
            edge_cell_input = torch.tanh(self.W_cell_in(in_reps)
                                         + self.U_cell_in(in_pre_hidden)
                                         + self.W_cell_out(out_reps)
                                         + self.U_cell_out(out_pre_hidden))
            edge_cell_input = edge_cell_input.reshape([batch_size, node_num_max, self.hidden_dim])

            temp_cell = edge_fg * node_cell + edge_ig * edge_cell_input
            temp_hidden = edge_og * torch.tanh(temp_cell)

            node_cell = temp_cell.mul(mask.unsqueeze(-1))
            node_hidden = temp_hidden.mul(mask.unsqueeze(-1))

            graph_representations.append(node_hidden)

        # # shape: node_hidden, node_cell -> [batch, node_num, 256]
        # return graph_representations, node_hidden, node_cell


        batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index,
                      edge_index_negative]




        'add a readout function to capture graph-level reps, shape: [2]'
        readout = self.MLPreadout(node_hidden)



        'return node representation: shape [1, node_num, 256]'
        return node_hidden, batch_data, readout


    def collect_neighbors(self, node_reps, index):
        # node_rep: [batch_size, node_num, node_dim]
        # index: [batch_size, node_num, neighbors_num]
        batch_size = index.size(0)
        node_num = index.size(1)
        neighbor_num = index.size(2)
        rids = torch.arange(0, batch_size).to(self.args.device)  # [batch]
        rids = rids.reshape([-1, 1, 1])  # [batch, 1, 1]
        rids = rids.repeat(1, node_num, neighbor_num)  # [batch, nodes, neighbors]
        indices = torch.stack((rids, index), 3)  # [batch, nodes, neighbors, 2]
        return node_reps[indices[:, :, :, 0], indices[:, :, :, 1], :]

class GATConv(nn.Module):
    def __init__(self, args):
        super(GATConv, self).__init__()
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)

        'combine neighbours'
        self.neighbors = nn.Linear(self.edge_dim+self.node_dim,self.node_dim)

        'readout using MLP'
        self.MLPreadout = MLPreadout(self.node_dim)

        # 'aggregation using MLP'
        # self.MLPAggregation = MLPAggregation(self.node_dim)

        'objects for attention'
        self.nodeW = nn.Linear(self.edge_dim + self.node_dim, self.node_dim)
        self.attenW = nn.Parameter(torch.randn(self.node_dim, self.node_dim))


    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max # in_indices
        # edges shapes : batch_size, node_num, edge_labels # in_edges, out_edges

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative = batch_data
        node_reps = self.dropout(node_reps)

        # ==== input from in neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        in_edge_reps = self.edge_embedding(in_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        in_node_reps = self.collect_neighbors(node_reps, in_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        in_reps = torch.cat([in_node_reps, in_edge_reps], 3)

        in_reps = in_reps.mul(in_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        in_reps = in_reps.sum(dim=2)

        # ==== input from out neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        out_edge_reps = self.edge_embedding(out_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        out_node_reps = self.collect_neighbors(node_reps, out_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        out_reps = torch.cat([out_node_reps, out_edge_reps], 3)

        'attention starts'
        out_alpha = self.attention_on_neighbors(node_reps.squeeze(0), out_reps.squeeze(0))
        out_reps = torch.mul(out_reps, out_alpha.unsqueeze(0))
        'attention ends'

        out_reps = out_reps.mul(out_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        out_reps = out_reps.sum(2)

        out_nodes = self.neighbors(out_reps.squeeze(0))
        in_nodes = self.neighbors(in_reps.squeeze(0))




        'this works the best'
        node_hidden = node_reps + out_nodes + in_nodes
        # node_hidden = self.MLPAggregation(node_hidden).unsqueeze(0)

        # repack for the next layer
        batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative]



        'add a readout function to capture graph-level reps, shape: [2]'
        readout = self.MLPreadout(node_hidden)


        'return node representation: shape [1, node_num, 256]'
        return node_hidden, batch_data, readout


    def collect_neighbors(self, node_reps, index):
        # node_rep: [batch_size, node_num, node_dim]
        # index: [batch_size, node_num, neighbors_num]
        batch_size = index.size(0)
        node_num = index.size(1)
        neighbor_num = index.size(2)
        rids = torch.arange(0, batch_size).to(self.args.device)  # [batch]
        rids = rids.reshape([-1, 1, 1])  # [batch, 1, 1]
        rids = rids.repeat(1, node_num, neighbor_num)  # [batch, nodes, neighbors]
        indices = torch.stack((rids, index), 3)  # [batch, nodes, neighbors, 2]
        return node_reps[indices[:, :, :, 0], indices[:, :, :, 1], :]

    def attention_on_neighbors(self,node_reps, neighbors):
        '''Return alpha value for each neighbor'''
        alpha = 0

        soft = nn.Softmax(1)
        # [node_num,neighbor_num,256 dim]
        neighbors = self.nodeW(neighbors)
        # [node_num,neighbor_num,256 dim]
        attention = torch.matmul(neighbors,self.attenW)
        alpha = torch.bmm(attention,node_reps.unsqueeze(2))


        # alpha [node_num,neighbor_num, 1]
        return soft(alpha)

class GCNConvMultihop(nn.Module):
    def __init__(self, args):
        super(GCNConvMultihop, self).__init__()
        'only works without pre-training'
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)

        'combine neighbours'
        self.neighbors = nn.Linear(self.edge_dim+self.node_dim,self.node_dim)

        'readout using MLP'
        self.MLPreadout = MLPreadout(self.node_dim)

        # 'aggregation using MLP'
        # self.MLPAggregation = MLPAggregation(self.node_dim)
        'multihop W'
        # self.multiW = nn.Linear(self.node_dim,self.node_dim)
        # self.multilambda = torch.empty(1, requires_grad=True).to(self.args.device)
        # torch.nn.init.normal_(self.multilambda)

    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max # in_indices
        # edges shapes : batch_size, node_num, edge_labels # in_edges, out_edges

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative = batch_data
        # node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative, concept_class = batch_data
        node_reps = self.dropout(node_reps)

        'add multi-hop info starts'
        # n_nodes = concept_class.shape[0]
        # multi_adj = torch.zeros(n_nodes, n_nodes).to(self.args.device)
        # for id in torch.nonzero(concept_class).squeeze(1):
        #     multi_adj[id, :] = 1
        #     multi_adj[:, id] = 1
        # multi_hop = torch.mm(multi_adj, node_reps.squeeze(0))
        # node_reps = torch.sigmoid(self.multilambda) *self.multiW(multi_hop).unsqueeze(0) + (1- torch.sigmoid(self.multilambda)) * node_reps
        'add multi-hop info ends'


        # ==== input from in neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        in_edge_reps = self.edge_embedding(in_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        in_node_reps = self.collect_neighbors(node_reps, in_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        in_reps = torch.cat([in_node_reps, in_edge_reps], 3)

        in_reps = in_reps.mul(in_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        in_reps = in_reps.sum(dim=2)

        # ==== input from out neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        out_edge_reps = self.edge_embedding(out_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        out_node_reps = self.collect_neighbors(node_reps, out_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        out_reps = torch.cat([out_node_reps, out_edge_reps], 3)


        out_reps = out_reps.mul(out_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        out_reps = out_reps.sum(2)

        out_nodes = self.neighbors(out_reps.squeeze(0))
        in_nodes = self.neighbors(in_reps.squeeze(0))




        'this works the best'
        node_hidden = node_reps + out_nodes + in_nodes
        # node_hidden = self.MLPAggregation(node_hidden).unsqueeze(0)




        # repack for the next layer
        # batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative, concept_class]
        batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index,
                      edge_index_negative]



        'add a readout function to capture graph-level reps, shape: [2]'
        readout = self.MLPreadout(node_hidden)


        'return node representation: shape [1, node_num, 256]'
        return node_hidden, batch_data, readout


    def collect_neighbors(self, node_reps, index):
        # node_rep: [batch_size, node_num, node_dim]
        # index: [batch_size, node_num, neighbors_num]
        batch_size = index.size(0)
        node_num = index.size(1)
        neighbor_num = index.size(2)
        rids = torch.arange(0, batch_size).to(self.args.device)  # [batch]
        rids = rids.reshape([-1, 1, 1])  # [batch, 1, 1]
        rids = rids.repeat(1, node_num, neighbor_num)  # [batch, nodes, neighbors]
        indices = torch.stack((rids, index), 3)  # [batch, nodes, neighbors, 2]
        return node_reps[indices[:, :, :, 0], indices[:, :, :, 1], :]



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



class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE, self).__init__()

        self.args = args
        if args.vgae_type == 'gcn':
            'this is default'

            self.gc1 = GCNConv(args)
            self.gc2 = GCNConv(args)
            self.gc3 = GCNConv(args)
        elif args.vgae_type == 'grn':
            self.gc1 = GRNConv(args)
            self.gc2 = GRNConv(args)
            self.gc3 = GRNConv(args)
        else:
            'best starts'
            self.gc1 = GATConv(args)
            self.gc2 = GATConv(args)
            self.gc3 = GATConv(args)
            'best ends'



        self.dc = InnerProductDecoder(dropout=0.5, act=lambda x: x)

        self.device = args.device
        self.softmax = nn.Softmax()

        'add edge type prediction'
        self.edge_type_layer = nn.Linear(args.embed_dim * 2, args.edge_vocab_size)

    def encode(self, data):

        hidden, batch_data, readout = self.gc1(data)
        mu, _ , _= self.gc2(batch_data)
        logvar, _ , _= self.gc3(batch_data)
        '''hidden shape [1, num_node, 256]'''
        return hidden, mu, logvar, readout

        # mu, batch_data = self.gc1(data)
        # logvar, _ = self.gc2(data)

        # return mu,logvar



    def reparameterize(self, mu, logvar):
        if self.training:

            # import pdb;
            # pdb.set_trace()

            eps = torch.randn_like(logvar)
            std = torch.exp(torch.tanh(logvar))
            after = eps.mul(std).add_(torch.tanh(mu))


            return after

        else:
            return mu


    def distribution_loss(self, mu, logvar):
        # TODO: mu, logvar too large
        mu = torch.tanh(mu)
        logvar = torch.tanh(logvar)


        n_nodes = mu.shape[0]

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        return KLD

    def _normalize_matrix(self, x):
        'normalzied input x'
        'before move to gcn, we normalized x'
        'normalize into [-1,1]'
        min = torch.min(x)
        max = torch.max(x)
        x = (((x - min) / (max - min)) - 0.5) * 2

        return x


    def predict_edge_type(self,out_indices,out_edges):


        n_node = out_indices.shape[0]

        row = []
        col = []
        values = [] # edge type (ids)

        for i in range(n_node):
            for j, position in enumerate(out_indices[i]):
                if position != -1:
                    row.append(i)
                    col.append(position)
                    values.append(out_edges[i][j])


        edge_type_adjacency = torch.sparse_coo_tensor(indices=[row,col], values=values,size=[n_node,n_node]).to(self.device)



        return edge_type_adjacency.to_dense()


    def predict_edge(self, X, reconstruct_edge):

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_indices, negative_edges = X


        pred_neg = reconstruct_edge[negative_edges[0],negative_edges[1]]
        pred_pos = reconstruct_edge[edge_indices[0],edge_indices[1]]



        true_labels = torch.cat([torch.ones_like(negative_edges[0]), torch.zeros_like(negative_edges[0])]).float().to(self.device)
        sigmoid = nn.Sigmoid()
        pred_labels = sigmoid(torch.cat([pred_pos,pred_neg]))

        '''TODO: change to distmult xWx?'''
        # import pdb;pdb.set_trace()
        loss = nn.MSELoss(reduction='mean')


        if  pred_labels.shape == true_labels.shape:
            edge_loss = loss(pred_labels,true_labels)
            return edge_loss
        else:
            print (pred_labels.shape,true_labels.shape)
            return 0.


        # '''Optimize: add edge type prediction, make prediction on 105'''
        # edge_type_adjacency = self.predict_edge_type(out_indices[0], out_edges[0])
        # edge_type = edge_type_adjacency[edge_indices[0], edge_indices[1]]
        # 'get predicted edge embeddings, concate then map'
        # x = node_reps[0][edge_indices[0]]
        # y = node_reps[0][edge_indices[1]]
        #
        # predict_edge_embeddings = torch.cat([x,y],dim=1)
        # predict_edge_embeddings = self.edge_type_layer(predict_edge_embeddings)
        # predict_edge_embeddings = self.softmax(predict_edge_embeddings)
        # CEloss = nn.CrossEntropyLoss()
        # edge_type_loss = CEloss(predict_edge_embeddings, edge_type)
        #
        #
        # return edge_loss + edge_type_loss


    def predict_edge_coref(self, edge_labels, reconstruct_edge):

        'predict based on coref starts'

        pos_ids = edge_labels[0].detach().cpu().numpy()
        neg_ids = edge_labels[1].detach().cpu().numpy()
        pos = reconstruct_edge[pos_ids][:, pos_ids]
        pos = torch.sigmoid(pos).reshape(-1)
        neg = reconstruct_edge[neg_ids][:, neg_ids]
        neg = torch.sigmoid(neg).reshape(-1)

        true_labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).float().to(self.device)
        pred_labels = torch.cat([pos, neg])
        loss = nn.MSELoss(reduction='mean')

        edge_loss = loss(pred_labels, true_labels)


        return edge_loss

    def predict_if_coref(self, pred,truth):
        'this is to predict graph level coref'


        loss = nn.CrossEntropyLoss()

        if_coref_loss = loss(pred.unsqueeze(0), truth)

        return if_coref_loss

    def forward(self, X, pretrain=False):
        '''
        edge_labels is a list of 2 things: positive concept ids, and neg concept ids
        :param data:
        :param edge_labels:
        :return:
        '''


        hidden, mu, logvar, readout = self.encode(X)

        Z = self.reparameterize(mu.squeeze(0), logvar.squeeze(0)).unsqueeze(0)
        KLDloss = self.distribution_loss(mu.squeeze(0), logvar.squeeze(0))

        reconstruct_edge = self.dc(Z.squeeze(0))



        'Z shape: [1, node_num, 256]'


        if pretrain:

            '20210727'

            'predict based on coref starts'
            # if edge_labels is not None:
            #     edge_loss = self.predict_edge_coref(edge_labels, reconstruct_edge)
            'predict based on coref ends'

            'best starts'

            # import pdb;
            # pdb.set_trace()

            edge_loss = self.predict_edge(X, reconstruct_edge)
            return Z, edge_loss + KLDloss
            'best ends'


            # if_coref_loss = self.predict_if_coref(readout,graph_label)
            # return Z, edge_loss + KLDloss + if_coref_loss

        else:


            'best starts'
            edge_loss = self.predict_edge(X, reconstruct_edge)
            return hidden, KLDloss + edge_loss
            'best_ends'


            'works but not best'
            # return hidden, KLDloss
            'works but not best'

            'negative transfer'
            # return mu, KLDloss

            'very negative transfer'
            # return Z, KLDloss + edge_loss
