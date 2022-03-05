import torch
import torch.nn as nn
import argparse
from config import parse_config



class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)

        # combine neighbours
        self.neighbors = nn.Linear(self.edge_dim+self.node_dim,self.node_dim)


    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max # in_indices
        # edges shapes : batch_size, node_num, edge_labels # in_edges, out_edges

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, _ = batch_data[:-1]
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

        # TODO: GAT on neighbors?
        node_hidden = node_reps + out_nodes + in_nodes


        return node_hidden


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

