import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from config import parse_config



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

        # combine neighbours
        self.neighbors = nn.Linear(self.edge_dim+self.node_dim,self.node_dim)




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


        # import pdb;pdb.set_trace()
        node_hidden = node_reps + out_nodes + in_nodes

        # repack for the next layer
        batch_data = [node_hidden, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, edge_index, edge_index_negative]

        'return node representation: shape [1, node_num, 256]'
        return node_hidden, batch_data


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
        self.gc1 = GCNConv(args)
        self.gc2 = GCNConv(args)
        self.gc3 = GCNConv(args)

        self.dc = InnerProductDecoder(dropout=0.5, act=lambda x: x)

        self.device = args.device
        self.softmax = nn.Softmax()

        'add edge type prediction'
        self.edge_type_layer = nn.Linear(args.embed_dim * 2, args.edge_vocab_size)

    def encode(self, data):
        hidden, batch_data = self.gc1(data)
        mu, _ = self.gc2(batch_data)
        logvar, _ = self.gc3(batch_data)
        '''hidden shape [1, num_node, 256]'''
        return hidden, mu, logvar

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

        edge_loss = loss(pred_labels,true_labels)

        return edge_loss


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

    def forward(self, X, pretrain=False):


        hidden, mu, logvar = self.encode(X)


        Z = self.reparameterize(mu.squeeze(0), logvar.squeeze(0)).unsqueeze(0)
        KLDloss = self.distribution_loss(mu.squeeze(0), logvar.squeeze(0))

        reconstruct_edge = self.dc(Z.squeeze(0))


        'Z shape: [1, node_num, 256]'
        if pretrain:

            edge_loss = self.predict_edge(X, reconstruct_edge)

            'this is good'
            return Z, edge_loss + KLDloss

        else:

            'this is good'
            return hidden, KLDloss

            'negative transfer'
            # return mu, KLDloss

            'very negative transfer'
            # return Z, KLDloss
