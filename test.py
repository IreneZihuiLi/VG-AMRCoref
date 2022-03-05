
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgae import GCNConv,InnerProductDecoder

class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE, self).__init__()
        self.gc1 = GCNConv(args)
        self.gc2 = GCNConv(args)
        self.gc3 =  GCNConv(args)
        self.dc = InnerProductDecoder(dropout=0.5, act=lambda x: x)


        self.device = args.device

    def encode(self, data):
        hidden1 = self.gc1(data)
        import pdb;pdb.set_trace()
        return self.gc2(hidden1), self.gc3(hidden1)

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

    def forward(self, X):
        Z = self.encode(X)
        A_pred = None
        return Z