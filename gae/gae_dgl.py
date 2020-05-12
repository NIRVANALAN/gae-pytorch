import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GVAE(nn.Module):
    def __init__(self, g, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0.5):
        super(GVAE, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        self.gc1 = GraphConv(input_feat_dim, hidden_dim1, activation=F.relu, bias=False)
        self.gc2 = GraphConv(hidden_dim1, hidden_dim2, bias=False)
        self.gc3 = GraphConv(hidden_dim1, hidden_dim2, bias=False)
        # self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x):
        hidden1 = self.gc1(self.g, self.dropout(x))
        return self.gc2(self.g, self.dropout(hidden1)), self.gc3(self.g, self.dropout(hidden1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


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
