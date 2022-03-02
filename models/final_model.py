import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SE(torch.nn.Module):
    # implementation of squeeze and excitation layer
    def __init__(self, d_in, s_rate):
        super(SE, self).__init__()
        self.s = torch.nn.Linear(d_in, int(d_in * s_rate))
        self.e = torch.nn.Linear(int(d_in * s_rate), d_in)

    def reset_parameters(self):
        self.s.reset_parameters()
        self.e.reset_parameters()

    def forward(self, x):
        s = F.relu(self.s(x), inplace=True)
        w = torch.sigmoid(self.e(s))
        return x * w


class GCNBlock(torch.nn.Module):
    # implementation of 2 GCNConv layer with residual style and SElayer
    def __init__(self, d_in, d_out, use_se=True, s_rate=0.125):
        super(GCNBlock, self).__init__()
        self.conv1 = GCNConv(d_in, d_out, improved=True)
        self.conv2 = GCNConv(d_out, d_out, improved=True)
        self.norm1 = torch.nn.LayerNorm(d_out)
        self.norm2 = torch.nn.LayerNorm(d_out)
        self.use_se = use_se
        if use_se:
            self.se = SE(d_out, s_rate)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        if self.use_se:
            self.se.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        skip = x
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.norm2(self.conv2(x, edge_index))
        if self.use_se:
            x = self.se(x)
        x = F.relu(x + skip, inplace=True)
        return x


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p, n_blocks):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, improved=True)
        self.norm = torch.nn.LayerNorm(d_out)
        self.blocks = torch.nn.ModuleList([GCNBlock(d_out, d_out) for _ in range(n_blocks)])
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()
        for l in self.blocks:
            l.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.norm(self.conv(x, edge_index)))

        for l in self.blocks:
            x = l(x, edge_index)

        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, improved=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class Net(torch.nn.Module):
    def __init__(self, dataset, hidden=32, blocks=5, dropout=0.):
        super(Net, self).__init__()
        self.crd = CRD(dataset.num_features, hidden, dropout, blocks)
        self.cls = CLS(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data, mask=None):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x
