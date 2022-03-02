import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, improved=True)
        self.norm = torch.nn.LayerNorm(d_out)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.norm(self.conv(x, edge_index)), inplace=True)
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
    def __init__(self, dataset, hidden=32, dropout=0.):
        super(Net, self).__init__()
        self.crd = CRD(dataset.num_features, hidden, dropout)
        self.cls = CLS(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x