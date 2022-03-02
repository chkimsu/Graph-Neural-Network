import os.path as osp

import numpy as np
import torch
import pandas as pd
import torch_geometric.datasets
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class CustomDataset():
    def __init__(self):
        x_train = pd.read_csv('data/node_feat_train.csv', index_col=0).to_numpy()
        x_val = pd.read_csv('data/node_feat_valid.csv', index_col=0).to_numpy()
        x_test = pd.read_csv('data/node_feat_test.csv', index_col=0).to_numpy()
        train_mask = [True for _ in range(x_train.shape[0])] + [False for _ in range(x_val.shape[0]+x_test.shape[0])]
        self.train_mask = np.asarray(train_mask)
        val_mask = [False for _ in range(x_train.shape[0])] + [True for _ in range(x_val.shape[0])] + [False for _ in range(x_test.shape[0])]
        self.val_mask = np.asarray(val_mask)
        test_mask = [False for _ in range(x_train.shape[0]+x_val.shape[0])] + [True for _ in range(x_test.shape[0])]
        self.test_mask = np.asarray(test_mask)
        self.x = np.concatenate([x_train, x_val, x_test], axis=0)
        for i in range(self.x.shape[1]):
            mean = np.mean(self.x[:, i])
            var = np.std(self.x[:, i])
            self.x[:, i] = (self.x[:, i]-mean)/var
        # for i in range(self.x.shape[1]):
        #     min_v = np.min(self.x[:, i])
        #     max_v = np.max(self.x[:, i])
        #     self.x[:, i] = (self.x[:, i]-min_v)/(max_v-min_v)
        self.x = torch.tensor(self.x, dtype=torch.float32)

        self.edge_index = torch.tensor(pd.read_csv('data/edge.csv').to_numpy().transpose([1, 0]), dtype=torch.long)

        y_train = pd.read_csv('data/node_label_train.csv', index_col=0).to_numpy()
        y_val = pd.read_csv('data/node_label_valid.csv', index_col=0).to_numpy()
        y_test = pd.read_csv('data/node_label_test.csv', index_col=0).to_numpy()
        self.y = torch.tensor(np.concatenate([y_train, y_val, y_test], axis=0).squeeze(), dtype=torch.long)

        self.num_features = self.x.shape[1]
        self.num_classes = self.y.max().cpu().numpy()+1


if __name__ == '__main__':
    CustomDataset()
