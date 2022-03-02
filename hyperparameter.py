import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from datasets import get_planetoid_dataset
from custom_datasets import CustomDataset
from train_eval import run
from models.final_model import Net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=4e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--logger', type=str, default='wd4e-4_hidden32_lr4e-2_0block_lrscheduler0.9995_srate125')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--gamma', type=float, default=None)
    args = parser.parse_args()

    dataset = CustomDataset()

    kwargs = {
        'dataset': dataset,
        'model': Net(dataset, args.hidden, args.blocks, args.dropout),
        'str_optimizer': args.optimizer,
        'runs': args.runs,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping': args.early_stopping,
        'logger': args.logger,
        'gamma': args.gamma,
    }

    #   searching hyperparameters for final model
    for lr_decay in [0.995, 0.999, 0.9995, 0.9999]:
        for hidden in [16, 32, 64]:
            for blocks in [5, 7]:
                for lr in [4e-2, 1e-2, 4e-3]:
                    kwargs['lr_decay'] = lr_decay
                    kwargs['lr'] = lr
                    log = f'wd{args.weight_decay}_lr{lr}_decay{lr_decay}_hidden{hidden}_{blocks}blocks'
                    kwargs['logger'] = log

                    kwargs['model']: Net(dataset, hidden, blocks)
                    run(**kwargs)


