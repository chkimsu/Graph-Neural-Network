from __future__ import division

import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils as ut
import psgd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path_runs = "runs"

def run(
    dataset, 
    model, 
    str_optimizer, 
    runs,
    epochs, 
    lr,
    lr_decay,
    weight_decay, 
    early_stopping,  
    logger, 
    gamma,
    model_path
    ):
    if logger is not None:
        path_logger = os.path.join(path_runs, logger)
        print(f"path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = SummaryWriter(log_dir=os.path.join(path_runs, logger)) if logger is not None else None

    val_losses, accs, durations = [], [], []
    torch.manual_seed(42)
    for i_run in range(runs):
        # data = dataset[0]
        # data = data.to(device)
        data = dataset

        model.to(device).reset_parameters()
        preconditioner = None

        scheduler = None
        if str_optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            if lr_decay is not None:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            lam = (float(epoch)/float(epochs))**gamma if gamma is not None else 0.
            train(model, optimizer, data, preconditioner, lam)
            if scheduler is not None:
                scheduler.step()
            eval_info = evaluate(model, data)
            eval_info['epoch'] = int(epoch)
            eval_info['run'] = int(i_run+1)
            eval_info['time'] = time.perf_counter() - t_start

            if gamma is not None:
                eval_info['gamma'] = gamma
            
            if logger is not None:
                for k, v in eval_info.items():
                    logger.add_scalar(k, v, global_step=epoch)
                
                    
            if eval_info['val loss'] < best_val_loss:
                best_val_loss = eval_info['val loss']
                test_acc = eval_info['test acc']
                val_acc = eval_info['val acc']
                if model_path is not None:
                    torch.save(model, model_path)

            val_loss_history.append(eval_info['val loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val loss'] > tmp.mean().item():
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        # accs.append(test_acc)
        accs.append(val_acc)
        durations.append(t_end - t_start)
    
    if logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('Val Loss: {:.4f}, valid Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100*acc.mean().item(),
                 100*acc.std().item(),
                 duration.mean().item()))

def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False

    # print(len(label[data.train_mask]))
    # _, counts = np.unique(label[data.train_mask].cpu().numpy(), return_counts=True)
    # w = torch.zeros(40, dtype=torch.float32).cuda()
    # for i in range(40):
    #     w[i] = 1/np.sqrt(counts[i])
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    
    loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step(lam=lam)
    optimizer.step()

def evaluate(model, data):
    model.eval()
    # print(data.edge_index, data.edge_index.shape)
    # print(data.x, data.x.shape)
    # print(data.y, data.y.shape)
    # print(data.train_mask)
    # print(data.test_mask)

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for mask, key in zip([data.train_mask, data.val_mask, data.test_mask], ['train', 'val', 'test']):
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs
