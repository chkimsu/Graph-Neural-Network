import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from custom_datasets import CustomDataset



def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for mask, key in zip([data.train_mask, data.val_mask, data.test_mask], ['train', 'val', 'test']):
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        precision_weight = precision_score(dataset.y[mask], logits[mask].max(1)[1], average='weighted')
        recall_weight = recall_score(dataset.y[mask], logits[mask].max(1)[1], average='weighted')
        f1_score_weight = f1_score(dataset.y[mask], logits[mask].max(1)[1], average='weighted')


        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc
        outs['{} precision_weight'.format(key)] = precision_weight
        outs['{} recall_weight'.format(key)] = recall_weight
        outs['{} f1_score_weight'.format(key)] = f1_score_weight

    return outs


if __name__ == '__main__':
    dataset = CustomDataset()
    model = torch.load('save/model_fin.pth')
    outs = evaluate(model, dataset)
    print(outs)

