import torch
from torch.nn import functional as F


def f_score(logit, label, threshold=0.5, beta=2, reduction='none', **kwargs):
    label = F.one_hot(label)
    prob = torch.sigmoid(logit)
    prob = prob > threshold
    label = label > threshold

    tp = (prob & label).sum(1).float()
    tn = ((~prob) & (~label)).sum(1).float()
    fp = (prob & (~label)).sum(1).float()
    fn = ((~prob) & label).sum(1).float()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f2 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-12)
    if reduction == 'none':
        return f2
    return f2.mean()
