from copy import copy
import torch
from torch.nn import functional as F
from .misc import f_score
from loss_functions.isomax import IsoMaxLossSecondPart


class ELR(torch.nn.Module):
    def __init__(self, num_samples, num_classes=10, alpha=3, beta=0.7):
        r"""Early Learning Regularization.
        Parameters
        * `num_samples` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controlling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super(ELR, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_samples, self.num_classes).cuda() \
            if self.USE_CUDA else torch.zeros(num_samples, self.num_classes)
        self.beta = beta
        self.alpha = alpha

    def forward(self, output, label, index,  sub_index=None, weight=None, **kwargs):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples
         in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """
        if sub_index is not None:
            index = copy(index[sub_index])
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        # ce_loss = F.cross_entropy(output, label, reduction='none', weight=weight)
        ce_loss = IsoMaxLossSecondPart()(output, label, reduction='none')
        # ce_loss = f_score(output, label, reduction='none')
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log())
        final_loss = ce_loss + self.alpha * elr_reg

        if 'reduction' in kwargs.keys():
            if kwargs['reduction'] == 'none':
                return final_loss
        return final_loss.mean()
