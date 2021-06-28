import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from loss_functions.isomax import IsoMaxLossSecondPart
from .misc import f_score


from utils.get_loss_function import get_loss_function


# Loss function for Co-Teaching
def loss_coteaching(
        y_1,
        y_2,
        t,
        forget_rate,
        class_weights=None,
        num_classes=2,
        **kwargs
):
    """Co-Teaching Loss function.

    Parameters
    ----------
    y_1 : Tensor array
      Output logits from model 1

    y_2 : Tensor array
      Output logits from model 2

    t : np.array
      List of Noisy Labels (t means targets)

    forget_rate : float
      Decimal between 0 and 1 for how quickly the models forget what they learn.
      Just use rate_schedule[epoch] for this value

    class_weights : Tensor array, shape (Number of classes x 1), Default: None
      A np.torch.tensor list of length number of classes with weights
    relax: None, 0 or 1 (binary)
    """
    loss_func = kwargs['loss_func'] if 'loss_func' in kwargs.keys() else [F.cross_entropy, F.cross_entropy]
    loss_func1, loss_func2 = loss_func

    # loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_1 = IsoMaxLossSecondPart()(y_1, t, reduction='none')
    # loss_1 = f_score(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    # loss_2 = F.cross_entropy(y_2, t, reduction='none')
    loss_2 = IsoMaxLossSecondPart()(y_2, t, reduction='none')
    # loss_2 = f_score(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu())

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # ind_12 = torch.unique((torch.cat([ind_1_sorted[num_remember:], ind_2_sorted[num_remember:]])))
    # Share updates between the two models.
    # TODO: these class weights should take into account the ind_mask filters.

    # Compute class weights to counter class imbalance in selected samples
    # class_weights_1 = estimate_class_weights(t[ind_1_update], num_class=num_classes)
    # class_weights_2 = estimate_class_weights(t[ind_2_update], num_class=num_classes)

    # Equalizing the number of instances in every class
    ind_1_update = balance_clipping(t, ind_1_update, num_classes)
    ind_2_update = balance_clipping(t, ind_2_update, num_classes)

    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduction='none')
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduction='none')

    loss_1_update = loss_func1(
        y_1[ind_2_update], t[ind_2_update], reduction='none',
        weight=None,
        # weight=class_weights_2,
        sub_index=ind_2_update, **kwargs)
    loss_2_update = loss_func2(
        y_2[ind_1_update], t[ind_1_update], reduction='none',
        weight=None,
        # weight=class_weights_1,
        sub_index=ind_1_update, **kwargs)

    return (
        torch.sum(loss_1_update) / len(loss_1_update),
        torch.sum(loss_2_update) / len(loss_2_update),
        get_chosen_index(t, ind_1_update, ind_2_update),
        # ind_12
    )


def loss_coteaching_plus(logits, logits2, labels, forget_rate, class_weights=None, **kwargs):
    step = kwargs['step']
    loss_func = kwargs['loss_func'] if 'loss_func' in kwargs.keys() else [F.cross_entropy, F.cross_entropy]
    loss_func1, loss_func2 = loss_func

    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = logical_disagree_id.astype(np.int64)
    if 'index' in kwargs.keys():
        temp_disagree *= kwargs['index'].cpu().numpy()
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]
        if 'index' in kwargs.keys():
            kwargs['index'] = kwargs['index'][disagree_id]

        loss_1, loss_2, chosen_ind = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate,
                                                     **kwargs)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        loss_1_update = loss_func1(
            update_outputs, update_labels, reduction='none', weight=class_weights, **kwargs)
        loss_2_update = loss_func2(
            update_outputs2, update_labels, reduction='none', weight=class_weights, **kwargs)

        loss_1 = torch.sum(update_step * loss_1_update) / labels.size()[0]
        loss_2 = torch.sum(update_step * loss_2_update) / labels.size()[0]
        chosen_ind = get_chosen_index(labels, np.argsort(loss_1.data.cpu()), np.argsort(loss_2.data.cpu()))

    return loss_1, loss_2, chosen_ind


def get_loss_coteaching(loss_name, num_classes, use_plus=False, relax=False, **kwargs_):
    loss_func = [get_loss_function(loss_name, num_classes, **kwargs_) for _ in range(2)]
    cot_func = loss_coteaching_plus if use_plus else loss_coteaching

    def wrapper(*args, **kwargs):
        return cot_func(*args, **kwargs, loss_func=loss_func, relax=relax, num_classes=num_classes)

    return wrapper


def get_chosen_index(target, ind_1, ind_2):
    ind = {
        'benign': [np.argwhere(target[ind_1].cpu() == 0)[0], np.argwhere(target[ind_2].cpu() == 0)[0]],
        'cancer': [np.argwhere(target[ind_1].cpu() == 1)[0], np.argwhere(target[ind_2].cpu() == 1)[0]],
        'cancer_ratio': [target[ind_1].sum().cpu().item() / len(ind_1),
                         target[ind_2].sum().cpu().item() / len(ind_2)]
    }
    return ind


def balance_clipping(label, index, num_classes=2):
    """
    :param label:
    :param index:
    :param num_classes:
    :return:
    """

    min_num = torch.histc(label[index], num_classes).min().item()
    index_b = []
    for k in range(num_classes):
        index_b.append(index[label[index] == k][:min_num])
    return torch.cat(index_b)


def estimate_class_weights(label, num_class: int):
    """

    :param label: 1D array
    :param num_class: int
    :return:
    """
    freq_inv = 1 / (torch.histc(label, num_class) / len(label))
    class_weights = (freq_inv / freq_inv.sum()) / (1 / num_class)
    return class_weights
