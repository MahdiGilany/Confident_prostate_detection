import torch
import numpy as np
from torch.utils.data import DataLoader
from .dataset import DatasetV1, to_categorical
from tslearn.clustering import TimeSeriesKMeans
from torch.utils.data import DataLoader


def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses
    for l in dataset.label:
        count[l[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(dataset)
    for idx, l in enumerate(dataset.label):
        weight[idx] = weight_per_class[l[1]]
    return weight


def create_loaders_test(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    tst_ds = data  # , tst_ds
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return tst_dl


def create_loader(dataset, bs=128, jobs=0, add_sampler=False, shuffle=False):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    # For unbalanced dataset we create a weighted sampler
    # sampler = ImbalancedDatasetSampler(dataset)
    sampler = None
    if add_sampler:
        weights = make_weights_for_balanced_classes(dataset, 2)
        weights = torch.Tensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=jobs,
                            pin_memory=True)
    return dataloader
