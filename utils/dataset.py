from collections import namedtuple
from utils.misc import load_pkl

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RFDataset(Dataset):
    """Prostate RF dataset"""

    def __init__(self, patients, transform=None):
        super(RFDataset, self).__init__()
        self.patients = patients
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = {'data': self.patients[idx].rf, 'label': self.patients[idx].cancer}
        if self.transform:
            item = self.transform(item)

        return item


def make_stationary(x: np.ndarray):
    """
    x: shape = (Num, Time-point)
    """
    x -= x.min(axis=1)[..., np.newaxis]
    x[:, 1:] -= x[:, :-1]
    return x[:, 1:]


def normalize(x):
    # return x / x.max()
    return (x - x.mean()) / x.std()


class RandomSelect(object):
    """Randomly select subset of RF data"""

    def __init__(self, n_rows=32, n_cols=150, n_separate_rows=1):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_separate_rows = n_separate_rows

    def __call__(self, item: namedtuple, *args, **kwargs):
        rf, label = item['data'], item['label']
        if self.n_rows is None:
            self.n_rows = rf.shape[0]
        # idx_row = 0
        idx_col = 20  # np.random.randint(rf.shape[1] - self.n_cols)
        # idx_col = 0
        if self.n_separate_rows is None:
            idx_separate_row = np.arange(0, len(rf) - self.n_rows, self.n_rows)
        else:
            idx_separate_row = np.random.choice(rf.shape[0] - self.n_rows, size=self.n_separate_rows, replace=False)

        # idx_separate_row = [0,]
        # idx_col = 0
        item['data'] = np.zeros((len(idx_separate_row), self.n_rows, self.n_cols))
        item['label'] = np.zeros(len(idx_separate_row))
        # for i, idx_row in enumerate(idx_separate_row):
        #     x = rf[idx_row: idx_row + self.n_rows, idx_col:idx_col + self.n_cols]
            # x = (x - x.mean(axis=0)) / x.std(axis=0)
            # x -= x.min(axis=0)
            # x[1:] -= x[:-1]
            # x = x[1:]
        idx_row = (torch.randperm(rf.shape[0]))[:self.n_rows].numpy()
        rf = normalize(rf)
        item['data'] = rf[idx_row, idx_col:idx_col + self.n_cols]
        # item['data'] = make_stationary(make_stationary(item['data']))
        item['label'] = np.ones(len(idx_row)) * label
        return item


class ToTensor(object):
    """Convert ndarrays in item to Tensors"""

    def __call__(self, item, *args, **kwargs):
        item['data'] = torch.from_numpy(item['data'])
        return item


def main():
    data = load_pkl()
    composed = transforms.Compose([RandomSelect(), ToTensor()])
    rf_train = RFDataset(data['train'], transform=composed)
    data_loader = DataLoader(rf_train, batch_size=32, shuffle=True, num_workers=0)
    for i, batch in enumerate(data_loader):
        data, label = batch['data'], batch['label']
        data, label = data.view(-1, data.size(-1)), label.view(-1)
        print(data, label)
        exit()


if __name__ == '__main__':
    main()
