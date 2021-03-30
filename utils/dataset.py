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


class RandomSelect(object):
    """Randomly select subset of RF data"""

    def __init__(self, n_rows=200, n_cols=100):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, item, *args, **kwargs):
        rf, label = item['data'], item['label']
        if self.n_rows is None:
            self.n_rows = rf.shape[0]
        idx_row = np.random.choice(rf.shape[0], self.n_rows, replace=False)
        idx_col = np.random.randint(rf.shape[1] - self.n_cols)

        item['data'] = rf[idx_row, idx_col:idx_col + self.n_cols]
        item['label'] = np.ones(self.n_rows) * label
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
    data_loader = DataLoader(rf_train, batch_size=4, shuffle=True, num_workers=0)
    for i, batch in enumerate(data_loader):
        data, label = batch['data'], batch['label']
        data, label = data.view(-1, data.size(-1)), label.view(-1)
        print(data, label)
        exit()


if __name__ == '__main__':
    main()
