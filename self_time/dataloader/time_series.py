import numpy as np
import torch.utils.data as data


class TsLoader(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img_transformed = self.transform(img.copy())
        else:
            img_transformed = img

        return img_transformed, target

    def __len__(self):
        return self.data.shape[0]

