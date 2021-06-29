import hdf5storage
import numpy as np

import torch
import torch.multiprocessing
from torch.utils.data import TensorDataset
from self_time.optim.pretrain import get_transform

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.misc import load_pickle


def preproc_input(x, norm_per_signal, condition=None):
    """Preprocess training or test data, filter data by condition"""
    if condition is not None:
        x = x[condition]

    x = np.array([norm01_rf(d, per_signal=norm_per_signal) for d in x])
    return x


def norm01_rf(x, per_signal=True):
    """Normalize RF signal magnitude to [0 1] (per signal or per core)"""
    ax = 1 if per_signal else (0, 1)
    mi = x.min(axis=ax)
    ma = x.max(axis=ax)
    rfnorm = (x - mi) / (ma - mi)
    return rfnorm


def to_categorical(y):
    n_classes = np.max(y) + 1
    y_c = np.zeros((len(y), np.int(n_classes)))
    for i in range(len(y)):
        y_c[i, np.int(y[i])] = 1
    return y_c


def create_datasets_cores(ftrs_train, inv_train, corelen_train, ftrs_val, inv_val, corelen_val):
    counter = 0
    signal_train = []
    for i in range(len(corelen_train)):
        temp = ftrs_train[counter:(counter + corelen_train[i])]
        signal_train.append(temp)
        counter += corelen_train[i]

    counter = 0
    signal_val = []
    for i in range(len(corelen_val)):
        temp = ftrs_val[counter:(counter + corelen_val[i])]
        signal_val.append(temp)
        counter += corelen_val[i]

    label_train = to_categorical(inv_train > 0)
    label_val = to_categorical(inv_val > 0)
    return signal_train, label_train, inv_train, signal_val, label_val, inv_val


class DatasetV1(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    from typing import Union

    def __init__(self, data, label, location,
                 aug_type: Union[list, str] = ('G2',), n_views=1):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve

        self.data = torch.tensor(data, dtype=torch.float32) if aug_type == 'none' else data
        self.label = torch.tensor(label, dtype=torch.uint8)
        self.location = torch.tensor(location, dtype=torch.uint8)
        self.n_views = n_views
        self.aug_type = aug_type
        self.transform, self.to_tensor_transform = get_transform(
            aug_type if isinstance(aug_type, list) else [aug_type, ]
        )
        # self.transform = (
        #         TimeWarp() * 1  # random time warping 5 times in parallel
        #         # + Crop(size=170)  # random crop subsequences with length 300
        #         + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        #         + Reverse() @ 0.5  # with 50% probability, reverse the sequence
        #         + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        #         + Convolve(window='flattop', size=11, prob=.25)
        # )

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        z = self.location[index]

        if self.aug_type == 'none':
            return x, y, z, index

        img_list = list()
        label_list = list()
        loc_list = list()
        idx_list = list()

        if self.transform is not None:
            # img_transformed = self.transform.augment(x)
            # img_list.append(self.to_tensor_transform(img_transformed))  # .unsqueeze(0)
            for _ in range(self.n_views):
                img_transformed = self.transform(x[..., np.newaxis]).squeeze()
                img_list.append(self.to_tensor_transform(img_transformed))  # .unsqueeze(0)
                loc_list.append(z)
                label_list.append(y)
                idx_list.append(index)
        return img_list, label_list, loc_list, idx_list

    def __len__(self):
        return self.label.size(0)

    def __updateitem__(self, newones):
        'update labels'
        # Select sample
        # to_categorical(newones)
        newones.shape
        self.label = torch.tensor(newones, dtype=torch.uint8)

    def __updateinv__(self, newones):
        'update labels'
        # Select sample
        self.siginv = torch.tensor(newones, dtype=torch.uint8)


def create_datasets_v1(data_file, norm=None, min_inv=0.4, augno=0, inv_state='none', aug_type='none', n_views=4):
    input_data = load_pickle(data_file)
    data_train = input_data["data_train"]
    label_train = input_data["label_train"]
    inv_train = input_data["inv_train"]
    CoreN_train = input_data["corename_train"].astype(np.float)

    included_idx = [True for lid in label_train]
    included_idx = [False if inv < min_inv and inv > 0 else tr_idx for inv, tr_idx in zip(inv_train, included_idx)]

    corelen_train = []

    # Working with BK dataset
    corecounter = 0
    bags_train = []
    target_train = []
    name_train = []
    siginv_train = []
    coreno_train = []
    for i in range(len(data_train)):
        if included_idx[i]:
            bags_train.append(data_train[i])
            target_train.append(np.repeat(label_train[i], data_train[i].shape[0]))
            temp = np.tile(CoreN_train[i], data_train[i].shape[0])
            name_train.append(temp.reshape((data_train[i].shape[0], 8)))
            corelen_train.append(data_train[i].shape[0])
            if inv_train[i] == 0:
                siginv_train.append(np.repeat(inv_train[i], data_train[i].shape[0]))
            else:
                siginv_train.append(np.repeat(inv_train[i], data_train[i].shape[0]))

            coreno_train.append(np.repeat(corecounter, data_train[i].shape[0]))
            corecounter += 1

    signal_train = np.concatenate(bags_train).astype('float32')
    signal_train, train_stats = preprocess(signal_train)
    target_train = np.concatenate(target_train)
    name_train = np.concatenate(name_train)
    label_train = to_categorical(target_train)
    trn_ds = DatasetV1(signal_train, label_train, name_train,
                       aug_type=aug_type, n_views=n_views)  # ['magnitude_warp', 'time_warp'])

    # for s in ['train', 'val', 'test']:
    #     input_data[f"data_{s}"] = np.concatenate(input_data[f"data_{s}"])

    train_set = create_datasets_test(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                     train_stats=train_stats)
    val_set = create_datasets_test(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                   train_stats=train_stats)
    test_set = create_datasets_test(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                    train_stats=train_stats)

    return trn_ds, inv_train[included_idx], corelen_train, train_set, val_set, test_set


def create_datasets_test(data_file, state, norm='Min_Max', min_inv=0.4, augno=0, inv_state='normal', input_data=None,
                         return_array=False, train_stats=None):
    if input_data is None:
        input_data = load_pickle(data_file)

    data_test = input_data["data_" + state]  # [0]
    roi_coors_test = input_data['roi_coors_' + state]
    label_test = input_data["label_" + state]  # [0]
    label_inv = input_data["inv_" + state]  # [0]
    CoreN_test = input_data["corename_" + state]
    patient_id_bk = input_data["PatientId_" + state]  # [0]
    involvement_bk = input_data["inv_" + state]  # [0]
    gs_bk = np.array(input_data["GS_" + state])  # [0]

    included_idx = [True for _ in label_test]
    included_idx = [False if (inv < min_inv) and (inv > 0) else tr_idx for inv, tr_idx in zip(label_inv, included_idx)]

    corelen = []
    target_test = []
    name_tst = []

    # Working with BK dataset
    bags_test = []
    sig_inv_test = []
    for i in range(len(data_test)):
        if included_idx[i]:
            bags_test.append(data_test[i])
            target_test.append(np.repeat(label_test[i], data_test[i].shape[0]))
            # temp=np.repeat(onehot_corename(CoreN_test[i]),data_test[i].shape[0])
            temp = np.tile(CoreN_test[i], data_test[i].shape[0])
            name_tst.append(temp.reshape((data_test[i].shape[0], 8)))
            corelen.append(data_test[i].shape[0])
            if label_inv[i] == 0:
                sig_inv_test.append(np.repeat(label_inv[i], data_test[i].shape[0]))
            elif inv_state == 'uniform':
                dist = abs(label_inv[i] - np.array([0.25, 0.5, 0.75, 1]))
                min_ind = np.argmin(dist)
                temp_inv = abs(torch.rand(data_test[i].shape[0]) * 0.25) + 0.25 * min_ind
                sig_inv_test.append(temp_inv)
            elif inv_state == 'normal':
                temp_inv = torch.randn(data_test[i].shape[0]) * np.sqrt(0.1) + label_inv[i]
                sig_inv_test.append(np.abs(temp_inv))
            else:
                sig_inv_test.append(np.repeat(label_inv[i], data_test[i].shape[0]))

    signal_test = np.concatenate(bags_test)
    target_test = np.concatenate(target_test)
    sig_inv_test = np.concatenate(sig_inv_test)
    name_tst = np.concatenate(name_tst)

    roi_coors_test = [roi_coors_test[i] for i in range(len(roi_coors_test)) if included_idx[i] == 1]
    patient_id_bk = patient_id_bk[included_idx]
    gs_bk = gs_bk[included_idx]
    label_inv = label_inv[included_idx]

    if train_stats:
        x_train_min, x_train_max = train_stats
        signal_test = 2. * (signal_test - x_train_min) / (x_train_max - x_train_min) - 1.
    #
    # signal_test = (signal_test - signal_test.mean(axis=1)[..., np.newaxis]) / signal_test.std(axis=1)[
    #     ..., np.newaxis]

    if return_array:
        return signal_test, corelen, label_inv, patient_id_bk, gs_bk

    label_test = to_categorical(target_test)

    tst_ds = TensorDataset(torch.tensor(signal_test, dtype=torch.float32).unsqueeze(1),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           torch.tensor(sig_inv_test, dtype=torch.float32).unsqueeze(1))
    return tst_ds, corelen, label_inv, patient_id_bk, gs_bk, roi_coors_test


def preprocess(x_train):
    # Normalize
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    # Test is secret
    # x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
    # x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
    return x_train, (x_train_min, x_train_max)
