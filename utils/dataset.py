import os
from typing import Union

import torch
import numpy as np
import sklearn.utils as sk
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from self_time.optim.pretrain import get_transform

torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils.misc import load_matlab
from utils.misc import load_pickle
from utils.misc import squeeze_Exact
from preprocessing.s02b_create_unsupervised_dataset import load_datasets as load_unlabelled_datasets


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

    def __init__(self, data, label, location, inv, core_id, patient_id, frame_id, core_len, roi_coors, stn,
                 aug_type: Union[list, str] = ('G2',), initial_min_inv=.7, n_neighbor=0,
                 n_views=1, transform_prob=.2, degree=1, aug_lib='self_time', n_duplicates=1,
                 stn_alpha=1e-2):
        """"""
        self.data_dict = {
            'data': data,
            'label': label,
            'location': location,
            'inv': inv,
            'core_id': core_id,
            'patient_id': patient_id,
            'frame_id': frame_id,
            # 'roi_coors': roi_coors,
            # 'stn': stn,  # stationery
        }
        self.data, self.label, self.location, self.inv, self.core_id, self.patient_id, self.inv_pred, \
        self.frame_id, self.roi_coors, self.stn, = [None, ] * 10
        self.stn_alpha = stn_alpha
        self.label_corrected = False
        self.last_updated_label = None
        self.transformer = None
        self.core_len = core_len
        self.n_views, self.aug_type, self.aug_lib, self.n_duplicates = n_views, aug_type, aug_lib, n_duplicates
        self.initial_min_inv = initial_min_inv
        self.n_neighbor = n_neighbor
        self.make_dataset()
        self.index = np.arange(len(self.data))

        # if aug_lib == 'self_time':
        #     self.transform, self.to_tensor_transform = get_transform(
        #         aug_type if isinstance(aug_type, list) else [aug_type, ],
        #         prob=transform_prob, degree=degree,
        #     )
        # elif aug_lib == 'tsaug':
        #     self.transform = (
        #             TimeWarp(max_speed_ratio=(2, 3), prob=transform_prob, seed=0) * n_duplicates
        #             + Quantize(n_levels=(10, 60), prob=transform_prob, seed=0)
        #             + Drift(max_drift=(0.01, 0.4), seed=0) @ transform_prob
        #             + Convolve(window=['flattop', 'hann', 'triang'], size=(3, 11), prob=transform_prob, seed=0)
        #             + AddNoise(0, (0.01, 0.05), prob=transform_prob, seed=0)
        #         #+ Reverse() @ transform_prob  # with 50% probability, reverse the sequence
            # )

    def __getitem__(self, index):
        x, y, z, loss_weight = self.getitem_by_index(index)
        index = np.array(index)
        # if self.aug_lib == 'tsaug':
        #     x = self.transform.augment(x)
        #     if self.n_duplicates > 1:
        #         rp = lambda _: np.repeat(_, self.n_duplicates, axis=0)
        #         y, z, loss_weight = rp(y), rp(z), rp(loss_weight)
        #     return x, y, z, index, loss_weight

        if self.aug_type == 'none':
            return x, y, z, index, loss_weight

        img_list, label_list, loc_list, idx_list, loss_weight_list = [], [], [], [], []

        x = x.T if x.ndim > 1 else x[..., np.newaxis]
        for _ in range(self.n_views):
            img_transformed = self.transform(x).T
            img_list.append(self.to_tensor_transform(img_transformed))
            loc_list.append(z)
            label_list.append(y)
            idx_list.append(index)
            loss_weight_list.append(loss_weight)
        return img_list, label_list, loc_list, idx_list, loss_weight_list

    @property
    def inv_pred(self):
        return self._inv_pred

    @inv_pred.setter
    def inv_pred(self, inv_pred):
        if inv_pred is not None:
            self._inv_pred = inv_pred
        else:
            self._inv_pred = None

    def get_neighbors(self, index):
        core_loc = self.core_id == self.core_id[index]
        dist = np.abs(self.roi_coors[core_loc] - self.roi_coors[index]).sum(axis=1)
        min_dist_loc = np.argsort(dist)[:self.n_neighbor]
        neighbor_index = self.index[core_loc][min_dist_loc]
        return np.append(index, neighbor_index)

    def getitem_by_index(self, index):
        x = self.data[index] if self.n_neighbor == 0 else self.data[self.get_neighbors(index)]
        x = x.astype('float32')
        y = self.label[index]
        z = self.location[index]
        # if self.inv_pred is not None:
        #     if isinstance(index, int):
        #         inv_pred = self.inv_pred[self.frame_id[index]]
        #     else:
        #         inv_pred = torch.tensor([self.inv_pred[_] for _ in self.frame_id[index]])
        #     loss_weight = 1 + torch.abs(inv_pred - self.inv[index])
        # else:
        #     loss_weight = np.repeat(1, x.shape[0]).astype('float32') if not isinstance(index, int) else 1
        loss_weight = np.repeat(1, x.shape[0]).astype('float32') if not isinstance(index, int) else 1
        return x, y, z, loss_weight

    def _filter(self, key, condition):
        return self.data_dict[key][condition]

    def make_dataset(self, condition=None):
        if condition is None:
            condition = (self.data_dict['inv'] >= self.initial_min_inv) + (self.data_dict['inv'] == 0.)
            # condition *= (self.data_dict['stn'] < self.stn_alpha)

            # Range outlier
            # clf = IsolationForest(n_estimators=100, warm_start=True)
            # X = np.vstack((self.data_dict['data'].max(1),
            #                self.data_dict['data'].min(1),
            #                self.data_dict['data'].max(1) - self.data_dict['data'].min(1))).T
            # condition *= clf.fit_predict(X).astype(condition.dtype)

        for k in self.data_dict.keys():
            setattr(self, k, self._filter(k, condition))

        # self.data = torch.tensor(self.data, dtype=torch.float32) if self.aug_type == 'none' else self.data
        self.label = torch.tensor(self.label, dtype=torch.long)
        self.inv = torch.tensor(self.inv, dtype=torch.float32)
        self.location = torch.tensor(self.location, dtype=torch.uint8)
        # if self.transformer is not None:
        #     self.data, _ = robust_norm(self.data, self.transformer)

    def correct_labels(self, frame_id, core_len, predictions, true_involvement, predicted_involvement, correcting_params):
        """
        Correcting labels if the true and predicted involvements are similar
        :param frame_id: same as 'frame_id' item, except grouped by core
        :param core_len:
        :param predictions:
        :param predicted_involvement:
        :param true_involvement: same as 'true_involvement' item, except grouped by core
        :param correcting_params:
        :return:
        """
        inv_dif_thr, prob_thr = correcting_params.inv_dif_thr, correcting_params.prob_thr
        for _, cl in zip(frame_id, core_len):
            assert len(_) == cl
        inv_dif = np.abs(np.subtract(predicted_involvement, true_involvement))
        inv_dif[np.array(true_involvement) == 0] = 1  # no correction for benign labels
        correcting_mask = np.array(inv_dif <= inv_dif_thr)
        correcting_mask = np.concatenate([np.repeat(_, cl) for (_, cl) in zip(correcting_mask, core_len)])
        print(correcting_mask.sum(), predictions.max(), predictions[correcting_mask].max())
        correcting_mask[predictions.max(1) < prob_thr] = False
        n_correct = correcting_mask.sum()

        if n_correct > 0:
            # add new time-series if available
            ts_id_corrected = np.concatenate(frame_id)[correcting_mask]
            ts_id_updated = np.unique(np.concatenate([self.frame_id, ts_id_corrected]))
            if not np.all(np.isin(ts_id_updated, np.sort(self.frame_id))):
                condition = np.isin(self.data_dict['frame_id'], ts_id_updated)
                # condition *= (self.data_dict['stn'] < self.stn_alpha)

                print(f'{np.sum(condition) - len(self.frame_id)} new frames added')
                self.make_dataset(condition)
            else:
                condition = np.isin(self.data_dict['frame_id'], self.frame_id)  # use the current time-series IDs
            # Assign new labels
            if self.last_updated_label is None:
                new_label = torch.tensor(self.data_dict['label'].argmax(1).copy()).long()
            else:
                new_label = self.last_updated_label.clone()
            new_label[correcting_mask] = torch.tensor(predictions.argmax(1)[correcting_mask]).long()

            print(f'Cls_ratio: Old = {self.label.argmax(1).sum() / len(self.label):.3f}, '
                  f'New = {new_label[condition].sum() / condition.sum():.3f}')

            self.label = F.one_hot(new_label[condition])
            self.last_updated_label = new_label.clone()
            self.label_corrected = True
            print(f'Correcting amount: {100 * n_correct / len(correcting_mask):.1f}%')
            print(np.unique(self.data_dict['inv'][correcting_mask]))
        else:
            self.label_corrected = False

    def __len__(self):
        return self.label.size(0)

    @staticmethod
    def estimate_inv(label, core_len):
        new_inv = []
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if np.ndim(label) == 2:
            label = label.argmax(1)
        cur_idx = 0
        for cl in core_len:
            new_inv.append(np.round(label[cur_idx: cur_idx + cl].sum() / cl, 2))
            cur_idx += cl
        return np.array(new_inv).T

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


class DatasetV1_OLD(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, label, location, inv=None, unsup_data=None,
                 aug_type: Union[list, str] = ('G2',), unsup_aug_type=None, unsup_transform_prob=.4,
                 n_views=1, transform_prob=.2):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve
        self.data = torch.tensor(data, dtype=torch.float32) if aug_type == 'none' else data
        self.label = torch.tensor(label, dtype=torch.uint8)
        self.location = torch.tensor(location, dtype=torch.uint8)
        self.n_views = n_views
        self.aug_type = aug_type
        self.inv_pred = None
        self.transform, self.to_tensor_transform = get_transform(
            aug_type if isinstance(aug_type, list) else [aug_type, ],
            prob=transform_prob,
        )
        if inv is not None:
            self.inv = torch.tensor(inv, dtype=torch.float32)
        # self.transform = (
        #         TimeWarp() * 1  # random time warping 5 times in parallel
        #         # + Crop(size=170)  # random crop subsequences with length 300
        #         + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        #         + Reverse() @ 0.5  # with 50% probability, reverse the sequence
        #         + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        #         + Convolve(window='flattop', size=11, prob=.25)
        # )

        # For unlabeled dataset
        self.unsup_data, self.unsup_ratio, self.unsup_index, self.unsup_ratio = None, None, None, 1
        if self.aug_type != 'none':
            if (unsup_aug_type == 'none') or (unsup_aug_type is None):
                raise Exception('Data augmentations for unsupervised learning must be specified!')
            self.unsup_transform, _ = get_transform(
                unsup_aug_type if isinstance(unsup_aug_type, list) else [unsup_aug_type, ],
                prob=unsup_transform_prob,
            )
            self.unsup_data = unsup_data.astype('float32') if unsup_data is not None else None
            if self.unsup_data is not None:
                self.unsup_interval = int(np.floor(self.unsup_data.shape[0] / len(self.data)))
                self.unsup_index = np.arange(self.unsup_data.shape[0])

    def __getitem__(self, index):
        if self.unsup_data is None:
            return self.getitem_sup(index)
        return self.getitem_semi_sup(index)

    @property
    def inv_pred(self):
        return self._inv_pred

    @inv_pred.setter
    def inv_pred(self, inv_pred):
        if inv_pred is not None:
            self._inv_pred = torch.tensor(inv_pred, dtype=torch.float32)
        else:
            self._inv_pred = None

    def getitem_by_index(self, index):
        x = self.data[index]
        y = self.label[index]
        z = self.location[index]
        # loss_weight = 1 if self.inv_pred is None else \
        #     torch.exp(torch.abs(self.inv_pred[index] - self.inv[index])).item()
        # loss_weight = self.inv[index].item() if y[1] == 1 else 1
        loss_weight = 1
        return x, y, z, loss_weight

    def getitem_sup(self, index):
        x, y, z, loss_weight = self.getitem_by_index(index)
        if self.aug_type == 'none':
            return x, y, z, index, loss_weight

        img_list, label_list, loc_list, idx_list, loss_weight_list = [], [], [], [], []

        # img_transformed = self.transform.augment(x)
        # img_list.append(self.to_tensor_transform(img_transformed))  # .unsqueeze(0)
        for _ in range(self.n_views):
            # print("check this",x.shape) (1,286)
            img_transformed = self.transform(x.transpose()).squeeze()
            img_list.append(self.to_tensor_transform(img_transformed).unsqueeze(0))
            loc_list.append(z)
            label_list.append(y)
            idx_list.append(index)
            loss_weight_list.append(loss_weight)
        return img_list, label_list, loc_list, idx_list, loss_weight_list

    def getitem_semi_sup(self, index):
        img_list, label_list, loc_list, idx_list, loss_weight_list = self.getitem_sup(index)
        index = self.unsup_index[index] * self.unsup_interval
        x_unsup = self.unsup_data[index: index + self.unsup_ratio]
        img_unsup_list = []
        for _x_unsup in x_unsup:
            for _ in range(2):
                img_unsup_list.append(self.to_tensor_transform(
                    self.unsup_transform(_x_unsup[..., np.newaxis]).squeeze()))

        return img_list, label_list, loc_list, idx_list, loss_weight_list, img_unsup_list

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


def remove_empty_data(input_data, set_name, p_thr=.2):
    """

    :param input_data:
    :param set_name:
    :param p_thr: threshold of zero-percentage
    :return:
    """
    data = input_data[f"data_{set_name}"]
    s = [(data[i] == 0).sum() for i in range(len(data))]
    zero_percentage = [s[i] / np.prod(data[i].shape) for i in range(len(data))]
    include_idx = np.array([i for i, p in enumerate(zero_percentage) if p < p_thr])
    if len(include_idx) == 0:
        return input_data
    if len(include_idx) == len(data):
        return input_data

    for k in input_data.keys():
        if set_name in k:
            if isinstance(input_data[k], list):
                input_data[k] = [_ for i, _ in enumerate(input_data[k]) if i in include_idx]
            else:
                input_data[k] = input_data[k][include_idx]
    return input_data


def norm_01(x, *args):
    from sklearn.preprocessing import RobustScaler
    transformer = args[1]

    orig_shape = x.shape
    flattened = x.flatten()[..., np.newaxis]

    if transformer is None:
        new_x = x.reshape(orig_shape[0], 286, -1)
        new_x = np.swapaxes(new_x,1,2)
        new_x = new_x.reshape(orig_shape[0]*15, 286)
        transformer = RobustScaler().fit(new_x.T)
        new_x = transformer.transform(new_x.T).T
        new_x = new_x.reshape(orig_shape[0], 15, 286)
        new_x = np.swapaxes(new_x,1,2)
        x = new_x.reshape(orig_shape[0], 286, 3, 5)
        return x

    transfomed = transformer.transform(flattened)
    x = transfomed[..., 0].reshape(orig_shape)

    return x
    # return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
    # return (x - x.min()) / (x.max() - x.min())


def norm_framemax(x, max_):
    raise Exception('not correct yet')
    return x/max_


def stratify_groups(groups, num_time_series, marked_array, mark_low_threshold=.2):
    """Get a number of time-series within each group"""
    row_idx = []
    group_unique = np.unique(groups)
    for g in group_unique:
        # Find the time series in group g & mark off those already selected in previous iterations
        is_group = groups == g
        is_group_marked = is_group * marked_array

        # Reset marked array if 80% of the time series in the current group have been selected
        if (is_group_marked.sum() / is_group.sum()) < mark_low_threshold:
            is_group_marked = is_group
            marked_array[is_group] = True

        # Randomly selected time-series within those of the current group
        replace = True if np.sum(is_group_marked) < num_time_series else False
        row_idx.append(np.random.choice(np.where(is_group_marked)[0], num_time_series, replace=replace))
        # print(g, (sum(is_group_marked) / sum(is_group)))

        marked_array[row_idx[-1]] = False
    return np.concatenate(row_idx), marked_array


def normalize(input_data, set_name, to_all_cores=False, to_framemax=False):
    _norm = norm_framemax if to_framemax else norm_01
    frame_max = input_data[f'Frame_max_{set_name}']
    transformer = None

    if to_all_cores:
        from sklearn.preprocessing import RobustScaler
        signal_data = input_data[f'data_{set_name}']
        # signal_data = input_data[f'data_train']
        list_s = [i for i in signal_data]
        signal_data = np.concatenate(list_s)
        a = signal_data.flatten()
        transformer = RobustScaler().fit(a[...,np.newaxis])

    for i, x in enumerate(input_data[f'data_{set_name}']):
        input_data[f'data_{set_name}'][i] = _norm(x.astype('float32'), frame_max[i], transformer)
    return input_data


def shuffle_signals(input_data, data, label, corename, GS, frame_max, patientId):

    data_list = []
    label_list = []
    corename_list = []
    # GS_list = []
    # frame_max_list = []
    # patientId_list = []
    names = ['data_', 'corename_', 'label_']

    for i in range(len(data)):
        data_list.append(data[i])
        label_list.append(np.repeat(label[i], data[i].shape[0]))
        temp = np.tile(corename[i], data[i].shape[0])
        corename_list.append(temp.reshape((data[i].shape[0], 1)))
        # GS_list.append(GS[i])
        # frame_max_list.append(frame_max[i])
        # patientId_list.append(patientId[i])

    data_list = np.concatenate(data_list)
    corename_list = np.concatenate(corename_list)
    label_list = np.concatenate(label_list)[:, np.newaxis]
    # frame_max_list = np.concatenate(frame_max_list)
    # GS_list = np.concatenate(GS_list)
    # patientId_list = np.concatenate(patientId_list)

    # bundle by 15
    # no_data = data_list.shape[0]
    # data_list = data_list[:, np.newaxis, ...].reshape(int(no_data/15), 15, -1)
    # corename_list = corename_list[:, np.newaxis, ...].reshape(int(no_data/15), 15, -1)
    # label_list = label_list[:, np.newaxis, ...].reshape(int(no_data/15), 15, -1)

    data, corename, label = sk.shuffle(data_list, corename_list, label_list, random_state=0)

    # debundle by 15
    # data = data.reshape(int(no_data), 1, -1).squeeze(axis=1)
    # corename = corename.reshape(int(no_data), 1, -1).squeeze(axis=1)
    # label = label.reshape(int(no_data), 1, -1).squeeze(axis=1)

    variables = [data, corename, label]

    for i, n in enumerate(names):
        l1 = int(len(variables[i])*(61/100))
        l2 = int(len(variables[i])*(21/100))
        input_data[n+'train'] = variables[i][np.newaxis, :l1, ...]
        input_data[n+'val'] = variables[i][np.newaxis, l1:l1+l2, ...]
        input_data[n+'test'] = variables[i][np.newaxis, l1+l2:, ...]

    return input_data


def shuffle_patients(input_data, signal_split=False):

    data = []
    corename = []
    frame_max = []
    GS = []
    label = []
    patientId = []

    names = ['data_', 'corename_', 'Frame_max_', 'GS_', 'label_', 'PatientId_']

    for k in input_data.keys():
        temp_data = input_data[k]
        if names[0] in k:
            data.append(temp_data)
        elif names[1] in k:
            corename.append(temp_data)
        elif names[2] in k:
            frame_max.append(temp_data)
        elif names[3] in k:
            GS.append(temp_data)
        elif names[4] in k:
            label.append(temp_data)
        elif names[5] in k:
            patientId.append(temp_data)

    data = np.concatenate(data)
    corename = np.concatenate(corename)
    frame_max = np.concatenate(frame_max)
    GS = np.concatenate(GS)
    label = np.concatenate(label)
    patientId = np.concatenate(patientId)

    if signal_split:
        return shuffle_signals(input_data, data, label, corename, GS, frame_max, patientId)

    data, corename, frame_max, GS, label, patientId = sk.shuffle(data, corename, frame_max,
                                                                 GS, label, patientId, random_state=0)
    variables = [data, corename, frame_max, GS, label, patientId]

    for i, n in enumerate(names):
        input_data[n+'train'] = variables[i][:386]
        input_data[n+'val'] = variables[i][386:516]
        input_data[n+'test'] = variables[i][516:]

    return input_data


def preprocess(input_data, p_thr=.2, to_norm=False, shffl_patients=False, signal_split=False,
               split_rs=-1, val_size=0.2):
    """
    Remove data points which have percentage of zeros greater than p_thr
    :param input_data:
    :param p_thr:
    :param to_norm:
    :param shffl_patients:
    :param split_rs:
    :param val_size:
    :return:
    """
    for set_name in ['train', 'val', 'test']:
        # input_data = remove_empty_data(input_data, set_name, p_thr)
        if to_norm:
            input_data = normalize(input_data, set_name, to_framemax=False)
    if shffl_patients or signal_split:
        input_data = shuffle_patients(input_data, signal_split=signal_split)
    if split_rs >= 0:
        input_data = merge_split_train_val(input_data, random_state=split_rs, val_size=val_size)
    return input_data


def concat_data(included_idx, data, label, core_name, inv, patient_id, roi_coors, stn):
    """ Concatenate data from different cores specified by 'included_idx' """
    core_counter = 0
    data_c, label_c, core_name_c, inv_c, core_len, core_id_c, patient_id_c, frame_id_c, roi_coors_c, stn_c = \
        [[] for _ in range(10)]
    core_len_all = []

    for i in range(len(data)):
        # these three lines are to determine frame_id for each frame in the whole dataset
        core_len_all.append(data[i].shape[0])
        start = sum(core_len_all[:-1])
        end = sum(core_len_all)

        if included_idx[i]:
            data_c.append(data[i])
            label_c.append(np.repeat(label[i], data[i].shape[0]))
            temp = np.tile(core_name[i], data[i].shape[0])
            core_name_c.append(temp.reshape((data[i].shape[0], -1)))
            inv_c.append(np.repeat(inv[i], data[i].shape[0]))
            core_id_c.append(np.repeat(i+1, data[i].shape[0]))
            patient_id_c.append(np.repeat(patient_id[i], data[i].shape[0]))
            core_len.append(data[i].shape[0])
            frame_id_c.append(list(range(start+1, end+1)))
            # roi_coors_c.append(roi_coors[i])
            # stn_c.append(stn[i])
            core_counter += 1
    # roi_coors_c = np.concatenate(roi_coors_c)
    data_c = np.concatenate(data_c).astype('float32')
    data_c = fix_dim(data_c, 'patch')
    label_c = to_categorical(np.concatenate(label_c))
    core_name_c = to_categorical(np.concatenate(core_name_c))
    inv_c = np.concatenate(inv_c)
    core_id_c = np.concatenate(core_id_c)
    patient_id_c = np.concatenate(patient_id_c)
    frame_id_c = np.concatenate(frame_id_c)
    # stn_c = np.concatenate(stn_c)
    return data_c, label_c, core_name_c, inv_c, core_id_c, patient_id_c, frame_id_c, core_len, None, None #roi_coors_c, stn_c


def create_datasets_v1(data_file, norm=None, min_inv=0.4, aug_type='none', n_views=2,
                       unlabelled_data_file=None, unsup_aug_type=None,
                       to_norm=False):
    """
    Create training, validation and test sets
    :param data_file:
    :param norm:
    :param min_inv:
    :param aug_type:
    :param n_views:
    :param unlabelled_data_file:
    :param to_norm:
    :return:
    """
    input_data = load_pickle(data_file)
    input_data = preprocess(input_data, to_norm=to_norm)

    data_train = input_data["data_train"]
    inv_train = input_data["inv_train"]
    label_train = (inv_train > 0).astype('uint8')
    core_name_train = input_data["corename_train"].astype(np.float)

    # data_train = data_train + input_data["data_val"]
    # label_train = np.concatenate([label_train, input_data["label_val"]], axis=0)
    # inv_train = np.concatenate([inv_train, input_data["inv_val"]], axis=0)
    # CoreN_train = np.concatenate([CoreN_train, input_data["corename_val"].astype(np.float)])

    included_idx = [True for _ in label_train]
    included_idx = [False if (inv < min_inv) and (inv > 0) else tr_idx for inv, tr_idx in zip(inv_train, included_idx)]
    signal_train, label_train, name_train, inv_train = concat_data(
        included_idx, data_train, label_train, core_name_train, inv_train,
    )

    unsup_data = np.concatenate(data_train)
    # unsup_data = None
    # unsup_data = load_unlabelled_datasets(unlabelled_data_file) if 'none' not in unlabelled_data_file else None

    trn_ds = DatasetV1(signal_train, label_train, name_train, inv_train, transform_prob=.2,
                       unsup_data=unsup_data, aug_type=aug_type, n_views=n_views,
                       unsup_aug_type=unsup_aug_type, unsup_transform_prob=.8,
                       )  # ['magnitude_warp', 'time_warp'])

    train_stats = None
    train_set = create_datasets_test(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                     train_stats=train_stats)
    val_set = create_datasets_test(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                   train_stats=train_stats)
    test_set = create_datasets_test(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                    train_stats=train_stats)

    return trn_ds, train_set, val_set, test_set


class DatasetV2(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    from typing import Union

    def __init__(self, data, label, location, inv, groups,
                 aug_type: Union[list, str] = ('G2',), n_views=1, transform_prob=.2,
                 time_series_per_group=16):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve

        self.data = torch.tensor(data, dtype=torch.float32) if aug_type == 'none' else data
        self.label = torch.tensor(label, dtype=torch.uint8)
        self.location = torch.tensor(location, dtype=torch.uint8)
        self.n_views = n_views
        self.aug_type = aug_type
        self.inv_pred = None
        self.transform, self.to_tensor_transform = get_transform(
            aug_type if isinstance(aug_type, list) else [aug_type, ],
            prob=transform_prob,
        )
        # self.transform = (
        #         TimeWarp() * 1  # random time warping 5 times in parallel
        #         # + Crop(size=170)  # random crop subsequences with length 300
        #         + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        #         + Reverse() @ 0.5  # with 50% probability, reverse the sequence
        #         + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        #         + Convolve(window='flattop', size=11, prob=.25)
        # )
        self.inv = torch.tensor(inv, dtype=torch.float32)
        self.time_series_per_group = time_series_per_group
        self.groups = groups
        self.marked_array = [np.ones((d.shape[0]), dtype='bool') for d in self.data]

    @property
    def inv_pred(self):
        return self._inv_pred

    @inv_pred.setter
    def inv_pred(self, inv_pred):
        if inv_pred is not None:
            self._inv_pred = torch.tensor(inv_pred, dtype=torch.float32)
        else:
            self._inv_pred = None

    def __getitem__(self, index):
        row_idx, self.marked_array[index] = stratify_groups(self.groups[index], self.time_series_per_group,
                                                            self.marked_array[index])
        x = self.data[index][row_idx]
        y = self.label[index]
        z = self.location[index]
        if self.inv_pred is None:
            loss_weight = 1
        else:
            loss_weight = torch.exp(torch.abs(self.inv_pred[index] - self.inv[index])).item()
            # loss_weight *= 2 if y == 0 else 1

        if self.aug_type == 'none':
            return x, y, z, index

        img_list, label_list, loc_list, idx_list, loss_weight_list = [], [], [], [], []
        if self.transform is not None:
            # img_list = list(self.to_tensor_transform(self.transform.augment(x)))  # .unsqueeze(0)
            # for _x in x:
            img_list = list(x)
            for i in range(x.shape[0]):
                for _ in range(self.n_views):
                    # img_transformed = self.transform(_x[..., np.newaxis]).squeeze()
                    # img_list.append(self.to_tensor_transform(img_transformed))  # .unsqueeze(0)
                    # img_list.append(_x)
                    loc_list.append(z)
                    label_list.append(y)
                    idx_list.append(index)
                    loss_weight_list.append(loss_weight)
        return img_list, label_list, loc_list, idx_list, loss_weight_list

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


def create_datasets_v2(data_file, norm=None, min_inv=0.4, aug_type='none', n_views=4, to_norm=False):
    input_data = load_pickle(data_file)
    # unsup_data = load_pickle(data_file.replace('mimic', 'unsup'))
    input_data = preprocess(input_data, to_norm=to_norm)

    data_train = input_data["data_train"]
    inv_train = input_data["inv_train"]
    label_train = to_categorical((inv_train > 0).astype('uint8'))
    CoreN_train = input_data["corename_train"].astype(np.float)
    groups_train = load_pickle(data_file.replace('.pkl', '_groups.pkl'))['nc30']

    included_idx = [False if ((inv < min_inv) and (inv > 0)) else True for inv in inv_train]

    # Filter unwanted cores
    label_train = label_train[included_idx]
    inv_train = inv_train[included_idx]
    CoreN_train = CoreN_train[included_idx]
    data_train = [data_train[i].astype('float32') for i, included in enumerate(included_idx) if included]
    groups_train = [groups_train[i] for i, included in enumerate(included_idx) if included]

    # Create training dataset
    trn_ds = DatasetV2(data_train, label_train, CoreN_train, inv_train, groups=groups_train,
                       aug_type=aug_type, n_views=n_views)  # ['magnitude_warp', 'time_warp'])

    train_set = create_datasets_test(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                     train_stats=None)
    val_set = create_datasets_test(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                   train_stats=None)
    test_set = create_datasets_test(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                    train_stats=None)

    return trn_ds, train_set, val_set, test_set


def create_datasets_test(data_file, state, norm='Min_Max', min_inv=0.4, augno=0, inv_state='normal', input_data=None,
                         return_array=False, train_stats=None):
    if input_data is None:
        input_data = load_pickle(data_file)

    data_test = input_data["data_" + state]  # [0]
    roi_coors_test = input_data['roi_coors_' + state]
    label_inv = input_data["inv_" + state]  # [0]
    label_test = (label_inv > 0).astype('uint8')
    CoreN_test = input_data["corename_" + state]
    patient_id_bk = input_data["PatientId_" + state]  # [0]
    involvement_bk = input_data["inv_" + state]  # [0]
    gs_bk = np.array(input_data["GS_" + state])  # [0]

    included_idx = [False if ((inv < min_inv) and (inv > 0)) else True for inv in label_inv]

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


def _preprocess(x_train):
    # Normalize
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    # Test is secret
    # x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
    # x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
    return x_train, (x_train_min, x_train_max)


########################################################################################################################
def create_datasets_Exact(dataset_name, data_file, norm=None, min_inv=0.4, aug_type='none', n_views=2,
                          unlabelled_data_file=None, unsup_aug_type=None, to_norm=False,
                          signal_split=False, use_inv=True, use_patch=False, dynmc_dataroot=None,
                          split_random_state=-1, val_size=.2):
    """
    Create training, validation and test sets
    :param data_file:
    :param norm:
    :param min_inv:
    :param aug_type:
    :param n_views:
    :param unlabelled_data_file:
    :param to_norm:
    :param signal_split:
    :return:
    """

    print("load matlab dataset")
    input_data = load_matlab(data_file)
    print("loading done")

    input_data = preprocess(input_data, to_norm=to_norm, shffl_patients=False, signal_split=signal_split,
                            split_rs=split_random_state, val_size=val_size)

    trn_ds = DatasetV1(*extract_subset(input_data, 'train', 0.4), initial_min_inv=min_inv, aug_type=aug_type)

    train_stats = None
    train_set = create_datasets_test_Exact(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                           train_stats=train_stats, dataset_name=dataset_name,
                                           signal_split=signal_split, use_inv=use_inv, use_patch=use_patch)
    val_set = create_datasets_test_Exact(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                         train_stats=train_stats, dataset_name=dataset_name,
                                         signal_split=signal_split, use_inv=use_inv, use_patch=use_patch)
    test_set = create_datasets_test_Exact(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                          train_stats=train_stats, dataset_name=dataset_name,
                                          signal_split=signal_split, use_inv=use_inv, use_patch=use_patch)

    # return trn_ds, train_set, val_set, test_set
    return trn_ds, train_set, val_set, test_set


def concat_data_Exact(included_idx, data, label=None, core_name=None, inv=None,
                      signal_split=False, dataset_name=None, use_inv=True, use_patch=False):
    """ Concatenate data from different cores specified by 'included_idx' """
    core_counter = 0
    data_c, label_c, core_name_c, inv_c, core_len = [], [], [], [], []
    corelen = []
    for i in range(len(data)):
        if included_idx[i] or (not use_inv):
            data_c.append(data[i])
            core_len.append(np.repeat(core_counter, data[i].shape[0]))
            corelen.append(data[i].shape[0])
            core_counter += 1
            inv_c.append(np.repeat(inv[i], data[i].shape[0]))
            if not signal_split:
                label_c.append(np.repeat(label[i], data[i].shape[0]))
                temp = np.tile(core_name[i], data[i].shape[0])
                core_name_c.append(temp.reshape((data[i].shape[0], 1)))

    data_c = np.concatenate(data_c).astype('float32')
    data_c = fix_dim(data_c, dataset_name)
    # import matplotlib.pyplot as plt
    # for i in range(286):
    #     plt.plot(range(15), data_c[15000, 0, i, :],'b')
    # plt.show()
    inv_c = np.concatenate(inv_c)
    inv_out = inv_c if use_inv else None

    if signal_split:
        label_c = to_categorical(label.squeeze())
        core_name_c = to_categorical(core_name.squeeze())
    else:
        label_c = to_categorical(np.concatenate(label_c))
        core_name_c = to_categorical(np.concatenate(core_name_c))

    # manually balance the number of cancers and benigns
    # data_c, label_c, core_name_c, inv_out = manual_balance(data_c, label_c, core_name_c, inv_out)

    # creating patches
    if use_patch:
        data_c, label_c, core_name_c, inv_out, corelen = create_patch(data_c, label_c, core_name_c,
                                                                      inv_out, corelen, window=26, overlap=15)

    # seperating 3 different focal points as channels and using them as 3 independent data
    # label_c = np.repeat(label_c,3, axis=0)
    # core_name_c = np.repeat(core_name_c,3, axis=0)
    # inv_out = np.repeat(inv_out,3, axis=0)

    return data_c, label_c, core_name_c, inv_out


def create_datasets_test_Exact(data_file, state, dataset_name, norm='Min_Max', min_inv=0.4, augno=0,
                               input_data=None, inv_state='normal', return_array=False, train_stats=None,
                               signal_split=False, use_inv=True, use_patch=False, split_patches=False):
    if input_data is None:
        input_data = load_matlab(data_file)

    data_test = input_data["data_" + state]
    label_inv = input_data["inv_" + state]
    # label_test = (label_inv > 0).astype('uint8')
    label_test = input_data['label_' + state].astype('uint8')
    CoreN_test = input_data["corename_" + state]
    patient_id_bk = input_data["PatientId_" + state]
    gs_bk = input_data["GS_" + state]
    true_label = label_test

    included_idx = [False if ((inv < min_inv) and (inv > 0)) else True for inv in label_inv]

    corelen, core_len_all = [], []
    target_test = []
    name_tst = []
    frame_id = []
    frame_id_c = []

    # Working with BK dataset
    bags_test = []
    sig_inv_test = []
    for i in range(len(data_test)):
        # these three lines are to determine frame_id for each frame in the whole dataset
        core_len_all.append(data_test[i].shape[0])
        start = sum(core_len_all[:-1])
        end = sum(core_len_all)
        frame_id.append(list(range(start + 1, end + 1)))

        if included_idx[i] or (not use_inv):
            bags_test.append(data_test[i])
            corelen.append(data_test[i].shape[0])
            if not signal_split:
                target_test.append(np.repeat(label_test[i], data_test[i].shape[0]))
                temp = np.tile(CoreN_test[i], data_test[i].shape[0])
                name_tst.append(temp.reshape((data_test[i].shape[0], 1)))
                frame_id_c.append(frame_id[i])
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
    signal_test = fix_dim(signal_test, dataset_name)
    sig_inv_test = np.concatenate(sig_inv_test)
    patient_id_bk = patient_id_bk[included_idx]
    gs_bk = np.array(gs_bk)[included_idx]
    label_inv = label_inv[included_idx]
    # label_inv_out = label_inv
    label_inv_out = label_inv if use_inv else None

    if signal_split:
        target_test = label_test.squeeze()
        name_tst = to_categorical(CoreN_test.squeeze())
        true_label = [target_test[0]]
    else:
        target_test = np.concatenate(target_test)
        name_tst = to_categorical(np.concatenate(name_tst))

    if train_stats:
        x_train_min, x_train_max = train_stats
        signal_test = 2. * (signal_test - x_train_min) / (x_train_max - x_train_min) - 1.
    #
    # signal_test = (signal_test - signal_test.mean(axis=1)[..., np.newaxis]) / signal_test.std(axis=1)[
    #     ..., np.newaxis]

    if return_array:
        return signal_test, corelen, None, patient_id_bk, gs_bk

    label_test = to_categorical(target_test)
    # creating patches
    if use_patch:
        signal_test, label_test, name_tst, sig_inv_test, corelen = create_patch(signal_test, label_test, name_tst,
                                                                                sig_inv_test, corelen, window=26, overlap=15)

    # # manually balance the number of cancers and benigns
    # data_c, label_c, core_name_c, inv_out = manual_balance(signal_test, label_test, name_tst, sig_inv_test)

    # spliting the patches to 32x32 instead of 256x256
    if split_patches:
        raise NotImplemented ## for using along label refinement the function below should include frame_id_c
        signal_test, label_test, name_tst, sig_inv_test, corelen = split_patch(signal_test, label_test, name_tst,
                                                                               sig_inv_test, corelen)

    print("cancer labels shape", label_test[label_test[:,1]>label_test[:,0],:].shape)
    print("data shape", signal_test.shape)
    tst_ds = TensorDataset(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           torch.tensor(sig_inv_test, dtype=torch.float32).unsqueeze(1)) if use_inv else \
             TensorDataset(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8))
    return tst_ds, corelen, label_inv_out, patient_id_bk, gs_bk, None, true_label, frame_id_c


def fix_dim(data, dataset_name):
    if '2D' in dataset_name:
        tmp = np.swapaxes(data, 1, 2)
        # tmp = tmp[:,0:1,:,: ]
        # tmp = data.reshape((data.shape[0], data.shape[1], -1))
        # tmp = np.swapaxes(tmp, 1, 2)
        # tmp = data
        # return np.swapaxes(tmp.reshape((tmp.shape[0], tmp.shape[1], -1)),1,2)
        # return tmp.reshape((tmp.shape[0], tmp.shape[1], -1))
        # tmp = tmp.reshape((tmp.shape[0] * tmp.shape[1], 1, tmp.shape[2], 1))
        tmp = tmp.reshape((tmp.shape[0] * tmp.shape[1], 1, tmp.shape[2], tmp.shape[3]))
        # tmp_list = []
        # tmp_list.append(tmp[:,:,:,[1,3]])
        # tmp_list.append(tmp[:,:,:,[0,1]])
        # tmp = np.concatenate(tmp_list)
        # return np.repeat(tmp, 2, axis=3)
        # return tmp.reshape((tmp.shape[0] * tmp.shape[1], 1, tmp.shape[2], tmp.shape[3]))
        # return tmp.reshape((tmp.shape[0], 1, tmp.shape[1], -1))
        # return np.swapaxes(tmp, 1, 2)
        return tmp
    elif 'patch' in dataset_name:
        tmp = data[:, np.newaxis, ...]
        # tmp = np.repeat(tmp, 3, axis=1)
        return tmp
    else:
        tmp = data[:, np.newaxis, ...]
        # tmp = np.repeat(tmp, 3, axis=1)
        return tmp


def manual_balance(data_c, label_c, core_name_c, inv_out):
    no_cancer = (label_c[:,1]==1).sum()
    cancer_ind = np.where(label_c[:, 1] == 1)
    benign_ind = np.where(label_c[:, 0] == 1)

    benign_ind = sk.shuffle(benign_ind[0], random_state=0)
    benign_ind = benign_ind[:no_cancer]
    ind = np.concatenate((benign_ind, cancer_ind[0])).astype(int)

    data_c = data_c[ind, ...]
    label_c = label_c[ind, ...]
    core_name_c = core_name_c[ind, ...]
    inv_out = inv_out[ind]
    return data_c, label_c, core_name_c, inv_out


def create_patch(data_c, label_c, core_name_c, inv_c, corelen, window=5, overlap=0):
    data = []
    label = []
    core_name = []
    inv = []
    core_cumsum = np.cumsum(corelen)
    for i in range(len(corelen)):
        for j in range((corelen[i]-window)//(window-overlap)+1):
            core_start = core_cumsum[i-1] if i!=0 else 0
            window_start = j*(window-overlap)
            window_end = window_start + window
            for k in range(11):
                data.append(data_c[np.newaxis, core_start+window_start:core_start+window_end, k*26:(k+1)*26, :])
                label.append(label_c[np.newaxis, core_start+window_start, ...])
                core_name.append(core_name_c[np.newaxis, core_start+window_start, ...])
                if inv_c is not None:
                    inv.append(inv_c[np.newaxis, core_start+window_start, ...])
        corelen[i] = (len(data) - np.cumsum(corelen)[i-1]) if i!=0 else len(data) - np.array(0)

    data = np.concatenate(data)
    data = np.swapaxes(data, 1, 3)
    label = np.concatenate(label)
    core_name = np.concatenate(core_name)
    inv = np.concatenate(inv)
    # print("data shape", data.shape)
    return data, label, core_name, inv, corelen


def split_patch(signal_test, label_test, name_tst, sig_inv_test, corelen, patch_size=256):
    from einops import rearrange

    no_patches = (256 // patch_size) ** 2

    signal = rearrange(signal_test, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size)
    label = np.repeat(label_test, no_patches, axis=0)
    name = np.repeat(name_tst, no_patches, axis=0)
    sig_inv = np.repeat(sig_inv_test, no_patches, axis=0)
    corelen_ = [no_patches * cl for cl in corelen]

    return signal, label, name, sig_inv, corelen_


def extract_subset(input_data, set_name, min_included_inv=.4, to_concat=True, core_list=None):
    """
    Extract subset 'set_name' from input_data using 'min_inv'
    :param input_data: loaded from pkl or matlab file
    :param set_name: 'train', 'val', or 'test'
    :param min_included_inv: minimum involvement for keeping data in the training set
    :param to_concat: concat selected data
    :param core_list: ID of included cores
    :return:
    """
    data = input_data["data_" + set_name]
    inv = input_data["inv_" + set_name]
    label = input_data['label_' + set_name].astype('uint8')
    core_name = input_data["corename_" + set_name]
    patient_id = input_data["PatientId_" + set_name]

    # roi_coors = [np.array(_).T for _ in input_data[add_suf('roi_coors')]]
    # stn = deepcopy(frame_id)  # input_data[add_suf('stn')]
    # stn = deepcopy(frame_id)  # input_data[add_suf('stn')]
    included_idx = [True for _ in label]
    included_idx = [False if (inv < min_included_inv) and (inv > 0) else tr_idx for inv, tr_idx in
                    zip(inv, included_idx)]
    if core_list is not None:  # filter by core-id
        # included_idx = np.bitwise_and(included_idx, np.isin(core_id, core_list))
        raise NotImplemented
    if to_concat:
        return concat_data(included_idx, data, label, core_name, inv, patient_id, None, None)
    else:
        raise NotImplemented
        # core_len = [len(_) for _ in data]
        # included_idx = np.argwhere(included_idx).T[0]
        # get_included = lambda x: [x[i] for i in included_idx]
        # for v in ['data', 'label', 'core_name', 'inv', 'core_id', 'patient_id', 'frame_id', 'core_len',
        #           'roi_coors', 'stn']:
        #     eval(f'{v} = get_included({v})')
        # return data, label, core_name, inv, core_id, patient_id, frame_id, core_len, roi_coors, stn


def merge_split_train_val(input_data, random_state=0, val_size=.4, verbose=False):
    """

    :param input_data:
    :param random_state:
    :param val_size:
    :param verbose:
    :return:
    """
    # merge train-val then randomize patient ID to split train-val again
    gs = list(input_data["GS_train"]) + list(input_data['GS_val'])
    pid = list(input_data["PatientId_train"]) + list(input_data['PatientId_val'])
    inv = list(input_data["inv_train"]) + list(input_data['inv_val'])

    df1 = pd.DataFrame({'pid': pid, 'gs': gs, 'inv': inv})
    # df1 = df1.assign(gs_merge=df1.gs.replace({'-': 'Benign',
    #                                           '3+3': 'G3', '3+4': 'G3', '4+3': 'G4',
    #                                           '4+4': 'G4', '4+5': 'G4', '5+4': 'G4'}))
    df1 = df1.assign(condition=df1.gs.replace({'Benign': 'Benign', 'G7': 'Cancer', 'G8': 'Cancer',
                                                     'G9': 'Cancer', 'G10': 'Cancer'}))
    # df1.gs.replace({'-': 'Benign'}, inplace=True)

    train_inds, test_inds = next(GroupShuffleSplit(test_size=val_size, n_splits=2,
                                                   random_state=random_state).split(df1, groups=df1['pid']))
    df1 = df1.assign(group='train')
    df1.loc[test_inds, 'group'] = 'val'
    df1 = df1.sort_values(by='pid')

    # for _ in df1.pid.unique():
    #     tmp1 = list(np.unique(df1[df1.pid <= _].gs_merge, return_counts=True))
    #     tmp1[1] = np.round(tmp1[1] / tmp1[1].sum(), 2)
    #     tmp2 = list(np.unique(df1[df1.pid > _].gs_merge, return_counts=True))
    #     tmp2[1] = np.round(tmp2[1] / tmp2[1].sum(), 2)
    #     print(_, dict(zip(*tmp1)), dict(zip(*tmp2)))

    # import seaborn as sns
    # import pylab as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # # tr, v = df1[df1.group == 'train'], df1[df1.group == 'val']
    # plt.close('all')
    # sns.countplot(x='gs_merge', data=df1[(df1.inv >= .4) | (df1.inv == 0)], hue='group')
    # plt.show()

    pid_tv = {
        'train': df1[df1.group == 'train'].pid.unique(),
        'val': df1[df1.group == 'val'].pid.unique(),
    }
    # Merge train - val
    keys = [f[:-4] for f in input_data.keys() if 'val' in f]
    merge = {}
    for k in keys:
        if isinstance(input_data[f'{k}_train'], list):
            merge[k] = input_data[f'{k}_train'] + input_data[f'{k}_val']
        else:
            merge[k] = np.concatenate([input_data[f'{k}_train'], input_data[f'{k}_val']], axis=0)

    # Initialize the new input_data
    target = {}
    for set_name in ['train', 'val']:
        for k in keys:
            target[f'{k}_{set_name}'] = []
    # Re-split data into two sets based on randomized patient ID
    for i, pid in enumerate(merge['PatientId']):
        for set_name in ['train', 'val']:
            if pid in pid_tv[set_name]:
                for k in keys:
                    k_target = f'{k}_{set_name}'
                    target[k_target].append(merge[k][i])
    # Assign to original input data after finishing creating a new one
    for set_name in ['train', 'val']:
        for k in keys:
            k_target = f'{k}_{set_name}'
            input_data[k_target] = target[k_target]
            if isinstance(merge[k], np.ndarray):
                input_data[k_target] = np.array(target[k_target]).astype(merge[k].dtype)

    if verbose:
        for set_name in ['train', 'val']:
            for k in keys:
                print(k, len(input_data[f'{k}_{set_name}']))

    return input_data