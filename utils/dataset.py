import os
from typing import Union

import torch
import numpy as np
import sklearn.utils as sk
import torch.multiprocessing
from torch.utils.data import TensorDataset
from self_time.optim.pretrain import get_transform

torch.multiprocessing.set_sharing_strategy('file_system')

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
    return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))


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


def normalize(input_data, set_name, to_framemax=False):
    if to_framemax:
        _norm = norm_framemax
    else:
        _norm = norm_01

    max_ = input_data[f'Frame_max_{set_name}']
    for i, x in enumerate(input_data[f'data_{set_name}']):
        input_data[f'data_{set_name}'][i] = _norm(x.astype('float32'), max_[i])
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


def preprocess(input_data, p_thr=.2, to_norm=True, shffl_patients=False, signal_split=False):
    """
    Remove data points which have percentage of zeros greater than p_thr
    :param input_data:
    :param p_thr:
    :param to_norm:
    :param shffl_patients:
    :return:
    """
    for set_name in ['train', 'val', 'test']:
        input_data = remove_empty_data(input_data, set_name, p_thr)
        if to_norm:
            input_data = normalize(input_data, set_name, to_framemax=False)
    if shffl_patients or signal_split:
        input_data = shuffle_patients(input_data, signal_split=signal_split)
    return input_data


def concat_data(included_idx, data, label=None, core_name=None, inv=None):
    """ Concatenate data from different cores specified by 'included_idx' """
    core_counter = 0
    data_c, label_c, core_name_c, inv_c, core_len = [], [], [], [], []
    for i in range(len(data)):
        if included_idx[i]:
            data_c.append(data[i])
            label_c.append(np.repeat(label[i], data[i].shape[0]))
            temp = np.tile(core_name[i], data[i].shape[0])
            core_name_c.append(temp.reshape((data[i].shape[0], 8)))
            inv_c.append(np.repeat(inv[i], data[i].shape[0]))
            core_len.append(np.repeat(core_counter, data[i].shape[0]))
            core_counter += 1
    data_c = np.concatenate(data_c).astype('float32')
    label_c = to_categorical(np.concatenate(label_c))
    core_name_c = np.concatenate(core_name_c)
    inv_c = np.concatenate(inv_c)
    return data_c, label_c, core_name_c, inv_c


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


######################################################################################
######################################################################################
def create_datasets_Exact(dataset_name, data_file, norm=None, min_inv=0.4, aug_type='none', n_views=2,
                          unlabelled_data_file=None, unsup_aug_type=None,
                          to_norm=False, signal_split=False, use_inv=True):
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

    input_data = preprocess(input_data, to_norm=to_norm, shffl_patients=False, signal_split=signal_split)
    data_train = input_data["data_train"]
    inv_train = input_data["inv_train"]
    # label_train = (inv_train > 0).astype('uint8')
    label_train = input_data['label_train'].astype('uint8')
    core_name_train = input_data["corename_train"].astype(np.float)

    # inv_train = input_data["inv_train"]
    # label_train = (inv_train > 0).astype('uint8')

    # data_train = data_train + input_data["data_val"]
    # label_train = np.concatenate([label_train, input_data["label_val"]], axis=0)
    # inv_train = np.concatenate([inv_train, input_data["inv_val"]], axis=0)
    # CoreN_train = np.concatenate([CoreN_train, input_data["corename_val"].astype(np.float)])

    included_idx = [True for _ in label_train]
    included_idx = [False if (inv < min_inv) and (inv > 0) else tr_idx for inv, tr_idx in zip(inv_train, included_idx)]
    signal_train, label_train, name_train, inv_train = concat_data_Exact(included_idx, data_train, label_train,
                                                                         core_name_train, inv_train,
                                                                         dataset_name=dataset_name,
                                                                         signal_split=signal_split, use_inv=use_inv)

    # unsup_data = np.concatenate(data_train)
    unsup_data = None
    # unsup_data = load_unlabelled_datasets(unlabelled_data_file) if 'none' not in unlabelled_data_file else None
    trn_ds = DatasetV1(signal_train, label_train, name_train, inv_train, transform_prob=.2,
                       unsup_data=unsup_data, aug_type=aug_type, n_views=n_views,
                       unsup_aug_type=unsup_aug_type, unsup_transform_prob=.8,
                       )  # ['magnitude_warp', 'time_warp'])

    train_stats = None
    train_set = create_datasets_test_Exact(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                           train_stats=train_stats, dataset_name=dataset_name,
                                           signal_split=signal_split, use_inv=use_inv)
    val_set = create_datasets_test_Exact(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                         train_stats=train_stats, dataset_name=dataset_name,
                                         signal_split=signal_split, use_inv=use_inv)
    test_set = create_datasets_test_Exact(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                          train_stats=train_stats, dataset_name=dataset_name,
                                          signal_split=signal_split, use_inv=use_inv)

    return trn_ds, train_set, val_set, test_set


def concat_data_Exact(included_idx, data, label=None, core_name=None, inv=None,
                      signal_split=False, dataset_name=None, use_inv=True):
    """ Concatenate data from different cores specified by 'included_idx' """
    core_counter = 0
    data_c, label_c, core_name_c, inv_c, core_len = [], [], [], [], []
    for i in range(len(data)):
        if included_idx[i] or (not use_inv):
            data_c.append(data[i])
            core_len.append(np.repeat(core_counter, data[i].shape[0]))
            core_counter += 1
            inv_c.append(np.repeat(inv[i], data[i].shape[0]))
            if not signal_split:
                label_c.append(np.repeat(label[i], data[i].shape[0]))
                temp = np.tile(core_name[i], data[i].shape[0])
                core_name_c.append(temp.reshape((data[i].shape[0], 1)))

    data_c = np.concatenate(data_c).astype('float32')
    data_c = fix_dim(data_c, dataset_name)
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
    return data_c, label_c, core_name_c, inv_out


def create_datasets_test_Exact(data_file, state, dataset_name, norm='Min_Max', min_inv=0.4, augno=0,
                               input_data=None, inv_state='normal', return_array=False, train_stats=None,
                               signal_split=False, use_inv=True):
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

    corelen = []
    target_test = []
    name_tst = []

    # Working with BK dataset
    bags_test = []
    sig_inv_test = []
    for i in range(len(data_test)):
        if included_idx[i] or (not use_inv):
            bags_test.append(data_test[i])
            corelen.append(data_test[i].shape[0])
            if not signal_split:
                target_test.append(np.repeat(label_test[i], data_test[i].shape[0]))
                temp = np.tile(CoreN_test[i], data_test[i].shape[0])
                name_tst.append(temp.reshape((data_test[i].shape[0], 1)))
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
    gs_bk = gs_bk[included_idx]
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

    tst_ds = TensorDataset(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           torch.tensor(sig_inv_test, dtype=torch.float32).unsqueeze(1)) if use_inv else \
             TensorDataset(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8))
    return tst_ds, corelen, label_inv_out, patient_id_bk, gs_bk, None, true_label

def fix_dim(data, dataset_name):
    if '2D' in dataset_name:
        tmp = np.swapaxes(data, 2, 3)
        # return np.swapaxes(tmp.reshape((tmp.shape[0], tmp.shape[1], -1)),1,2)
        # return tmp.reshape((tmp.shape[0], tmp.shape[1], -1))
        return tmp.reshape((tmp.shape[0], 1, tmp.shape[1], -1))
    else:
        return data[:, np.newaxis, :]

def manual_balance(data_c, label_c, core_name_c, inv_out):
    no_cancer = (label_c[:,1]==1).sum()
    cancer_ind = np.where(label_c[:, 1] == 1)
    benign_ind = np.where(label_c[:, 0] == 1)

    benign_ind = sk.shuffle(benign_ind[0], random_state=0)
    benign_ind = benign_ind[:no_cancer]
    ind = np.concatenate((benign_ind,cancer_ind[0])).astype(int)

    data_c = data_c[ind, ...]
    label_c = label_c[ind, ...]
    core_name_c = core_name_c[ind, ...]
    inv_out = inv_out[ind]
    return data_c, label_c, core_name_c, inv_out