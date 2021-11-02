import os
import re
import torch
import numpy as np
import sklearn.utils as sk
import torch.multiprocessing
from torch.utils.data import TensorDataset
from self_time.optim.pretrain import get_transform
from typing import Union
# torch.multiprocessing.set_sharing_strategy('file_system')
from utils.misc import load_matlab
from utils.misc import load_pickle
from utils.misc import squeeze_Exact
from preprocessing.s02b_create_unsupervised_dataset import load_datasets as load_unlabelled_datasets


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

class Dataset_dynmc(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, label, location, inv=None, dynmc_data_folder='', state=''):
        # self.tensors contains data, label, location and inv
        self.tensors = []
        self.tensors.append(data)
        self.tensors.append(torch.tensor(label, dtype=torch.uint8))
        self.tensors.append(torch.tensor(location, dtype=torch.uint8))
        self.data_folder = dynmc_data_folder
        if inv is not None:
            self.tensors.append(torch.tensor(inv, dtype=torch.float32))

        # for keeping the loaded data
        self.counter = 0
        self.buffer_size = 60 if state=='train' else 30
        self.buffer_loaded_data = []
        self.buffer_loaded_names = []

    def __getitem__(self, index):
        data = self.get_img(index)
        label = self.tensors[1][index,...]
        loc = self.tensors[1][index,...]
        return data, label, loc, torch.tensor(1), torch.tensor(1)

    def __len__(self):
        return self.tensors[1].size(0)

    def get_img(self, index):
        data_name = self.tensors[0][index]
        elements = re.split('_', data_name)

        file_name = '_'.join(elements[:4])
        file_name = file_name + elements[6]

        data = self.load_data(file_name)

        # file_name = '/'.join([self.data_folder, file_name])
        # data = load_matlab(file_name)

        data = data['patch_data']
        data = data[:, :, int(elements[5])-1, int(elements[4])-1]
        return torch.tensor(data[np.newaxis, ...], dtype=torch.float32)

    def load_data(self, file_name):
        for i, name in enumerate(self.buffer_loaded_names):
            if name == file_name:
                return self.buffer_loaded_data[i]

        new_data = load_matlab('/'.join([self.data_folder, file_name]), dynmc=True)
        if self.counter < self.buffer_size:
            self.counter += 1
            self.buffer_loaded_data.append(new_data)
            self.buffer_loaded_names.append(file_name)
        return new_data



# def remove_empty_data(input_data, set_name, p_thr=.2):
#     """
#
#     :param input_data:
#     :param set_name:
#     :param p_thr: threshold of zero-percentage
#     :return:
#     """
#     data = input_data[f"data_{set_name}"]
#     s = [(data[i] == 0).sum() for i in range(len(data))]
#     zero_percentage = [s[i] / np.prod(data[i].shape) for i in range(len(data))]
#     include_idx = np.array([i for i, p in enumerate(zero_percentage) if p < p_thr])
#     if len(include_idx) == 0:
#         return input_data
#     if len(include_idx) == len(data):
#         return input_data
#
#     for k in input_data.keys():
#         if set_name in k:
#             if isinstance(input_data[k], list):
#                 input_data[k] = [_ for i, _ in enumerate(input_data[k]) if i in include_idx]
#             else:
#                 input_data[k] = input_data[k][include_idx]
#     return input_data

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


def preprocess(input_data, p_thr=.2, to_norm=False):
    """
    Remove data points which have percentage of zeros greater than p_thr
    :param input_data:
    :param p_thr:
    :param to_norm:
    :param shffl_patients:
    :return:
    """
    for set_name in ['train', 'val', 'test']:
        if to_norm:
            input_data = normalize(input_data, set_name, to_framemax=False)
    return input_data

#########################################################################################
def create_datasets_Exact_dynmc(dataset_name, data_file, norm=None, min_inv=0.4, aug_type='none', n_views=2,
                          unlabelled_data_file=None, unsup_aug_type=None,
                          to_norm=False, use_inv=True, dynmc_dataroot=''):
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

    print("load matlab dataset")
    input_data = load_matlab(data_file, dynmc=True)
    print("loading done")

    input_data = preprocess(input_data, to_norm=to_norm)

    train_stats = None
    train_set = create_datasets_test_Exact(None, min_inv=min_inv, state='train', norm=norm, input_data=input_data,
                                           train_stats=train_stats, use_inv=use_inv, dynmc_data_folder=dynmc_dataroot)
    val_set = create_datasets_test_Exact(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                         train_stats=train_stats, use_inv=use_inv, dynmc_data_folder=dynmc_dataroot)
    test_set = create_datasets_test_Exact(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                          train_stats=train_stats, use_inv=use_inv, dynmc_data_folder=dynmc_dataroot)

    # return trn_ds, train_set, val_set, test_set
    return None, train_set, val_set, test_set


def concat_data_Exact(included_idx, data, label=None, core_name=None, inv=None, dataset_name=None, use_inv=True):
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
            label_c.append(np.repeat(label[i], data[i].shape[0]))
            temp = np.tile(core_name[i], data[i].shape[0])
            core_name_c.append(temp.reshape((data[i].shape[0], 1)))

    data_c = np.concatenate(data_c).astype('float32')
    # data_c = fix_dim(data_c, dataset_name)
    # import matplotlib.pyplot as plt
    # for i in range(286):
    #     plt.plot(range(15), data_c[15000, 0, i, :],'b')
    # plt.show()
    inv_c = np.concatenate(inv_c)
    inv_out = inv_c if use_inv else None

    label_c = to_categorical(np.concatenate(label_c))
    core_name_c = to_categorical(np.concatenate(core_name_c))

    # manually balance the number of cancers and benigns
    # data_c, label_c, core_name_c, inv_out = manual_balance(data_c, label_c, core_name_c, inv_out)

    # seperating 3 different focal points as channels and using them as 3 independent data
    # label_c = np.repeat(label_c,3, axis=0)
    # core_name_c = np.repeat(core_name_c,3, axis=0)
    # inv_out = np.repeat(inv_out,3, axis=0)

    return data_c, label_c, core_name_c, inv_out


def create_datasets_test_Exact(data_file, state, norm='Min_Max', min_inv=0.4, augno=0,
                               input_data=None, inv_state='normal', return_array=False, train_stats=None, use_inv=True,
                               dynmc_data_folder=''):
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

    # Working with Exact dataset
    bags_test = []
    sig_inv_test = []
    for i in range(len(data_test)):
        if included_idx[i] or (not use_inv):
            data_test_sub = data_test[i][::, ...]
            bags_test.append(data_test_sub)
            corelen.append(data_test_sub.shape[0])
            target_test.append(np.repeat(label_test[i], data_test_sub.shape[0]))
            temp = np.tile(CoreN_test[i], data_test_sub.shape[0])
            name_tst.append(temp.reshape((data_test_sub.shape[0], 1)))
            if label_inv[i] == 0:
                sig_inv_test.append(np.repeat(label_inv[i], data_test_sub.shape[0]))
            elif inv_state == 'uniform':
                dist = abs(label_inv[i] - np.array([0.25, 0.5, 0.75, 1]))
                min_ind = np.argmin(dist)
                temp_inv = abs(torch.rand(data_test_sub.shape[0]) * 0.25) + 0.25 * min_ind
                sig_inv_test.append(temp_inv)
            elif inv_state == 'normal':
                temp_inv = torch.randn(data_test_sub.shape[0]) * np.sqrt(0.1) + label_inv[i]
                sig_inv_test.append(np.abs(temp_inv))
            else:
                sig_inv_test.append(np.repeat(label_inv[i], data_test_sub.shape[0]))

    signal_test = np.concatenate(bags_test)
    sig_inv_test = np.concatenate(sig_inv_test)
    patient_id_bk = patient_id_bk[included_idx]
    gs_bk = np.array(gs_bk)[included_idx]
    label_inv = label_inv[included_idx]
    # label_inv_out = label_inv
    label_inv_out = label_inv if use_inv else None

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

    # seperating 3 different focal points as channels and using them as 3 independent data
    # label_test = np.repeat(label_test, 3, axis=0)
    # name_tst = np.repeat(name_tst, 3, axis=0)
    # sig_inv_test = np.repeat(sig_inv_test, 3, axis=0)
    # corelen = [3 * cl for cl in corelen]

    print("cancer labels shape", label_test[label_test[:,1]>label_test[:,0],:].shape)
    print("data shape", signal_test.shape)

    tst_ds = Dataset_dynmc(signal_test,
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           torch.tensor(sig_inv_test, dtype=torch.float32).unsqueeze(1),
                           dynmc_data_folder=dynmc_data_folder, state=state) if use_inv else \
             Dataset_dynmc(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           dynmc_data_folder=dynmc_data_folder, state=state)
    return tst_ds, corelen, label_inv_out, patient_id_bk, gs_bk, None, true_label

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


