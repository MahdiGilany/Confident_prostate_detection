import os
import pickle
from os.path import join as pjoin

import numpy as np

from self_time.dataloader.ucr2018 import preprocess


def load_pkl(filename='BK_RF_P1_140_balance__20210203-175808', folder='../files'):
    """
    Load TeUS pickle data from file
    :param filename: the original matlab filename
    :param folder: the folder containing the pickle file
    :return: TeUS data
    """
    filename = pjoin(folder, filename + '.pkl')
    filename = filename if os.path.exists(filename) else filename.replace('../', '')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def from_pickle_to_self_time(include_cancer=False):
    """
    Convert TeUS data loaded by pickle to arrays with dimensionality that matches with
    the requirement of 'self-time' training
    :param include_cancer: cancerous cores is discarded by default in the training.
    Set this flag to true to train on both benign and malignant cores.
    :return: a list of arrays, similar to outputs of 'load_ucr2018' in 'urc2018.py'
    """
    data = load_pkl()  # load the TeUS data stored in the pickle format
    x, y = {}, {}
    sets = list(data.keys())
    nb_class = 2
    for s in sets:
        count = 0  # record the number of selected cores

        # skip all the benign cores if specified
        skip_cancer = True if ((s == 'train') and not include_cancer) else False

        # lists of data and labels
        x[s], y[s] = [], []

        # loop through each core
        for core in data[s]:
            if skip_cancer and core.cancer:  # the core is malignant
                continue
            x[s].append(core.rf)
            y[s].append([int(core.cancer), ] * len(x[s][-1]))
            count += 1

        # concatenate all of the time-series from all cores
        x[s] = np.concatenate(x[s]).astype('float32')[..., np.newaxis]
        y[s] = np.concatenate(y[s]).astype('uint8')
        print(f'{s}: {len(data[s])} cores', x[s].shape, x[s].dtype, y[s].shape)

    # data preprocessing
    x['train'], x['val'], x['test'] = preprocess(x['train'], x['val'], x['test'])

    return x['train'], y['train'], x['val'], y['val'], x['test'], y['test'], nb_class, []


if __name__ == '__main__':
    from_pickle_to_self_time()
