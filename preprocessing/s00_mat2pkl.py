import os
import pickle
import h5py
from functools import partial
from os.path import join as pjoin

import numpy as np

from utils.patient import Patient


def get_field(f, set_name, field_name, index=None):
    field_name = f'{field_name}_{set_name}'
    if index is not None:
        return np.array(f[f[field_name][index][0]]).T
    return f[field_name][()].T[0]


def mat2pkl(filename):
    data_dir = pjoin('/mnt', 'shared_local', 'images', 'ProstateVGH-2', 'Data', 'Dataset', 'InProstate')
    extension = '.mat'
    mat_filename = pjoin(data_dir, filename + extension if extension not in filename else '')
    data = {'train': [], 'val': [], 'test': []}
    with h5py.File(mat_filename, 'r') as f:
        for set_name in data.keys():
            print(set_name)

            _get_field = partial(get_field, f, set_name)

            patient_id = _get_field('PatientId')
            core_id = _get_field('idcore')
            label = _get_field('label')
            for i, (pid, cid, l) in enumerate(zip(patient_id, core_id, label)):
                rf = _get_field('data', index=i)
                p = Patient(pid, cid, rf, l, d_set=set_name)
                data[set_name].append(p)

    save_dir = '../files'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # np.save(pjoin(save_dir, filename), data, allow_pickle=True)
    with open(pjoin(save_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def main():
    filename = 'BK_RF_P1_140_balance__20210203-175808'
    mat2pkl(filename)


if __name__ == '__main__':
    main()
