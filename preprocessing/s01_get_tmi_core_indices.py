import os
import sys
import pickle

import mat73
import numpy as np

from utils.misc import Logger
from utils.cores import load_cores_h5py

set_names = ['train', 'val', 'test']
pth = '/media/minh/My Passport/workspace/TeUS/ProstateVGH-2/Data'
tmi_dataset_file = '/home/minh/PycharmProjects/prostate_cancer_classification/Samareh/Data/' \
                   'BK_RF_P1_140_balance__20210203-175808.mat'
NUM_WORKERS = 12


def get_tmi_patient_ids(tmi_dataset):
    patient_ids = {}
    for s in set_names:
        patient_ids[s] = np.unique(tmi_dataset[f'PatientId_{s}'])
    return patient_ids


def extract_tmi_cores(tmi_dataset, set_name, pid):
    d = tmi_dataset[f"data_{set_name}"]
    patient_ids = tmi_dataset[f"PatientId_{set_name}"]
    tmi_cores = [d[i] for i, _pid in enumerate(patient_ids) if _pid == pid]
    return tmi_cores


def match_cores(tmi_cores, cores):
    """

    :param tmi_cores:
    :param cores:
    :return:
    """
    cores_filtered = []
    cores_len = []
    for tmi_core in tmi_cores:
        core_len = tmi_core.shape[0]
        for core in cores:
            rf = core.rf[:, core.roi[0] == 1]
            if rf.shape[1] == core_len:
                cores_filtered.append(core.core_id)
                cores_len.append(rf.shape[1])
    if len(cores_filtered) != len(tmi_cores):
        unmatched_core_len = [_.shape[0] for _ in tmi_cores if _.shape[0] not in cores_len]
        print(f'{len(unmatched_core_len)} core(s) cannot be located.')
    cores_filtered.sort()
    return cores_filtered


def load_all_cores(patient_ids):
    cores = {}
    for pid in patient_ids:
        cores[pid] = load_cores_h5py(pid, pth)
    return cores


def process(tmi_dataset, set_name, pid):
    """

    :param tmi_dataset:
    :param set_name:
    :param pid:
    :return:
    """
    tmi_cores = extract_tmi_cores(tmi_dataset, set_name, pid)
    cores = load_cores_h5py(pid, pth)
    return match_cores(tmi_cores, cores)


def main():
    core_indices = {}
    tmi_dataset = mat73.loadmat(tmi_dataset_file)
    patient_ids = get_tmi_patient_ids(tmi_dataset)

    for set_name in set_names:
        core_indices[set_name] = {}

        for i, pid in enumerate(patient_ids[set_name]):
            print(' | '.join((set_name, f'{i}/{len(patient_ids[set_name])}', f'{pid}')))
            core_indices[set_name][pid] = process(tmi_dataset, set_name, pid)

    with open('../metadata/matched_tmi_cores_idx.pkl', 'wb') as fp:
        pickle.dump(core_indices, fp)


if __name__ == '__main__':
    sys.stdout = Logger(f'{os.path.basename(__file__)[:-3]}')
    main()

    with open('../metadata/matched_tmi_cores_idx.pkl', 'rb') as fp:
        core_indices = pickle.load(fp)
    print(core_indices)

