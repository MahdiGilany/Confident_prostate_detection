import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

from functools import partial

from utils.query_metadata import query_patient_info, open_connection, close_connection
from utils.misc import Logger, save_pickle, load_pickle
from utils.cores import load_cores_h5py as _load_cores_h5py

set_names = ['train', 'val', 'test']
pth = '/media/minh/My Passport/workspace/TeUS/ProstateVGH-2/Data'


def load_cores_h5py(patient_id, core_indices):
    """

    :param patient_id:
    :param core_indices:
    :return:
    """
    cores = _load_cores_h5py(patient_id, pth, core_indices, skip_timer=True)
    return [c for c in cores if c.core_id in core_indices]


def extract(core_indices, patient_id):
    """

    :param core_indices:
    :param patient_id:
    :return:
    """
    cores = load_cores_h5py(patient_id, core_indices[patient_id])
    outputs = {
        'rf': [c.rf[:, c.roi[0] == 1].T for c in cores],
        'roi_coors': [np.where(c.roi[0] == 1) for c in cores],
        'label': [int(c.label) for c in cores],
        'pid': [int(c.patient_id) for c in cores],
        'cid': [int(c.core_id) for c in cores]
    }
    return outputs


def add_metadata(input_dir, input_filename):
    """
    Add metadata to a created dataset
    :return:
    """
    print('Add metadata to dataset file...')
    cursor = open_connection()
    input_data = load_pickle(os.path.join(input_dir, input_filename))

    for set_name in set_names:

        patient_id = input_data[f'PatientId_{set_name}']
        core_id = input_data[f'CoreId_{set_name}']
        inv, gs, loc = [], [], []

        current_pid = -1
        for pid, cid in zip(patient_id, core_id):
            if pid != current_pid:
                metadata = query_patient_info(pid, cursor)

            idx = np.argwhere(np.array(metadata['CoreId']).T[0] == cid)[0][0]

            inv.append(float(metadata['CalculatedInvolvement'][idx][0]))
            gs.append(metadata['PrimarySecondary'][idx][0])
            loc.append(metadata['CoreName'][idx][0])

            current_pid = pid

        input_data[f'inv_{set_name}'] = np.array(input_data[f'inv_{set_name}'], dtype='float32')
        input_data[f'GS_{set_name}'] = gs
        input_data[f'loc_{set_name}'] = loc

    save_pickle(input_data, os.path.join(input_dir, input_filename))

    close_connection(cursor)
    print('Done')
    return


def create_dataset(output_dir, output_filename):
    input_data = {}
    os.makedirs(output_dir, exist_ok=True)

    with open('../metadata/matched_tmi_cores_idx.pkl', 'rb') as fp:
        core_indices = pickle.load(fp)

    for set_name in set_names:
        ci = core_indices[set_name]
        rf, label, pid, cid, roi_coors = [], [], [], [], []
        _extract = partial(extract, ci)
        for patient_id in tqdm(ci.keys(), desc=set_name, total=len(ci.keys())):
            outputs = _extract(patient_id)
            for k in ['rf', 'label', 'pid', 'cid', 'roi_coors']:
                eval(f'{k}.extend(outputs["{k}"])')

        input_data[f'data_{set_name}'] = rf
        input_data[f'roi_coors_{set_name}'] = roi_coors
        input_data[f'label_{set_name}'] = np.array(label, dtype='float32')
        input_data[f'PatientId_{set_name}'] = np.array(pid, dtype=int)
        input_data[f'CoreId_{set_name}'] = np.array(cid, dtype=int)
        input_data[f'corename_{set_name}'] = np.zeros((len(rf), 8), dtype='float32')

    save_pickle(input_data, os.path.join(output_dir, output_filename))


if __name__ == '__main__':
    dirname = '../datasets/'
    filename = 'BK_RF_P1_140_balance__20210203-175808_mimic.pkl'
    sys.stdout = Logger(f'{os.path.basename(__file__)[:-3]}')

    create_dataset(dirname, filename)
    add_metadata(dirname, filename)
