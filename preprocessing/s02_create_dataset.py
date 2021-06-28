import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

from utils.misc import Logger, save_pickle, load_pickle
from utils.cores import load_cores_h5py

set_names = ['train', 'val', 'test']
pth = '/media/minh/My Passport/workspace/TeUS/ProstateVGH-2/Data'
sys.stdout = Logger(f'{os.path.basename(__file__)[:-3]}')


def filter_cores(patient_id, core_indices):
    """

    :param patient_id:
    :param core_indices:
    :return:
    """
    cores = load_cores_h5py(patient_id, pth, core_indices, skip_timer=True)
    return [c for c in cores if c.core_id in core_indices]


def main():
    input_data = {}
    output_dir = '../datasets'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = 'BK_RF_P1_140_balance__20210203-175808_mimic.pkl'

    with open('../metadata/matched_tmi_cores_idx.pkl', 'rb') as fp:
        core_indices = pickle.load(fp)
    for set_name in set_names:

        ci = core_indices[set_name]
        rf, label, gs, inv, pid, roi_coors = [], [], [], [], [], []
        for i, patient_id in tqdm(enumerate(ci.keys()), desc=set_name, total=len(ci.keys())):
            cores = filter_cores(patient_id, ci[patient_id])

            rf.extend([c.rf[:, c.roi[0] == 1].T for c in cores])
            roi_coors.extend([np.where(c.roi[0] == 1) for c in cores])
            label.extend([int(c.label) for c in cores])
            gs.extend([f'{c.gs}' for c in cores])
            inv.extend([c.inv / 100 for c in cores])
            pid.extend([int(c.patient_id) for c in cores])

        input_data[f'data_{set_name}'] = rf
        input_data[f'roi_coors_{set_name}'] = roi_coors
        input_data[f'label_{set_name}'] = np.array(label, dtype='float32')
        input_data[f'inv_{set_name}'] = np.array(inv, dtype='float32')
        input_data[f'GS_{set_name}'] = gs
        input_data[f'PatientId_{set_name}'] = np.array(pid, dtype=int)
        input_data[f'corename_{set_name}'] = np.zeros((len(rf), 8), dtype='float32')

    save_pickle(input_data, os.path.join(output_dir, output_filename))


if __name__ == '__main__':
    main()
    from time import time

    tic = time()
    input_data2 = load_pickle('../datasets/BK_RF_P1_140_balance__20210203-175808_mimic.pkl')
    print(input_data2)
    print(time() - tic)
