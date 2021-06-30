import os
import h5py
import pickle

import numpy as np
from tqdm import tqdm

from utils.cores import load_cores_h5py
from utils.query_metadata import query_patient_info, open_connection, close_connection


def save_core_h5py(core, pth):
    with h5py.File(f'{pth}/core{core.core_id}.h5', 'w') as hf:
        hf.create_dataset('rf', data=core.rf, )
        hf.create_dataset('wp', data=core.wp.astype('uint8'))
        hf.create_dataset('roi', data=core.roi.astype('uint8'))
        hf.create_dataset('metadata', data=str(core.metadata))


def main():
    set_names = ['train', 'val', 'test']
    pth = '/media/minh/My Passport/workspace/TeUS/ProstateVGH-2/Data/'

    with open('../metadata/matched_tmi_cores_idx.pkl', 'rb') as fp:
        core_indices = pickle.load(fp)
    cursor = open_connection()

    for set_name in set_names:
        ci = core_indices[set_name]

        for patient_id in tqdm(ci.keys(), desc=set_name, total=len(ci.keys())):
            output_dir = f"{pth}/h5py/Patient{patient_id:03d}/"
            os.makedirs(output_dir, exist_ok=True)

            cores = load_cores_h5py(patient_id, pth, skip_timer=True, suffix='_v0')
            metadata = query_patient_info(patient_id, cursor)
            for core in cores:
                cid = core.core_id
                idx = np.argwhere(np.array(metadata['CoreId']).T[0] == cid)[0][0]
                core.inv = float(metadata['CalculatedInvolvement'][idx][0])
                core.gs = metadata['PrimarySecondary'][idx][0]
                core.metadata['CalculatedInvolvement'] = core.inv
                core.metadata['PrimarySecondary'] = core.gs
                core.metadata['CoreName'] = metadata['CoreName'][idx][0]
                save_core_h5py(core, output_dir)

    close_connection(cursor)
    print('Done')
    return


if __name__ == '__main__':
    main()
