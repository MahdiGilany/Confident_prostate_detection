import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

from functools import partial
from skimage.morphology import binary_erosion

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
    rois = [((binary_erosion(c.wp[0]) - c.roi[0]) > 0).astype('uint8') for c in cores]
    outputs = {
        'rf': [c.rf[:, roi == 1].T for c, roi in zip(cores, rois)],
        'pid': [int(c.patient_id) for c in cores],
        'cid': [int(c.core_id) for c in cores]
    }
    return outputs


def create_dataset(output_dir, output_filename):
    # from utils.query_metadata import query_patient_info, open_connection, close_connection
    # from utils.misc import Logger, save_pickle, load_pickle
    # from utils.cores import load_cores_h5py as _load_cores_h5py
    # sys.stdout = Logger(f'{os.path.basename(__file__)[:-3]}')

    input_data = {}
    os.makedirs(output_dir, exist_ok=True)

    with open('../metadata/matched_tmi_cores_idx.pkl', 'rb') as fp:
        core_indices = pickle.load(fp)

    ci = core_indices['train']
    rf = []
    _extract = partial(extract, ci)
    count, parts = 0, 6
    length = len(ci.keys()) // parts
    for i, patient_id in tqdm(enumerate(ci.keys()), total=len(ci.keys())):
        outputs = _extract(patient_id)
        for k in ['rf', ]:
            eval(f'{k}.extend(outputs["{k}"])')
            if ((i % length) == 0) or (i == len(ci.keys()) - 1):
                np.save(os.path.join(output_dir, output_filename.replace('.npy', f'part_{count}.npy')),
                        np.concatenate(rf, axis=0))
                rf = []
                count += 1

    # np.save(os.path.join(output_dir, output_filename), np.concatenate(rf, axis=0))
    # input_data[f'data'] = rf
    # input_data[f'PatientId'] = np.array(pid, dtype=int)
    # input_data[f'CoreId'] = np.array(cid, dtype=int)

    # save_pickle(input_data, os.path.join(output_dir, output_filename))


def load_datasets(dir_name, file_name):
    from glob import glob
    import time
    files = glob(os.path.join(dir_name, file_name).replace('.npy', '_part*.npy'))
    files.sort()
    tic = time.time()
    d = []
    for file in files:
        print(file)
        d.append(np.load(file))
        print(d[0].shape)
        # break
    print(np.concatenate(d, axis=0).shape)
    print(time.time() - tic)


if __name__ == '__main__':
    dirname = '../datasets/unlabelled'
    # dirname = '/raid/home/minht/projects/prostate_teus/ProstateCancerClassificationV1/datasets/unlabelled'
    filename = 'BK_RF_P1_140_balance__20210203-175808_unsup.npy'

    create_dataset(dirname, filename)
    # load_datasets(dirname, filename)
    print('abc')
