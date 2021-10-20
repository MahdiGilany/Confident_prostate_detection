import scipy.io
import os
import sys

import glob
import re
import random
import json
import numpy as np

from PIL import Image
from tqdm import tqdm


def data_gen(file_list, norm_dir, out_dir, type, id=0):
    if not os.path.exists(out_dir + type):
        os.makedirs(out_dir + type)
    label_dict = dict()
    id = 0

    for i in tqdm(file_list):
        file_name = os.path.splitext(os.path.basename(i))[0]
        # file_name_norm = '_'.join(re.split('_', file_name)[:3]) + '_patch_param.npy'
        # norm_array = np.load(norm_dir + file_name_norm)
        elements = re.split('[- _]', file_name)
        center = elements[0]
        name = elements[0] + '-' + elements[1]
        region = elements[2]
        grade = elements[3]
        current_label = {
            'center': center,
            'name': name,
            'region': region,
            'grade': grade,
        }
        data = scipy.io.loadmat(i)
        patch = data['patch_data']

        for j in range(patch.shape[2]):
            outfile = out_dir + type + '/' + str(id) + '.npy'
            np.save(outfile, patch[:, :, j])
            label_dict[str(id)] = current_label
            # label_dict[str(id)]['norm'] = norm_array[:, :, j].tolist()
            id += 1
            # can be saved as tif image
            # Image.fromarray(patch[:,:,j]).save('test.tif')
        outfile_name = out_dir + type + '_labels.json'

    with open(outfile_name, 'w') as outfile:
        json.dump(label_dict, outfile)
    print("id = " + str(id))
    return outfile_name


def main():
    # generate dataset and total json
    project_dir = 'C:\\Users\\Mahdi\\Desktop\\Summer21\\RA\\Codes\\Minh_Mahdi_mod\\prostate_cancer_classification'
    raw_dir = '\\'.join((project_dir, 'data\\UVA_patches\\'))
    norm_dir = '\\'.join((project_dir, 'norm_parameters\\'))
    out_dir = '\\'.join((project_dir, 'data\\'))

    file_list = glob.glob(raw_dir + "*.mat")
    file_list = [i for i in file_list if "_patches.mat" in i][:1000]

    total_json_path = data_gen(file_list, norm_dir, out_dir, 'patch_data_1000', 0)

if __name__ == '__main__':
    main()
