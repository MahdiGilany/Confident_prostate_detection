import os
import pickle
from os.path import join as pjoin


def load_pkl(filename='BK_RF_P1_140_balance__20210203-175808'):
    filename = pjoin('../files', filename + '.pkl')
    filename = filename if os.path.exists(filename) else filename.replace('../', '')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
