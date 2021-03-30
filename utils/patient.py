import numpy as np


class Patient:
    def __init__(self, patient_id: int, core_id: int, rf: np.ndarray, cancer: int, gs: int = None,
                 core_name: str = '', d_set: str = None):
        self.patient_id = patient_id
        self.core_id = core_id
        self.rf = rf
        self.cancer = cancer
        self.gs = gs
        self.corr_name = core_name
        self.d_set = d_set

    def __repr__(self):
        return f'Patient {self.patient_id}, core {self.core_id}: {"benign" if self.cancer == 0 else "cancer"} ' \
               f'[{self.d_set}]'
