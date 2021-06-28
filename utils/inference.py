import numpy as np


def infer_core_wise(predictions, core_len):
    """
    Infer core-wise predictions based on signal-wise predictions.
    Input must be 1-D array
    :param predictions: signal-wise predictions
    :param core_len: length of each core
    :return:
    """
    counter = 0
    core_feat1, core_res1, core_l1 = [], [], []

    # find a label for each core
    for i in range(len(core_len)):
        temp = predictions[counter:(counter + core_len[i])]
        core_res1.append(temp)
        core_l1.append(np.greater(temp[:, 1], temp[:, 0]).sum() / core_len[i])
        temp = temp[:, 1]
        th = 0.5
        core_feat1.append([temp[temp > th].mean(), len(temp[temp > th]), temp[temp < th].mean(), len(temp[temp < th])])
        counter += core_len[i]
    return core_l1

