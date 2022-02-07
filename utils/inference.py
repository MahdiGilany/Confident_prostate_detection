import numpy as np


def infer_core_wise(predictions, uncertainty, core_len, roi_coors, unc_thr, ood_scores=None):
    """
    Infer core-wise predictions based on signal-wise predictions.
    Input must be 1-D array
    :param predictions: signal-wise predictions
    :param core_len: length of each core
    :param roi_coors: pixel-wise coordinates of ROI area
    :param ood_scores:
    :return:
    """
    counter = 0
    no_uncCores = 0
    core_feat1, core_res1, core_l1, core_l2, core_l3, core_ood = [], [], [], [], [], []
    prediction_maps = []

    # find a label for each core
    for i in range(len(core_len)):
        temp = predictions[counter:(counter + core_len[i])]
        un = uncertainty[counter:(counter + core_len[i])]
        temp2 = temp[un <= unc_thr, ...]
        if len(temp2) <= np.round(0.6*core_len[i]):
            no_uncCores += 1
            core_l3.append(np.nan)
        else:
            core_l3.append(np.greater(temp2[:, 1], temp2[:, 0]).sum() / len(temp2))

            # wighted some of predictions. weighted by uncertainty.
            # temp2 = np.greater(temp[:, 1], temp[:, 0]) * (1. - un)
            # denom = (1. - un).sum()
            # core_l3.append(temp2.sum() / denom)


        # if core_len[i]<=15:
        #     core_l3.append(np.nan)
        # else:
        #     core_l3.append(np.greater(temp[:, 1], temp[:, 0]).sum() / core_len[i])

        core_l1.append(np.greater(temp[:, 1], temp[:, 0]).sum() / core_len[i])
        core_l2.append(temp[:, 1].sum() / core_len[i])
        core_ood.append(ood_scores)
        counter += core_len[i]
        # core_res1.append(temp)
        # temp = temp[:, 1]
        # th = 0.5
        # core_feat1.append([temp[temp > th].mean(), len(temp[temp > th]), temp[temp < th].mean(), len(temp[temp < th])])

        # heatmap = np.zeros((roi_coors[i][0].max()+1, roi_coors[i][1].max()+1))
        # heatmap[roi_coors[i][0], roi_coors[i][1]] = core_res1[i][:, 1]
        # prediction_maps.append(heatmap)
    if no_uncCores!=0:
        print(f'no uncertain cores {no_uncCores}')
    return (core_l1, core_l2, core_l3), core_ood, prediction_maps
