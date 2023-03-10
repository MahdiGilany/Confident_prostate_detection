import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, average_precision_score,
                             confusion_matrix, balanced_accuracy_score)


def cm_score(cfs_mtx):
    """Infer scores from confusion matrix. Implemented for using with 'compute_metrics'"""
    try:
        tn, fp, fn, tp = cfs_mtx.ravel()
    except:
        tn, fp, fn, tp = 0, 1, 1, 0
        print("exception occured!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def sen(*args):
        return tp / (tp + fn)  # sensitivity

    def spe(*args):
        return tn / (tn + fp)  # specificity

    def pre(*args):
        return tp / (tp + fp)  # precision

    def acc(*args):
        return (tp + tn) / (tp + tn + fp + fn)  # accuracy

    def wrapper(metric_name):
        if metric_name == 'sen':
            return sen
        elif metric_name == 'spe':
            return spe
        elif metric_name == 'pre':
            return pre
        elif metric_name == 'acc':
            return acc

    return wrapper


def get_metrics(cfs_mtx=None):
    metrics = {
        'auc': roc_auc_score,
        'auprc': average_precision_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef,
        'acc_b': balanced_accuracy_score,
    }
    if cfs_mtx is not None:
        for k in ['sen', 'spe', 'pre', 'acc']:
            metrics.update({k: cfs_mtx(k)})
    return metrics


def compute_metrics(predicted_involvement, true_involvement,
                    metric_list=('auc', 'auprc', 'f1', 'sen', 'spe', 'acc', 'acc_b'),
                    current_epoch=None, verbose=False, scores=None, threshold=0.5, edl=False) -> dict:
    predicted_involvement_nounc = predicted_involvement[0]
    pred_inv_wNan = predicted_involvement[2]
    predicted_involvement_unc = np.array([pred_inv_wNan[i] for i in range(len(pred_inv_wNan))
                                          if not np.isnan(pred_inv_wNan[i])])

    core_predictions = np.array([item > threshold for item in predicted_involvement_nounc])
    core_labels = np.array([item > 0 for item in true_involvement])

    cfs_mtx = cm_score(confusion_matrix(core_labels, core_predictions))  # tn, fp, fn, tp
    metrics = get_metrics(cfs_mtx)

    scores = {} if scores is None else scores
    for metric in metric_list:
        scores[metric] = metrics[metric](core_labels, core_predictions)
    scores['corr'] = np.corrcoef(predicted_involvement_nounc, true_involvement)[0, 1]
    scores['mae'] = (np.abs(predicted_involvement_nounc - true_involvement)).sum()
    scores['auc'] = metrics['auc'](core_labels, predicted_involvement_nounc)


    if edl:
        core_predictions_unc = np.array([item > threshold for item in predicted_involvement_unc])
        core_labels_unc = np.array([core_labels[i] for i in range(len(pred_inv_wNan))
                                    if not np.isnan(pred_inv_wNan[i])])
        true_involvement_unc = np.array([true_involvement[i] for i in range(len(pred_inv_wNan))
                                    if not np.isnan(pred_inv_wNan[i])])
        cfs_mtx_unc = cm_score(confusion_matrix(core_labels_unc, core_predictions_unc))
        metrics_unc = get_metrics(cfs_mtx_unc)
        for metric in ['acc_b', 'sen', 'spe']:
            scores[metric+'-uncrtan'] = metrics_unc[metric](core_labels_unc, core_predictions_unc)
        scores['corr-uncrtan'] = np.corrcoef(predicted_involvement_unc, true_involvement_unc)[0, 1]
        try:
            scores['auc-uncrtan'] = metrics_unc['auc'](core_labels_unc, predicted_involvement_unc)
        except:
            scores['auc-uncrtan'] = np.nan

    # andlabels = np.logical_and(predictions, labels)
    # norLabels = len(np.where(predictions + labels == 0)[0])
    # acc = (np.sum(andlabels) + norLabels) / len(labels)
    # sen = np.sum(andlabels) / np.sum(labels)
    # spe = norLabels / (len(labels) - np.sum(labels))
    # metric_names = ['AUC', 'F1', 'MCC', 'SEN', 'SPE', 'ACC', 'ACC_s']
    # metric_values = [auc, f1, mcc, sen, spe, acc, acc_s]
    # df = pd.DataFrame([metric_values], columns=metric_names,
    #                   index=[f'Epoch = {current_epoch if current_epoch else "Best"}'])
    if verbose:
        df = pd.DataFrame([scores.values()], columns=[_.upper() for _ in scores.keys()],
                          index=[f'Epoch = {current_epoch if current_epoch else "Best"}'])
        print(df.round(3))
    return scores


if __name__ == '__main__':
    y_true = [0, 1, 1, 1, 0]
    y_pred = [1, 0, 1, 1, 0]
    target_names = ['benign', 'cancer']
    print(confusion_matrix(y_true, y_pred))
