# -*- coding: utf-8 -*-


import datetime
import argparse
import torch
from self_time.evaluation.eval_ssl import clustering_evaluation
from self_time.utils.utils import get_config_from_json
from dataloader.prostate_teus import from_pickle_to_self_time


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--K', type=int, default=16,
                        help='Number of augmentation for each sample')
    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')

    parser.add_argument('--include_cancer', dest='include_cancer', action='store_true')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs_test', type=int, default=400,
                        help='number of test epochs')
    parser.add_argument('--patience_test', type=int, default=100,
                        help='number of training patience')

    # optimization
    parser.add_argument('--learning_rate_test', type=float, default=0.5,
                        help='learning rate')

    # model dataset
    parser.add_argument('--dataset_name', type=str, default='ProstateTeUS',
                        choices=['ProstateTeUS', ])
    parser.add_argument('--ucr_path', type=str, default='./self_time/datasets/',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='SelfTime',
                        choices=['InterSample', 'IntraTemporal', 'SelfTime'], help='choose method')
    parser.add_argument('--config_dir', type=str, default='./self_time/config', help='The Configuration Dir')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import numpy as np

    opt = parse_option()

    Seeds = [0, ]  # 1, 2, 3, 4]
    Runs = range(0, 1)  # range(0, 10, 1)

    exp = 'cluster-evaluation'
    exp_ckpt = 'linear_eval'
    backname = 'best'  # 'last'

    aug1 = 'magnitude_warp'
    aug2 = 'time_warp'

    config_dict = get_config_from_json('{}/{}_config.json'.format(
        opt.config_dir, opt.dataset_name))

    opt.class_type = config_dict['class_type']
    opt.piece_size = config_dict['piece_size']

    if opt.model_name == 'InterSample':
        model_paras = 'none'
    else:
        model_paras = '{}_{}'.format(opt.piece_size, opt.class_type)

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    if backname == 'init':
        log_dir = './results/{}/{}/Random'.format(
            exp, opt.dataset_name)
    else:
        log_dir = './results/{}/{}/{}'.format(
            exp, opt.dataset_name, opt.model_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print_test = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_test)
    print("Dataset\tAcc_mean\tAcc_std\tEpoch_max",
          file=file2print_test)
    file2print_test.flush()

    file2print_detail_test = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail_test)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tRun\tAcc_max\tEpoch_max", file=file2print_detail_test)
    file2print_detail_test.flush()

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        ACCs_run = {}
        MAX_EPOCHs_run = {}
        for run in Runs:

            opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
                exp_ckpt, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
                model_paras, str(seed))

            ckpt = '{}/backbone_{}.tar'.format(opt.ckpt_dir, backname)

            if not os.path.exists(ckpt):
                print('[ERROR] No such ckpt {}'.format(ckpt))
                break

            print('[INFO] Running at:', opt.dataset_name)

            x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ = from_pickle_to_self_time(opt.include_cancer)

            """ Test """
            acc_test, epoch_max_point = clustering_evaluation(
                x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt, opt, None)
