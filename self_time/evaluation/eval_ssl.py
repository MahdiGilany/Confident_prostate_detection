# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from tqdm import tqdm

import torch
import torch.utils.data as data

import self_time.utils.transforms as transforms
from self_time.model.model_backbone import SimConv4
from self_time.dataloader.time_series import TsLoader
from self_time.optim.pytorchtools import EarlyStopping
from utils.cluster import get_features, get_eval_results


def evaluation(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt, opt, ckpt_tosave=None):
    # no augmentations used for linear evaluation
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = TsLoader(data=x_train, targets=y_train, transform=transform_lineval)
    val_set_lineval = TsLoader(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = TsLoader(data=x_test, targets=y_test, transform=transform_lineval)

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=128, shuffle=True,
                                                       num_workers=opt.num_workers)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False,
                                                     num_workers=opt.num_workers)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False,
                                                      num_workers=opt.num_workers)
    signal_length = x_train.shape[1]

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model
    # backbone_lineval = OS_CNN(signal_length).cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, nb_class).cuda()
    # linear_layer = torch.nn.Linear(backbone_lineval.rep_dim, nb_class).cuda()

    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    if ckpt_tosave:
        torch.save(backbone_lineval.state_dict(), ckpt_tosave)

    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=opt.learning_rate_test)
    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience_test, verbose=True)
    best_acc = 0
    best_epoch = 0

    print(f'Linear evaluation [{opt.model_name}]')
    for epoch in range(opt.epochs_test):
        linear_layer.train()
        backbone_lineval.eval()

        acc_trains = list()
        with tqdm(train_loader_lineval, unit="batch") as t_epoch:
            for i, (data, target) in enumerate(t_epoch):
                t_epoch.set_description(f"Epoch {epoch + 1}")

                optimizer.zero_grad()
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                loss = CE(output, target)
                loss.backward()
                optimizer.step()
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_trains.append(accuracy.item())

                t_epoch.set_postfix(loss=loss.item(), acc=sum(acc_trains) / len(acc_trains))
        # print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
        #       .format(epoch + 1, opt.model_name, loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        linear_layer.eval()
        with torch.no_grad():
            with tqdm(val_loader_lineval, unit="batch") as t_epoch:
                for i, (data, target) in enumerate(t_epoch):
                    t_epoch.set_description(f"{'':5s}[Evaluation]")

                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('{:5s}[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            '', epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, None)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return test_acc, best_epoch


def clustering_evaluation(x, y, cid, pid, inv, nb_class, ckpt, opt, ckpt_tosave=None):
    # no augmentations used for linear evaluation
    transform_lineval = transforms.Compose([transforms.ToTensor()])
    data_loader = {}

    for s in x.keys():
        dataset = TsLoader(data=x[s], targets=y[s], transform=transform_lineval)
        data_loader[s] = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=(s != 'train'),
                                                     num_workers=opt.num_workers)
    signal_length = x['train'].shape[1]

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    if ckpt_tosave:
        torch.save(backbone_lineval.state_dict(), ckpt_tosave)
    print(f'Linear evaluation [{opt.model_name}]')
    backbone_lineval.eval()

    # Clustering
    features, labels = {}, {}
    with torch.no_grad():
        for s in data_loader.keys():
            features[s], labels[s] = get_features(backbone_lineval, data_loader[s], set_name=s)

        for s in data_loader.keys():
            if s in ['train', 'val']:
                continue
            pid_unq = np.unique(pid[s])
            for _pid in pid_unq:
                idx = (pid[s] == _pid)
                current_feat, current_label = features[s][idx], labels[s][idx]
                din, dood = get_eval_results(features['train'][labels['train'] == 0],
                                             current_feat[current_label == 0],
                                             current_feat[current_label == 1],
                                             None, opt)

                fig, ax = plt.subplots(1, 1, figsize=(400, 400))
                # ax = ax.flatten()
                ax.hist(din, bins=np.arange(0, 400), alpha=.5)
                ax.hist(dood, bins=np.arange(0, 400), alpha=.5)
                ax.set_title(f'Patient {_pid} [{s}' + f'mean distance: B[{din.mean():.1f}] vs. C[{dood.mean():.1f}]' +
                             f', > 200: B[{np.sum(din > 200)}/{len(din)}] vs. C[{np.sum(dood > 200)}/{len(dood)}]')
                ax.legend(['In-distribution', 'Out-of-distribution'])
                plt.show()

    exit()
