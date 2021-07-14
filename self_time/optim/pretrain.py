# -*- coding: utf-8 -*-

import self_time.utils.transforms as transforms
from self_time.dataloader.ucr2018 import *
import torch.utils.data as data
from self_time.model.model_RelationalReasoning import *
from self_time.model.model_backbone import SimConv4
from networks.inception1d import InceptionModel


def get_transform(aug_type, prob=.2):
    """

    :param prob: Transform Probability
    :return:
    """
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=.4, p=prob)  # 0.4
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()
    return transform, tensor_transform


def pretrain_IntraSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class = 2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class = 3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        nb_class = 4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        nb_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        nb_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        nb_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        nb_class = 8

    train_transform, tensor_transform = get_transform(opt.aug_type)

    backbone = SimConv4().cuda()
    model = RelationalReasoning_Intra(backbone, backbone.feature_size, nb_class).cuda()

    train_set = MultiUCR2018_Intra(data=x_train, targets=y_train, K=K,
                                   transform=train_transform, transform_cut=cut_piece,
                                   totensor_transform=tensor_transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_InterSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)
    train_transform = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    backbone = SimConv4().cuda()
    model = RelationalReasoning(backbone, feature_size).cuda()

    train_set = MultiUCR2018(data=x_train, targets=y_train, K=K, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_SelfTime(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class = 2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class = 3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        nb_class = 4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        nb_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        nb_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        nb_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        nb_class = 8

    # backbone = SimConv4().cuda()
    backbone = InceptionModel(3, 1, 30, 12, 15, use_residuals='default', self_train=True)
    model = RelationalReasoning_InterIntra(backbone, nb_class).cuda()

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               num_workers=opt.num_workers,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max
