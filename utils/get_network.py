from torchinfo import summary
from networks import *


def _construct_network(device, opt, backbone, num_class=None, num_positions=None):
    """
    Create networks
    :param device:
    :param opt:
    :param backbone:
    :param num_positions:
    :return:
    """

    def _get_network():
        return get_network(backbone, device, opt.input_channels[0],
                           num_class if num_class is not None else opt.tasks_num_class[0],
                           num_blocks=opt.arch.num_blocks,
                           out_channels=opt.arch.out_channels,
                           mid_channels=opt.arch.mid_channels,
                           num_positions=opt.arch.num_positions if num_positions is None else num_positions,
                           self_train=opt.self_train,
                           variational=opt.variational,
                           verbose=True)

    return _get_network


def _construct_classifier(device, opt, num_positions=None):
    """
    Create classifiers
    :param device:
    :param opt:
    :param num_positions:
    :return:
    """

    def _get_classifier(in_channels):
        """in_channels is needed, can be obtained by running the backbone once"""
        return get_classifier(device, in_channels,
                              # in_channels // 2,
                              1024,
                              opt.tasks_num_class[0],
                              num_positions=opt.arch.num_positions if num_positions is None else num_positions)

    return _get_classifier


def construct_network(device, opt):
    """
    Create one network for vanilla training and two networks for coteaching
    :param device:
    :param opt:
    :return:
    """
    if opt.is_eval:
        return _construct_network(device, opt, opt.backbone, opt.tasks_num_class[0])

    num_class = 1 if opt.self_train else opt.tasks_num_class[0]
    num_class += 1 if opt.loss_name == 'abc' else 0

    if 'coteaching' in opt.model_name:
        backbones = opt.backbone
        if isinstance(opt.backbone, str):
            backbones = [opt.backbone, ] * 2
        elif isinstance(opt.backbone, list) and (len(opt.backbone) == 1):
            backbones = opt.backbone * 2
        # Create two networks, in which the first one does not have location encoder
        networks = []
        # for (num_positions, backbone) in zip([opt.arch.num_positions, opt.arch.num_positions], backbones):
        for (num_positions, backbone) in zip([0, 0], backbones):
            networks.append(_construct_network(device, opt, backbone, num_class, num_positions))
        return networks
    return _construct_network(device, opt, opt.backbone, num_class)


def construct_classifier(device, opt):
    """
    Create one classifier for vanilla training and two classifiers for coteaching when self-training is used
    :param device:
    :param opt:
    :return:
    """
    if 'coteaching' in opt.model_name:
        clf = [_construct_classifier(device, opt, num_positions) for num_positions in
               [opt.arch.num_positions, opt.arch.num_positions]]
        return clf
    return _construct_classifier(device, opt, opt.arch.num_positions)


def get_network(backbone, device, in_channels, nb_class, num_positions=8,
                verbose=False, self_train=False, num_blocks=3,
                out_channels=30, mid_channels=32, variational=False,
                **kwargs):
    backbone = backbone.lower()
    if backbone == 'simconv4':
        from self_time.model.model_backbone import SimConv4
        net = SimConv4(in_channels, is_classifier=True, nb_class=nb_class, num_positions=num_positions,
                       self_train=self_train)
    elif 'inception' in backbone:
        if variational:
            raise NotImplemented('Variational version of inception is not implemented yet')

        _net = InceptionModel if not variational else InceptionModelVariational
        net = _net(num_blocks, in_channels, out_channels=out_channels,
                   bottleneck_channels=32, kernel_sizes=15, use_residuals=True,
                   num_pred_classes=nb_class, self_train=self_train, num_positions=num_positions)
    elif backbone == 'inception_time':
        net = InceptionTime(c_in=in_channels, c_out=nb_class, bottleneck=16, ks=40, nb_filters=16,
                            residual=True, depth=6)
    elif backbone == 'resnet_ucr':
        net = ResNet(in_channels, mid_channels=mid_channels,
                     num_pred_classes=nb_class, num_positions=num_positions)
    elif backbone == 'resnet':
        _net = resnet20 if not variational else resnet20_variational
        net = _net(num_classes=nb_class, in_channels=in_channels)
    elif backbone == 'fnet':
        net = FNet(dim=200, depth=5, mlp_dim=32, dropout=.5, num_pred_classes=nb_class, num_positions=num_positions)
    else:
        # This network already has a positional encoder
        net = Classifier_3L(in_channels, nb_class)
        if verbose:
            summarize_net(net, in_channels, num_positions)
        return net.to(device)

    # Wrap the feature extractor with the position encoding network
    # net = PosEncoder(net, num_positions=num_positions, variational=variational)
    if verbose:
        summarize_net(net, in_channels, num_positions)
    return net.to(device)


def summarize_net(net, in_channels, num_positions):
    input_size = [(2, in_channels, 200), ]
    if num_positions > 0:
        input_size = [input_size, (2, num_positions)]
    summary(net, input_size=input_size)


def get_classifier(device, in_channels, out_channels, nb_class, num_positions):
    """

    :param device:
    :param in_channels:
    :param out_channels:
    :param nb_class:
    :param num_positions:
    :return:
    """
    clf = Classifier(in_channels, out_channels, nb_class, num_positions).cuda(device)
    return clf
