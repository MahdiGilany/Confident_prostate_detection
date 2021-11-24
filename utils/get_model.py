import torch.cuda
from torch import nn
from training_strategy import *


def get_model(opt, network, device, mode, classifier=None):
    """

    :param classifier:
    :param mode: 'train' or 'eval'
    :param opt: output of 'read_yaml' (utils.misc) or munchify(dict)
    :param network: output of 'construct_network' (utils.misc)
    :param device:
    :return:
    """
    model = None
    if opt.model_name.lower() == 'coteaching':
        # cot = CoTeachingSelfTrain if opt.self_train else CoTeaching
        if opt.self_train:
            cot = CoTeachingSelfTrain
        elif opt.variational:
            cot = CoTeachingUncertaintyAvU
        elif opt.multitask:
            cot = CoTeachingMultiTask
        else:
            cot = CoTeaching
        if mode.lower() == 'train':
            model = cot(
                network, device, opt.tasks_num_class[0], mode, opt.aug_type,
                loss_name=opt.loss_name,
                use_plus=opt.train.coteaching.use_plus,
                relax=opt.train.coteaching.relax,
                classifier=classifier,
                ckpt=opt.paths.self_train_checkpoint,
                opt=opt,  # optional
            )
            model.init_optimizers(opt.lr, opt.n_epochs,
                                  opt.train.lr_scheduler.epoch_decay_start,
                                  opt.train.coteaching.forget_rate,
                                  opt.train.coteaching.num_gradual,
                                  opt.train.coteaching.exponent, n_batches=opt.num_batches['train'])
        else:
            model = cot(network, device, opt.tasks_num_class[0], mode='test', classifier=classifier,
                        opt=opt, epochs=opt.n_epochs,  # optional
                        )
        # if torch.cuda.device_count() > 1:
        #     model.net1 = nn.DataParallel(model.net1).to(device)
        #     model.net2 = nn.DataParallel(model.net2).to(device)
        return model
    elif opt.model_name.lower() in ['vanilla', 'sam']:
        _model = Model if opt.model_name.lower() == 'vanilla' else ModelSam
        model = _model(device, opt.tasks_num_class[0], mode, aug_type=opt.aug_type,
                       network=network, loss_name=opt.loss_name,
                       opt=opt,  # optional
                       )
        model.init_optimizers(opt.lr, n_epochs=opt.n_epochs, )
    return model
