from loss_functions import *


def get_loss_function(loss_name, num_classes, train_loader=None, **kwargs):
    loss_functions = {
        'ce': CrossEntropy(),
        'nce_rce': NCEandRCE(alpha=1., beta=1., num_classes=num_classes),
        'nfl_rce': NFLandRCE(alpha=1., beta=1., gamma=1, num_classes=num_classes),
        'gce': GeneralizedCrossEntropy(num_classes),
        'fl': FocalLoss(gamma=2.),
        'iso': IsoMaxLossSecondPart()
        # 'nlnl': NLNL(train_loader, num_classes)
    }
    opt = kwargs['opt'] if 'opt' in kwargs else None

    if loss_name == 'elr':  # https://github.com/shengliu66/ELR
        assert opt is not None
        loss_functions['elr'] = ELR(opt.num_samples['train'],
                                    num_classes,
                                    alpha=opt.elr_alpha,
                                    beta=opt.elr_beta,
                                    )
    if loss_name == 'abc':  # https://github.com/thulas/dac-label-noise
        assert opt is not None
        for field in ['abstention', 'n_epochs']:
            assert hasattr(opt, field)
        args = opt.abstention
        loss_functions['abc'] = DacLossPid(learn_epochs=args.learn_epochs,
                                           total_epochs=opt.n_epochs,
                                           abst_rate=args.abst_rate,
                                           alpha_final=args.alpha_final,
                                           alpha_init_factor=args.alpha_init_factor,
                                           pid_tunings=list(args.pid_tunings)
                                           )
    return loss_functions[loss_name]
