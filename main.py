from utils import *
from utils.dataloader import create_loader, create_loaders_test
from utils.dataset import create_datasets_v1 as create_datasets, create_datasets_test


def train(opt):
    writer = setup_tensorboard(opt)

    # Datasets / Dataloader
    trn_ds, train_set, val_set, test_set = create_datasets(
        '/'.join([opt.data_source.data_root, opt.data_source.train_set]),
        norm=opt.normalize_input, aug_type=opt.aug_type, min_inv=opt.min_inv, n_views=opt.train.n_views)
    trn_dl = create_loader(trn_ds, bs=opt.train_batch_size, jobs=opt.num_workers, add_sampler=True)
    opt.num_samples = {'train': len(trn_ds), 'val': len(val_set[0]), 'test': len(test_set[0])}

    # Setup models and training strategy
    # ToDo: Multiple GPUS
    device = torch.device(f'cuda:{opt.gpus_id[0]}') if torch.cuda.is_available() else 'cpu'
    network = construct_network(device, opt)
    model = get_model(opt, network, device, 'train',
                      classifier=construct_classifier(device, opt) if opt.self_train else None)

    best_acc = 0.
    print('Start model training')
    for epoch in range(opt.n_epochs):
        # Employ semi-supervised learning after forget-rate reaches the peak value
        # trn_ds.n_views = opt.train.n_views if epoch >= opt.train.coteaching.num_gradual else 1

        # Training
        model.train(epoch, trn_dl, writer=writer)

        if (opt.train.val_interval > 0) and ((epoch + 1) % opt.train.val_interval == 0):
            acc, _ = evaluate(opt, model, val_set, epoch, set_name='Val', writer=writer)

            if acc >= best_acc:  # check validation accuracy
                best_acc = acc
                model.save(opt.paths.checkpoint_dir, 'best')
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')

        # Periodical testing
        if (opt.test.test_interval > 0) and ((epoch + 1) % opt.test.test_interval == 0):
            evaluate(opt, model, test_set, epoch, set_name='Test', writer=writer)
            # testing on training set
            # _, trn_ds.inv_pred = evaluate(opt, model, train_set, epoch, set_name='Train', writer=writer)
            evaluate(opt, model, train_set, epoch, set_name='Train', writer=writer)

    model.save(opt.paths.checkpoint_dir, 'final')
    print('Done training!')


@eval_mode
def evaluate(opt, model=None, dataset_test=None, current_epoch=None, set_name='Test', writer=None,
             checkpoint_dir=None, checkpoint_prefix=''):
    """

    :param checkpoint_prefix:
    :param checkpoint_dir:
    :param writer: SummaryWriter object for logging on TensorboardX
    :param opt: output of 'read_yaml' or munchify(dict)
    :param model: the "model" passed in this argument should have their weights loaded already,
    otherwise, checkpoint_dir & checkpoint_prefix needs to be specified
    :param dataset_test:
    :param current_epoch:
    :param set_name:
    :return:
    """
    # Setup
    device = torch.device(f'cuda:{opt.gpus_id[0]}' if torch.cuda.is_available() else 'cpu')
    state = 'val'

    # Construct model
    if model is None:  # For standalone evaluation
        network = construct_network(device, opt)
        model = get_model(opt, network, device, 'eval')
        model.load(opt.paths.checkpoint_dir, opt.test.which_iter)
    else:  # For periodically testing
        if checkpoint_dir is not None:
            model.load(checkpoint_dir, checkpoint_prefix)

    # Load test set
    if dataset_test is None:  # For standalone evaluation
        # train_stat is missing currently, the evaluation perhaps will be wrong
        datasets, core_len, true_involvement, patient_id_bk, gs_bk, roi_coors = create_datasets_test(
            '/'.join([opt.data_source.data_root, opt.data_source.test_set]),
            min_inv=0.4, state=state, norm=opt.normalize_input)
    else:  # For periodically testing
        datasets, core_len, true_involvement, patient_id_bk, gs_bk, roi_coors = dataset_test

    # Create dataloader to test data
    tst_dl = create_loaders_test(datasets, bs=opt.test_batch_size, jobs=opt.num_workers)

    # Evaluation
    predictions, ood_scores, acc_s = model.eval(tst_dl, net_index=1)

    # Infer core-wise predictions
    predicted_involvement, ood, prediction_maps = infer_core_wise(predictions, core_len, roi_coors, ood_scores)

    # Calculating & logging metrics
    scores = {'acc_s': acc_s}
    scores = compute_metrics(predicted_involvement, true_involvement,
                             current_epoch=current_epoch, verbose=True, scores=scores)

    # import pylab as plt
    # heatmaps_dir = opt.paths.result_dir + f'_heatmaps/{state}'
    # os.makedirs(heatmaps_dir, exist_ok=True)
    # for i, pm in enumerate(prediction_maps):
    #     plt.imsave(f'{heatmaps_dir }/{i}_{true_involvement[i]:.2f}.png', pm, vmin=0, vmax=1, cmap='gray')

    net_interpretation(predicted_involvement, patient_id_bk,
                       true_involvement, gs_bk, opt.paths.result_dir,
                       ood=ood,
                       current_epoch=current_epoch, set_name=set_name, writer=writer, scores=scores)

    if set_name.lower() == 'train':
        predicted_involvement = np.concatenate(
            [np.repeat(pred_inv, cl) for (pred_inv, cl) in zip(predicted_involvement, core_len)])
    # Tensorboard logging
    if writer:
        for score_name, score in zip(scores.keys(), scores.values()):
            writer.add_scalar(f'{set_name}/{score_name.upper()}', score, current_epoch)

    return scores['acc'], predicted_involvement


def main():
    # read the yaml
    opt = setup_directories(read_yaml(verbose=False))
    fix_random_seed(opt.seed)

    # Training/Test
    if opt.eval:
        evaluate(opt)
    else:
        train(opt)


if __name__ == '__main__':
    main()
