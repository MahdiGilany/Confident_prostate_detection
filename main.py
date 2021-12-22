import numpy as np

from utils import *
from utils.dataloader import create_loader, create_loaders_test
from utils.dataset import create_datasets_Exact
from utils.dataset_dynmc import create_datasets_Exact_dynmc
from utils.dataset import create_datasets_test_Exact as create_datasets_test


def train(opt):
    writer = setup_tensorboard(opt)

    if opt.data_source.IDloading_dynmc:
        create_datasets = create_datasets_Exact_dynmc
    else:
        create_datasets = create_datasets_Exact

# Datasets / Dataloader
    trn_ds, train_set, val_set, test_set = create_datasets(
        dataset_name=opt.data_source.dataset,
        data_file='/'.join([opt.data_source.data_root, opt.data_source.train_set]),
        unlabelled_data_file='/'.join([opt.data_source.data_root, opt.data_source.unlabelled_set]),
        norm=opt.normalize_input, aug_type=opt.aug_type, aug_prob=opt.aug_prob, min_inv=opt.min_inv, n_views=opt.train.n_views,
        unsup_aug_type=opt.unsup_aug_type, dynmc_dataroot=opt.data_source.dynmc_dataroot,
        split_random_state=opt.split_random_state, val_size=opt.val_size
    )

    trn_ds2 = train_set[0]
    val_ds = val_set[0]
    tst_ds = test_set[0]

    # trn_ds is for training and trn_ds2 is for testing on train set
    trn_dl = create_loader(trn_ds, bs=opt.train_batch_size, jobs=opt.num_workers, add_sampler=True, pin_memory=True)##todo check pin memory
    trn_dl2 = create_loaders_test(trn_ds2, bs=opt.test_batch_size, jobs=opt.num_workers, pin_memory=False)
    val_dl = create_loaders_test(val_ds, bs=opt.test_batch_size, jobs=opt.num_workers, pin_memory=False)
    tst_dl = create_loaders_test(tst_ds, bs=opt.test_batch_size, jobs=opt.num_workers, pin_memory=False)

    train_set = [trn_dl2 if i == 0 else data for i, data in enumerate(train_set)]
    val_set = [val_dl if i == 0 else data for i, data in enumerate(val_set)]
    test_set = [tst_dl if i == 0 else data for i, data in enumerate(test_set)]

    opt.num_samples = {'train': len(train_set[0]), 'val': len(val_set[0]), 'test': len(test_set[0])}
    opt.num_batches = {'train': len(trn_dl)}

    # Setup models and training strategy
    # ToDo: Multiple GPUS
    # device = torch.device(
    #     f'cuda:{",".join([str(_) for _ in range(torch.cuda.device_count())])}') if torch.cuda.is_available() else 'cpu'
    device = torch.device(f'cuda:{opt.gpus_id[0]}' if torch.cuda.is_available() else 'cpu')
    print('Cuda?', device)
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
            _, trn_ds = evaluate(opt, model, train_set, epoch, set_name='Train', writer=writer, trn_ds=trn_ds)

        # Shuffle indices of unlabelled dataset
        # if trn_ds.unsup_data is not None:
        #     np.random.shuffle(trn_ds.unsup_index)

        if trn_ds.label_corrected:
            trn_dl = create_loader(trn_ds, bs=opt.train_batch_size, jobs=opt.num_workers,
                                   add_sampler=True, pin_memory=False)

    model.save(opt.paths.checkpoint_dir, 'final')
    print('Done training!')


@eval_mode
def evaluate(opt, model=None, dataset_test=None, current_epoch=None, set_name='Test', writer=None,
             checkpoint_dir=None, checkpoint_prefix='', trn_ds=None):
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
        print('loading dataset...')
        # train_stat is missing currently, the evaluation perhaps will be wrong
        datasets, core_len, true_involvement, patient_id_bk, gs_bk, roi_coors, true_labels, *ids = create_datasets_test(
            '/'.join([opt.data_source.data_root, opt.data_source.test_set]), dataset_name=opt.data_source.dataset,
            min_inv=0.4, state=state, norm=opt.normalize_input)
        tst_dl = create_loaders_test(datasets, bs=opt.test_batch_size, jobs=opt.num_workers)
    else:  # For periodically testing
        # datasets, core_len, true_involvement, patient_id_bk, gs_bk, roi_coors, *true_labels = dataset_test
        tst_dl, core_len, true_involvement, patient_id_bk, gs_bk, roi_coors, true_labels, *ids = dataset_test

    # Create dataloader to test data
    # tst_dl = create_loaders_test(datasets, bs=opt.test_batch_size, jobs=opt.num_workers)

    # Evaluation
    predictions, ood_scores, acc_s, acc_sb = model.eval(tst_dl, net_index=1)

    # Infer core-wise predictions
    predicted_involvement_thr, predicted_involvement_mean, ood, prediction_maps = infer_core_wise(predictions, core_len, roi_coors, ood_scores)

    # Calculating & logging metrics
    scores = {'acc_s': acc_s, 'acc_sb': acc_sb}
    scores = compute_metrics(predicted_involvement_thr, true_involvement,
                             current_epoch=current_epoch, verbose=True, scores=scores,
                             threshold=opt.core_th)

    # import pylab as plt
    # heatmaps_dir = opt.paths.result_dir + f'_heatmaps/{state}'
    # os.makedirs(heatmaps_dir, exist_ok=True)
    # for i, pm in enumerate(prediction_maps):
    #     plt.imsave(f'{heatmaps_dir }/{i}_{true_involvement[i]:.2f}.png', pm, vmin=0, vmax=1, cmap='gray')

    net_interpretation(predicted_involvement_thr, predicted_involvement_mean, patient_id_bk,
                       true_involvement, gs_bk, opt.paths.result_dir,
                       ood=ood, current_epoch=current_epoch, set_name=set_name,
                       writer=writer, scores=scores, threshold=opt.core_th)

    # if set_name.lower() == 'train':
    #     predicted_involvement = np.concatenate(
    #         [np.repeat(pred_inv, cl) for (pred_inv, cl) in zip(predicted_involvement, core_len)])
    # Tensorboard logging
    if writer:
        for score_name, score in zip(scores.keys(), scores.values()):
            writer.add_scalar(f'{set_name}/{score_name.upper()}', score, current_epoch)

    # correct labels if the difference between predicted and true involvements satisfies the threshold
    if (set_name.lower() == 'train') and (trn_ds is not None) and (current_epoch > opt.epoch_start_correct):
        trn_ds.correct_labels(ids[0], core_len, predictions, true_involvement, predicted_involvement_thr, opt.correcting)

    # return scores['acc'], predicted_involvement
    # return scores['acc_s'], predicted_involvement
    return scores['acc_b'], trn_ds


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
