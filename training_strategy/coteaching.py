from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from cleanlab.coteaching import initialize_lr_scheduler, adjust_learning_rate, forget_rate_scheduler

from loss_functions.coteaching_loss import get_loss_coteaching
from loss_functions import (
    avu_utils as util,
    avu_loss,
)
from .vanilla import Model
from utils.novograd import NovoGrad
from loss_functions.simclr import info_nce_loss


class CoTeaching(Model):
    def __init__(self, network: list, device, num_class, mode, aug_type='none',
                 loss_name='gce', use_plus=False, num_positions=8, *args, **kwargs):
        super(CoTeaching, self).__init__(device, num_class, mode, aug_type)
        self.net1 = network[0]()
        self.net2 = network[1]()
        self.optimizer1, self.optimizer2, self.scheduler1, self.scheduler2 = [None, ] * 4
        self.forget_rate_schedule = None
        self.loss_func = get_loss_coteaching(loss_name, self.num_class, use_plus=use_plus, **kwargs)
        self.loss_func_pos = get_loss_coteaching(loss_name, num_classes=num_positions, use_plus=use_plus, **kwargs)
        self.network_list = [self.net1, self.net2]
        self.params_list = [list(self.net1.parameters()), list(self.net2.parameters())]

    class _Decorators:
        @classmethod
        def pre_forward(cls, decorated):
            """Check and return correct inputs before doing supervised forward
            :param decorated: forward_backward method
            """

            def wrapper(model: Model, *args, **kwargs):
                if 'semi_supervised' in kwargs.keys():
                    if kwargs['semi_supervised']:
                        return model.forward_backward_semi_supervised(*args, **kwargs)
                return decorated(model, *args, **kwargs)

            return wrapper

    def init_optimizers(self, lr: float = 1e-3,
                        n_epochs=None, epoch_decay_start=-1, forget_rate=.2,
                        num_gradual=10, exponent=1):
        """
        Create optimizers for networks listed in self.network_list
        :param lr:
        :param lr_scheduler:
        :param weight_decay:
        :param n_epochs:
        :param epoch_decay_start:
        :param forget_rate:
        :param num_gradual:
        :param exponent:
        :return:
        """
        # Set-up learning rate scheduler alpha and betas for Adam Optimizer
        net1, net2 = self.network_list

        # self.optimizer1 = optim.Adam(self.params_list[0], lr=float(lr), amsgrad=True)  # , weight_decay=1e-3)
        # self.optimizer2 = optim.Adam(self.params_list[1], lr=float(lr), amsgrad=True)  # , weight_decay=1e-3)
        self.optimizer1 = NovoGrad(self.params_list[0], lr=float(lr), grad_averaging=True, weight_decay=0.001)
        self.optimizer2 = NovoGrad(self.params_list[1], lr=float(lr), grad_averaging=True, weight_decay=0.001)

        # self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer1, max_lr=lr, steps_per_epoch=554,
        #                                                       epochs=n_epochs)
        # self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer2, max_lr=lr, steps_per_epoch=554,
        #                                                       epochs=n_epochs)
        # self.scheduler1 = torch.optim.lr_scheduler.CyclicLR(self.optimizer1, base_lr=lr/10, max_lr=lr,
        #                                                     step_size_up=1000, cycle_momentum=False,
        #                                                     mode="triangular2")
        # self.scheduler2 = torch.optim.lr_scheduler.CyclicLR(self.optimizer2, base_lr=lr, max_lr=lr*10,
        #                                                     step_size_up=1000, cycle_momentum=False,
        #                                                     mode="triangular2")
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer1, 10, T_mult=1, eta_min=float(lr)/10, last_epoch=-1)  # lr/10
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer2, 10, T_mult=1, eta_min=float(lr)/10, last_epoch=-1)  # lr/10
        if n_epochs is not None:
            self.forget_rate_schedule = forget_rate_scheduler(
                n_epochs, forget_rate, num_gradual, exponent)

    def optimize(self, loss: tuple):
        """

        :param loss:
        :return:
        """
        loss1, loss2 = loss
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        if self.scheduler1:
            self.scheduler1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        if self.scheduler2:
            self.scheduler2.step()

    def save(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        [torch.save(net.state_dict(), f'{checkpoint_dir}/{prefix}{sep}{name}_{i + 1}.pth')
         for (i, net) in enumerate(self.network_list)]

    def load(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        [net.load_state_dict(torch.load(f'{checkpoint_dir}/{prefix}{sep}{name}_{i + 1}.pth'))
         for (i, net) in enumerate(self.network_list)]

    def infer(self, x_raw, positions=None, mode='train'):
        if mode == 'train':
            return self.net1(x_raw, positions), self.net2(x_raw, positions)
        # mode == test, torch.no_grad() was called outside
        return self.net1(x_raw, positions)

    @_Decorators.pre_forward
    def forward_backward(self, x_raw, n_batch, labels, *args, **kwargs):
        out1, out2 = self.infer(x_raw, n_batch)
        loss1, loss2, ind = self.compute_loss(out1, out2, labels, self.loss_func, **kwargs)
        self.optimize((loss1, loss2))
        return out1, out2, loss1, loss2, {'ind': ind}

    def forward_backward_semi_supervised(self, x_raw, n_batch, labels, *args, **kwargs):

        n_views = kwargs['n_views']

        idx_sup = np.arange(0, len(x_raw), n_views)
        out1, out2 = self.infer(x_raw[idx_sup], n_batch[idx_sup])
        loss1, loss2, ind = self.loss_func(out1, out2, torch.argmax(labels[idx_sup], dim=1), **kwargs)

        if n_views > 1:
            idx_unsup = np.ones(len(x_raw))
            idx_unsup[idx_sup] = 0
            # loss1, loss2, ind = self.loss_func(out1[0], out2[0], torch.argmax(labels[idx_sup], dim=1), **kwargs)

            # activation1, activation2 = {}, {}
            # for net, activation in zip(self.network_list, [activation1, activation2]):
            #     net.linear00.register_forward_hook(self.get_activation('linear00', activation))
            # out1_unsup, out2_unsup = self.infer(x_raw, n_batch)
            #
            # criteria_1 = torch.nn.CrossEntropyLoss().to(out1_unsup.device)
            # criteria_2 = torch.nn.CrossEntropyLoss().to(out2_unsup.device)
            # loss1_unsup = criteria_1(*info_nce_loss(activation1['linear00'], out1_unsup.device, n_views))
            # loss2_unsup = criteria_2(*info_nce_loss(activation2['linear00'], out2_unsup.device, n_views))

            out1_unsup, out2_unsup = self.infer(x_raw[idx_unsup == 1], )  # n_batch[idx_unsup == 1]
            loss1_unsup = F.kl_div(F.softmax(out1, dim=1), F.softmax(out1_unsup, dim=1), reduction='batchmean')
            loss2_unsup = F.kl_div(F.softmax(out2, dim=1), F.softmax(out2_unsup, dim=1), reduction='batchmean')

            loss1 = loss1 + loss1_unsup
            loss2 = loss2 + loss2_unsup

        if kwargs['x_unsup'] is not None:
            x_unsup = kwargs['x_unsup'].unsqueeze(1)
            out_unsup_11, out_unsup_12 = self.infer(x_unsup[::2])
            out_unsup_21, out_unsup_22 = self.infer(x_unsup[1::2])
            loss_unsup_1 = F.kl_div(F.softmax(out_unsup_11, dim=1), F.softmax(out_unsup_21, dim=1),
                                    reduction='batchmean')
            loss_unsup_2 = F.kl_div(F.softmax(out_unsup_12, dim=1), F.softmax(out_unsup_22, dim=1),
                                    reduction='batchmean')
            loss1 += loss_unsup_1 * 1e-3
            loss2 += loss_unsup_2 * 1e-3

        self.optimize((loss1, loss2))
        return out1, out2, loss1, loss2, {'ind': ind, 'idx_sup': idx_sup}

    @staticmethod
    def compute_loss(out1, out2, labels, loss_funcs, loss_coefficients=None, **kwargs):
        return loss_funcs(out1, out2, torch.argmax(labels, dim=1), **kwargs)

    @staticmethod
    def on_batch_end(t_epoch, **kwargs):
        t_epoch.set_postfix(**kwargs)

    def train(self, epoch, trn_dl, writer=None):
        """

        :param writer:
        :param epoch:
        :param trn_dl:
        :param semi_supervised:
        :return:
        """
        [_.train() for _ in self.network_list]
        correct, total = 0, 0
        semi_supervised = (trn_dl.dataset.n_views > 1) or (trn_dl.dataset.unsup_data is not None)
        forget_rate = 0 if self.forget_rate_schedule is None else self.forget_rate_schedule[epoch]
        all_ind = {'benign': [], 'cancer': [], 'cancer_ratio': []}
        unc_list = []

        with tqdm(trn_dl, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):
                if self.aug_type != 'none':
                    batch_data = [torch.cat(_, 0).to(self.device) for _ in batch]
                else:
                    batch_data = [_.to(self.device) for _ in batch]

                # Parse batch
                x_raw, y_batch, n_batch, index, *extra_data = batch_data
                if len(extra_data) == 2:
                    loss_weights, x_unsup = extra_data
                else:
                    loss_weights, x_unsup = extra_data[0], None

                # Forward & Backward
                out1, out2, loss1, loss2, extra = self.forward_backward(
                    x_raw.unsqueeze(1), n_batch, y_batch,
                    loss_weights=loss_weights,
                    forget_rate=forget_rate,
                    step=epoch * i,
                    index=index,
                    epoch=epoch,
                    batch_size=trn_dl.batch_size,
                    semi_supervised=semi_supervised,
                    n_views=trn_dl.dataset.n_views,
                    x_unsup=x_unsup,
                )
                if semi_supervised:  #
                    idx_sup = extra['idx_sup']
                    x_raw, n_batch, y_batch = x_raw[idx_sup], n_batch[idx_sup], y_batch[idx_sup]
                total += y_batch.size(0)
                correct += (F.softmax(out1, dim=1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                self.on_batch_end(t_epoch, loss=loss1.item() + loss2.item(), acc=correct / total,
                                  opt1_lr=self.optimizer1.param_groups[0]['lr'],
                                  opt2_lr=self.optimizer2.param_groups[0]['lr'])

                # if 'ind' in extra.keys():
                #     ind = extra['ind']
                #     [all_ind[k].extend(ind[k]) for k in ind.keys()]
                if 'unc' in extra.keys():
                    unc_list.append(extra['unc'])

                total += y_batch.size(0)
                correct += (F.softmax(out1, dim=1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()

        # if writer is not None:
        #     for k in range(2):
        #         writer.add_histogram(f'Benign_{k + 1}',
        #                              np.concatenate([_ for (i, _) in enumerate(all_ind['benign']) if i % 2 == k]),
        #                              epoch
        #                              )
        #         writer.add_histogram(f'Cancer_{k + 1}',
        #                              np.concatenate([_ for (i, _) in enumerate(all_ind['cancer']) if i % 2 == k]),
        #                              epoch
        #                              )
        #         writer.add_histogram(f'cancer_percentage_{k + 1}',
        #                              np.array([_ for (i, _) in enumerate(all_ind['cancer_ratio']) if i % 2 == k]),
        #                              epoch)
        return loss1.item() + loss2.item(), correct / total

    def eval(self, tst_dl, device=None, net_index=1):
        """

        :param net_index: 1 or 2
        :param tst_dl:
        :param device:
        :return: outputs and signal-wise accuracy
        """
        [_.eval() for _ in self.network_list]
        return super(CoTeaching, self).eval(tst_dl, device)


class CoTeachingMultiTask(CoTeaching):
    def __init__(self, *args, **kwargs):
        super(CoTeachingMultiTask, self).__init__(*args, **kwargs)
        self.loss_coefficients = kwargs['opt'].train.loss_coefficients

    @CoTeaching._Decorators.pre_forward
    def forward_backward(self, x_raw, n_batch, labels, *args, **kwargs):
        out1, out2 = self.infer(x_raw, n_batch)

        loss1, loss2, ind = self.compute_loss(out1, out2,
                                              [labels, n_batch],
                                              [self.loss_func, self.loss_func_pos],
                                              loss_coefficients=self.loss_coefficients,
                                              **kwargs)
        self.optimize((loss1, loss2))
        return out1[0], out2[0], loss1, loss2, {'ind': ind}

    def infer(self, x_raw, positions, mode='train'):
        if mode == 'train':
            return self.net1(x_raw, positions), self.net2(x_raw, positions)
        # mode == test, torch.no_grad() was called outside
        return self.net1(x_raw, positions)[0]  # only collect the first output

    @staticmethod
    def compute_loss(out1, out2, labels, loss_funcs, loss_coefficients=None, **kwargs):
        if loss_coefficients is None:
            loss_coefficients = [1. for _ in range(len(out1))]
        loss0, loss1, ind = [], [], []
        for _out1, _out2, _labels, _loss_func in zip(out1, out2, labels, loss_funcs):
            _loss0, _loss1, _ind = _loss_func(_out1, _out2, torch.argmax(_labels, dim=1), **kwargs)
            loss0.append(_loss0), loss1.append(_loss1), ind.append(_ind)

        loss0 = torch.stack([l * coef for (l, coef) in zip(loss0, loss_coefficients)]).sum()
        loss1 = torch.stack([l * coef for (l, coef) in zip(loss1, loss_coefficients)]).sum()
        ind = ind[0]

        return loss0, loss1, ind


class CoTeachingSelfTrain(CoTeaching):
    def __init__(self, *args, **kwargs):
        super(CoTeachingSelfTrain, self).__init__(*args, **kwargs)
        device = args[1]
        ckpt = kwargs['ckpt']

        self.net1.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
        self.net1.eval()
        out = self.net1(torch.rand((2, 1, 200)).cuda(device))
        self.clf1 = kwargs['classifier'][0](in_channels=out.shape[1])
        self.clf2 = kwargs['classifier'][1](in_channels=out.shape[1])

        # Override the network list
        # self.network_list = [self.clf1, self.clf2]
        self.params_list = [self.params_list[0] + list(self.clf1.parameters()),
                            self.params_list[1] + list(self.clf2.parameters())]
        # self.network_list += [self.clf1, self.clf2]

    def infer(self, x_raw, positions, mode='train'):
        if mode == 'train':
            # with torch.no_grad():
            #     out = self.net1(x_raw)
            # return self.clf1(out, positions), self.clf2(out, positions)
            return self.clf1(self.net1(x_raw), positions), self.clf2(self.net2(x_raw), positions)
        # mode == test, torch.no_grad() was called outside
        return self.clf1(self.net1(x_raw), positions)


class CoTeachingUncertaintyAvU(CoTeaching):
    """Co-teaching with uncertainty AvU"""
    opt_h = 1.  # optimal threshold
    beta = 3.
    avu_criterion = avu_loss.AvULoss().cuda()
    unc = 0

    def on_batch_end(self, t_epoch, **kwargs):
        kwargs['unc'] = self.unc
        super(CoTeachingUncertaintyAvU, self).on_batch_end(t_epoch, **kwargs)

    def add_unc_loss(self, loss, out, labels, kl=None):
        labels = labels.argmax(dim=1)

        probs_ = F.softmax(out, dim=1)
        probs = probs_.data.cpu().numpy()
        pred_entropy = util.entropy(probs)
        unc = np.mean(pred_entropy, axis=0)
        preds = np.argmax(probs, axis=-1)
        scaled_kl = kl.data / 1e6 if kl is not None else 0.

        # avu = util.accuracy_vs_uncertainty(np.array(preds),
        #                                    np.array(labels.cpu().data.numpy()),
        #                                    np.array(pred_entropy), self.opt_h)
        #
        # # cross_entropy_loss = criterion(output, target_var)
        #
        elbo_loss = loss + scaled_kl
        avu_loss = self.beta * self.avu_criterion(out, labels, self.opt_h, type=0)
        loss = loss + scaled_kl + avu_loss
        return loss, None, elbo_loss, unc

    def forward_backward(self, x_raw, n_batch, labels, **kwargs):
        # (out1, kl1), (out2, kl2) = self.infer(x_raw, n_batch)
        out1, out2 = self.infer(x_raw, n_batch)

        loss1, loss2, ind = self.loss_func(out1, out2, torch.argmax(labels, dim=1), **kwargs)
        # loss1 = loss1 + kl1.data / 1e6
        # loss2 = loss2 + kl2.data / 1e6

        loss1, avu1, elbo_loss1, unc1 = self.add_unc_loss(loss1, out1, labels, None)
        loss2, avu2, elbo_loss2, unc2 = self.add_unc_loss(loss2, out2, labels, None)

        extra = {'ind': ind}  # 'unc': unc1 + unc2, 'elbo_loss': elbo_loss1 + elbo_loss2}
        self.unc = unc1.item() + unc2.item()

        self.optimize((loss1, loss2))
        return out1, out2, loss1, loss2, extra
