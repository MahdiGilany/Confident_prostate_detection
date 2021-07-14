import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from utils.get_loss_function import *


class Model:
    def __init__(self, device, num_class, mode, aug_type='none', network=None,
                 loss_name=None, *args, **kwargs):
        self.num_class = num_class
        self.device = device
        self.aug_type = aug_type
        self.mode = mode
        self.optimizer, self.scheduler, self.net = [None, ] * 3

        if network:
            self.net = network()
        if loss_name:
            self.loss_func = get_loss_function(loss_name, self.num_class, **kwargs)

    def init_optimizers(self, lr=1e-3, n_epochs=None, *args, **kwargs):
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(lr), amsgrad=True)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=float(lr), momentum=.9, nesterov=True,
        #                            weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, steps_per_epoch=554,
        #                                                       epochs=n_epochs)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr * 10,
        #                                                    step_size_up=100, cycle_momentum=False,
        #                                                    mode="triangular")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 10, T_mult=1, eta_min=lr/10, last_epoch=-1)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def save(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        torch.save(self.net.state_dict(), f'{checkpoint_dir}/{prefix}{sep}{name}.pth')

    def load(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        self.net.load_state_dict(torch.load(f'{"/".join([checkpoint_dir, prefix, ])}{sep}{name}.pth'))

    def infer(self, x_raw, positions, **kwargs):
        return self.net(x_raw, positions)

    def train(self, epoch, trn_dl, **kwargs):
        self.net.train()
        correct, total = 0, 0

        with tqdm(trn_dl, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):

                if self.aug_type != 'none':
                    x_raw, y_batch, n_batch, index = [torch.cat(_, 0).to(self.device) for _ in batch]
                else:
                    x_raw, y_batch, n_batch, index = [_.to(self.device) for _ in batch]

                out = self.infer(x_raw, n_batch)

                loss = self.loss_func(out, torch.argmax(y_batch, dim=1), index=index).mean()

                self.optimize(loss)

                total += y_batch.size(0)
                correct += (F.softmax(out, dim=1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                t_epoch.set_postfix(loss=loss.item(), acc=correct / total,
                                    lr=self.optimizer.param_groups[0]['lr'])

        return loss, correct / total

    def eval(self, tst_dl, device=None, **kwargs):
        if device:
            self.device = device
        outputs = []
        entropic_scores = []
        total = correct = 0

        # apply model on test signals
        for batch in tst_dl:
            x_raw, y_batch, n_batch, _ = [t.to(self.device) for t in batch]
            pred = self.infer(x_raw, n_batch, mode='test')
            pred = F.softmax(pred, dim=1)

            probabilities = pred  # torch.nn.Softmax(dim=1)(pred)
            entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            entropic_scores.append((-entropies).cpu().numpy())

            outputs.append(pred.cpu().numpy())
            total += y_batch.size(0)
            correct += (pred.argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()

        outputs = np.concatenate(outputs)
        entropic_scores = np.concatenate(entropic_scores)
        return outputs, entropic_scores, correct / total

    def forward_backward_semi_supervised(self, *args, **kwargs):
        pass

    @staticmethod
    def get_activation(name: str, activation: dict):
        """For getting intermediate layer outputs"""
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook


class ModelSam(Model):
    """A adaptive version of Model to train with SAM optimizer"""
    def init_optimizers(self, lr=1e-3, n_epochs=None, rho=5e-2, weight_decay=0, *args, **kwargs):
        from utils.get_optimizer import SAMSGD
        self.optimizer = SAMSGD(self.net.parameters(), lr=float(lr), rho=rho, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 10, T_mult=1, eta_min=0, last_epoch=-1)

    def closure(self, x_raw, y_batch, n_batch, index):
        pass

    def train(self, epoch, trn_dl, **kwargs):
        self.net.train()
        correct, total = 0, 0

        with tqdm(trn_dl, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):

                if self.aug_type != 'none':
                    x_raw, y_batch, n_batch, index = [torch.cat(_, 0).to(self.device) for _ in batch]
                else:
                    x_raw, y_batch, n_batch, index = [_.to(self.device) for _ in batch]

                out = None

                def closure():
                    nonlocal out
                    self.optimizer.zero_grad()
                    out = self.infer(x_raw, n_batch)
                    loss = self.loss_func(out, torch.argmax(y_batch, dim=1), index=index).mean()
                    # loss = smooth_crossentropy(out, torch.argmax(y_batch, dim=1)).mean()
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
                self.scheduler.step()

                total += y_batch.size(0)
                correct += (F.softmax(out, dim=1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                t_epoch.set_postfix(loss=loss.item(), acc=correct / total,
                                    lr=self.optimizer.param_groups[0]['lr'])

        return loss, correct / total

