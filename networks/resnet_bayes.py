'''
Bayesian ResNet for CIFAR10.
ResNet architecture ref:
https://arxiv.org/abs/1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
try:
    from layers.conv_variational import Conv1dVariational
    from layers.linear_variational import LinearVariational
except:
    from .layers.conv_variational import Conv1dVariational
    from .layers.linear_variational import LinearVariational

__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'
]

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv1dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       in_planes,
                                       planes,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = Conv1dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       planes,
                                       planes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2],
                    (0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    Conv1dVariational(prior_mu,
                                      prior_sigma,
                                      posterior_mu_init,
                                      posterior_rho_init,
                                      in_planes,
                                      self.expansion * planes,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False),
                    nn.BatchNorm1d(self.expansion * planes))

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        out, kl = self.conv2(out)
        kl_sum += kl
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, kl_sum


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, first_dim=16, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = first_dim

        self.conv1 = Conv1dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       in_channels,
                                       first_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm1d(first_dim)
        self.layer1 = self._make_layer(block, first_dim, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, first_dim * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, first_dim * 4, num_blocks[2], stride=2)
        self.linear = LinearVariational(prior_mu, prior_sigma,
                                        posterior_mu_init, posterior_rho_init,
                                        first_dim * 4, num_classes)
        self.feature_size = first_dim * 4
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        for l in self.layer1:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer2:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer3:
            out, kl = l(out)
        kl_sum += kl

        out = F.avg_pool1d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        # out, kl = self.linear(out)
        # kl_sum += kl
        return out, kl_sum


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(lambda p: p.requires_grad and len(p.data.size()) > 1,
                       net.parameters()))))


if __name__ == "__main__":
    import numpy as np
    from torchinfo import summary

    # for net_name in __all__:
    #     if net_name.startswith('resnet20'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()
    net = resnet32(num_classes=2, in_channels=1)
    summary(net, input_size=[(2, 1, 200)])
