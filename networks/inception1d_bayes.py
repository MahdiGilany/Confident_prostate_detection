import math
import torch
from torch import nn

from networks.utils import Conv1dSamePadding

from typing import cast, Union, List
import torch.nn.functional as F
import torch.nn.init as init
try:
    from .layers.conv_variational import Conv1dVariational
    from .layers.linear_variational import LinearVariational
except:
    from layers.conv_variational import Conv1dVariational
    from layers.linear_variational import LinearVariational


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0


class InceptionModelVariational(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1, num_positions: int = 0, self_train=False,
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes
        }
        self.self_train = self_train
        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
                                                                          num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
                             )

        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.feature_size = channels[-1]
        self.num_positions = num_positions
        if num_positions > 0:
            self.pos_encoder1 = LinearVariational(prior_mu, prior_sigma,
                                                  posterior_mu_init, posterior_rho_init,
                                                  channels[-1] + num_positions, channels[-1])
            self.pos_bn1 = nn.Sequential(nn.PReLU(), nn.BatchNorm1d(channels[-1]))
            self.pos_encoder2 = LinearVariational(prior_mu, prior_sigma,
                                                  posterior_mu_init, posterior_rho_init,
                                                  channels[-1] + num_positions, channels[-1])
            self.pos_bn2 = nn.Sequential(nn.PReLU(), nn.BatchNorm1d(channels[-1]))
        self.linear = LinearVariational(prior_mu, prior_sigma,
                                        posterior_mu_init, posterior_rho_init,
                                        channels[-1], num_pred_classes)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:  # type: ignore
        kl_sum = 0
        x, kl = self.blocks(x)
        x = x.mean(dim=-1)  # the mean is the global average pooling
        kl_sum += kl
        if self.self_train:
            return F.normalize(x, dim=1)
        if self.num_positions > 0:
            # x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
            x, kl = self.pos_encoder1(F.dropout(torch.cat((x, args[0].float()), 1), .5))
            x = self.pos_bn1(x)
            kl_sum += kl
            x, kl = self.pos_encoder1(F.dropout(torch.cat((x, args[0].float()), 1), .5))
            x = self.pos_bn2(x)
            kl_sum += kl
        x, kl = self.linear(x)
        kl_sum += kl
        return x, kl_sum


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41, drop: float = None) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            # self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
            #                                     kernel_size=1, bias=False)
            self.bottleneck = Conv1dVariational(
                prior_mu,
                prior_sigma,
                posterior_mu_init,
                posterior_rho_init,
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=stride,
                dilation=1,
                groups=1,
            )
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            # Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
            #                   kernel_size=kernel_size_s[i], stride=stride, bias=False)
            Conv1dVariational(
                prior_mu,
                prior_sigma,
                posterior_mu_init,
                posterior_rho_init,
                channels[i],
                channels[i + 1],
                kernel_size_s[i],
                stride=stride,
                dilation=1,
                groups=1, )
            for i in range(len(kernel_size_s))
        ])

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                # Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                #                   kernel_size=1, stride=stride, bias=False),
                Conv1dVariational(
                    prior_mu,
                    prior_sigma,
                    posterior_mu_init,
                    posterior_rho_init,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=1,
                    groups=1,
                ),
            ])
            self.residual_bn = nn.BatchNorm1d(out_channels)
            self.residual_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        kl_sum = 0
        if self.use_bottleneck:
            x, kl = self.bottleneck(x)
            kl_sum += kl
        x, kl = self.conv_layers(x)
        kl_sum += kl

        if self.use_residual:
            res, kl = self.residual(org_x)
            x = self.residual_relu(self.residual_bn(x))
            x = x + res
            kl_sum += kl
        return x, kl_sum


def main():
    from torchinfo import summary
    num_blocks, in_channels, pred_classes = 3, 1, 2
    net = InceptionModelVariational(num_blocks, in_channels, out_channels=30,
                                    bottleneck_channels=12, kernel_sizes=15, use_residuals=True,
                                    num_pred_classes=pred_classes, num_positions=8)
    summary(net, input_size=[(2, 1, 200), (2, 8)])


if __name__ == '__main__':
    main()
