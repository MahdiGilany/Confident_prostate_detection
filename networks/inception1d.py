import math
import torch
from torch import nn

from networks.utils import Conv1dSamePadding

from typing import cast, Union, List
import torch.nn.functional as F
import torch.nn.init as init
from loss_functions.isomax import IsoMaxLossFirstPart


class InceptionModel(nn.Module):
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
                 num_pred_classes: int = 1, self_train=False, stride=1,
                 num_positions=0,
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
            'num_pred_classes': num_pred_classes,
            'stride': stride,
        }
        self.self_train = self_train
        # channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
        #                                                                   num_blocks))
        # bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        channels = [in_channels] + [out_channels * stride**i for i in range(num_blocks)]
        bottleneck_channels = [bottleneck_channels] * num_blocks
        # bottleneck_channels = [c//2 for c in channels[1:]]
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        strides = cast(List[int], self._expand_to_blocks(stride, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
                             )

        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i], stride=strides[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.feature_size = channels[-1]
        self.num_positions = num_positions
        # self.pos_encoder = nn.Linear(in_features=channels[-1] + num_positions, out_features=channels[-1])
        # linear_in = channels[-1] + num_positions if num_positions > 0 else channels[-1]
        # self.pos_encoder1 = nn.Sequential(
        #     nn.Dropout(.5),
        #     nn.Linear(linear_in, channels[-1]), nn.PReLU(), nn.BatchNorm1d(channels[-1]))
        # self.pos_encoder2 = nn.Sequential(
        #     nn.Dropout(.5),
        #     nn.Linear(linear_in, channels[-1]), nn.PReLU(), nn.BatchNorm1d(channels[-1]))

        self.linear00 = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            # nn.Dropout(.5),
        )
        self.linear01 = IsoMaxLossFirstPart(channels[-1], num_pred_classes)
        # self.linear01 = nn.Linear(channels[-1], num_pred_classes)
        # self.fc = IsoMaxLossFirstPart(channels[-1], num_pred_classes)

        # self.linear10 = nn.Linear(channels[-1], channels[-1])
        # self.linear11 = IsoMaxLossFirstPart(channels[-1], num_positions)

        # self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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

    def forward(self, x: torch.Tensor, *args):
        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        if self.self_train:
            return F.normalize(x, dim=1)
        out1 = self.linear01(self.linear00(x))
        # out1 = self.linear01(x)
        # out1 = self.fc(x)

        # if self.num_positions > 0:
        #     # x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
        #     # x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
        #     # x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        #     # if self.num_positions > 0:
        #     #     x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
        #     #     x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
        #     # else:
        #     #     x = self.pos_encoder2(self.pos_encoder1(x))
        #     out2 = self.linear11(F.relu(self.linear10(x)))
        #     return out1, out2
        return out1


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41, drop: float = None) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        strides = [1, 1, stride]
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=strides[i], bias=False)
            for i in range(len(kernel_size_s))
        ])

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        # print(x.shape, self.residual(org_x).shape)
        if self.use_residual:
            x = x + self.residual(org_x)
        return x


def main():
    from torchinfo import summary
    num_blocks, in_channels, pred_classes = 5, 1, 2
    net = InceptionModel(num_blocks, in_channels, out_channels=16,
                         bottleneck_channels=12, kernel_sizes=15, use_residuals='default', stride=2,
                         num_pred_classes=pred_classes, num_positions=0)
    summary(net, input_size=[(2, in_channels, 286), (2, 12)])

    # net = InceptionBlock(in_channels=in_channels, out_channels=3,
    #                      residual=True, stride=1, bottleneck_channels=12,
    #                      kernel_size=15)
    # summary(net, input_size=[(2, in_channels, 200)])

if __name__ == '__main__':
    main()
