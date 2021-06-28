import torch
from torch import nn

from networks.utils import ConvBlock, Conv1dSamePadding


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1, num_positions: int = 0) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes,
            'num_positions': num_positions,
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        if num_positions > 0:
            self.pos_encoder = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(mid_channels * 2 + num_positions, mid_channels * 2), nn.PReLU(),
                nn.BatchNorm1d(mid_channels * 2))
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

        # import math
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.xavier_normal_(m.weight.data)
        #     #        nn.init.xavier_normal_(m.bias.data)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:  # type: ignore
        x = self.layers(x).mean(dim=-1)
        if self.input_args['num_positions'] > 0:
            x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        return self.final(x)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


def main():
    from torchinfo import summary
    mid_channels, in_channels, pred_classes = 64, 1, 2
    net = ResNetBaseline(in_channels, mid_channels=mid_channels,
                         num_pred_classes=pred_classes, num_positions=8)
    summary(net, input_size=[(2, 1, 200), (2, 8)])


if __name__ == '__main__':
    main()

