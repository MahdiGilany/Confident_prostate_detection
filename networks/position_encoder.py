import torch
from torch import nn


class PosEncoder(nn.Module):
    variational = False
    num_positions = 0

    def __init__(self, feat_extractor: nn.Module, num_positions: int = 0, variational: bool = False):
        super(PosEncoder, self).__init__()
        self.num_positions = num_positions
        self.variational = variational
        self.feat_extractor = feat_extractor
        self.update_forward()
        feature_size = feat_extractor.feature_size

        if num_positions > 0:
            self.pos_encoder1 = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(feature_size + num_positions, feature_size), nn.PReLU(), nn.BatchNorm1d(feature_size))
            self.pos_encoder2 = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(feature_size + num_positions, feature_size), nn.PReLU(), nn.BatchNorm1d(feature_size))

    def pos_encode(self, x, pos):
        x = self.pos_encoder1(torch.cat((x, pos.float()), 1))
        x = self.pos_encoder2(torch.cat((x, pos.float()), 1))
        return x

    def _forward_normal(self, x, pos):
        x = self.feat_extractor.forward(x)
        x = self.pos_encode(x, pos)
        return self.feat_extractor.linear(x)

    def _forward_normal_non_pos(self, x, *args):
        x = self.feat_extractor.forward(x)
        return self.feat_extractor.linear(x)

    def _forward_variational(self, x, pos):
        sum_kl = 0
        x, kl = self.feat_extractor.forward(x)
        sum_kl += kl
        x = self.pos_encode(x, pos)
        x, kl = self.feat_extractor.linear(x)
        sum_kl += kl
        return x, sum_kl

    def _forward_variational_non_pos(self, x, *args):
        sum_kl = 0
        x, kl = self.feat_extractor.forward(x)
        sum_kl += kl
        x, kl = self.feat_extractor.linear(x)
        sum_kl += kl
        return

    def update_forward(self):
        variational, num_positions = self.variational, self.num_positions
        if variational and num_positions:
            self.forward = self._forward_variational
        elif variational and (not num_positions):
            self.forward = self._forward_variational_non_pos
        elif (not variational) and num_positions:
            self.forward = self._forward_normal
        else:
            self.forward = self._forward_normal_non_pos


def main():
    from torchinfo import summary
    from resnet import resnet32
    from resnet_bayes import resnet32 as resnet_variational

    num_blocks, in_channels, num_classes = 3, 1, 2
    num_positions = 0

    # _net = resnet_variational(in_channels=in_channels, num_classes=num_classes)
    _net = resnet32(in_channels=in_channels, num_classes=num_classes)
    net = PosEncoder(_net, num_positions=num_positions, variational=False)
    summary(net, input_size=[(2, 1, 200), (2, 8)])


if __name__ == '__main__':
    main()
