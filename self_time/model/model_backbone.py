# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimConv4(torch.nn.Module):
    def __init__(self, ch_multiplier=1, is_classifier=False, nb_class=None, num_positions=0, self_train=False):
        super(SimConv4, self).__init__()
        # self.feature_size = feature_size
        self.name = "conv4"
        self.is_classifier = is_classifier
        self.num_positions = num_positions
        self.self_train = self_train

        cin, cout = 1, 8 * ch_multiplier
        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(cin, cout, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(cout),
            torch.nn.ReLU()
        )

        cin, cout = cout, cout * 2
        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(cin, cout, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(cout),
            torch.nn.ReLU(),
        )

        cin, cout = cout, cout * 2
        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(cin, cout, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(cout),
            torch.nn.ReLU(),
        )

        cin, cout = cout, cout * 2
        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(cin, cout, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(cout),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()
        self.feature_size = cout

        if is_classifier:
            try:
                assert nb_class is not None
            except AssertionError:
                raise Exception('The number of classes has to be set when using the network as the classier')
            if num_positions > 0:
                self.pos_encoder = nn.Sequential(
                    nn.Dropout(.5),
                    nn.Linear(cout + num_positions, cout), nn.PReLU(), nn.BatchNorm1d(cout))
            self.linear = torch.nn.Linear(cout, nb_class)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, *args):
        # x_ = x.view(x.shape[0], 1, -1)

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h = self.flatten(h)

        if self.self_train:
            return F.normalize(h, dim=1)

        if self.is_classifier:
            if self.num_positions > 0:
                h = self.pos_encoder(torch.cat((h, args[0].float()), 1))
            return self.linear(h)

        return h


def main():
    from torchinfo import summary
    net = SimConv4(1, True, True, nb_class=2).to('cpu')
    print(summary(net, input_size=[(2, 1, 200), (2, 8)]))


if __name__ == '__main__':
    main()
