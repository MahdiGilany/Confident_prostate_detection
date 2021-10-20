import torch
from torch import nn


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout
    layers right after a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad,
                 drop=None, bn=True,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier3LV1(nn.Module):
    def __init__(self, raw_ni, no, drop=.5, num_positions=12):
        super().__init__()
        self.num_positions = num_positions
        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 32, 5, 2, 3, drop=drop),
            SepConv1d(32, 32, 3, 1, 1, drop=drop),
            SepConv1d(32, 64, 5, 4, 3, drop=drop),
            SepConv1d(64, 64, 3, 1, 1, drop=drop),
            SepConv1d(64, 128, 5, 4, 1, drop=drop),
            SepConv1d(128, 128, 3, 1, 1, drop=drop),
            SepConv1d(128, 256, 1, 4, 2),
            Flatten())
        self.feat1 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(1024 + self.num_positions, 256), nn.PReLU(), nn.BatchNorm1d(256))
        self.feat2 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(256 + self.num_positions, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.PReLU())

        self.out = nn.Sequential(
            nn.Linear(128 + self.num_positions, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            out_counter = 0
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
                if isinstance(child, nn.Linear):
                    out_counter += 1
                    nn.init.uniform_(child.weights, a=0.0, b=1.0)
                    if out_counter > 4:
                        nn.init.normal_(child.weights, mean=0.0, std=1.0)

        init(self)

    def forward(self, t_raw, n_raw):
        raw_out = self.raw(t_raw)
        feat = torch.cat((raw_out, n_raw.float()), 1)
        f1 = self.feat1(feat)
        feat = torch.cat((f1, n_raw.float()), 1)
        f2 = self.feat2(feat)
        feat = torch.cat((f2, n_raw.float()), 1)
        return self.out(feat)



class _SepConv2d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv2d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv2d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv2d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout
    layers right after a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad,
                 drop=None, bn=True,
                 activ=lambda: nn.ReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv2d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm2d(no))
        if drop is not None:
            layers.append(nn.Dropout2d(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Classifier3L_2D(nn.Module):
    def __init__(self, raw_ni, no, drop=.5, num_positions=12):
        super().__init__()
        self.num_positions = num_positions
        self.raw = nn.Sequential(
            SepConv2d(raw_ni, 32, (5,1), (2,1), 3, drop=drop),
            SepConv2d(32, 32, (3,2), (2,1), 1, drop=drop),
            SepConv2d(32, 64, (5, 3), (4, 2), 3, drop=drop),
            SepConv2d(64, 64, 3, 1, 1, drop=drop),
            SepConv2d(64, 128, (5, 4), (4, 1), 1, drop=drop),
            SepConv2d(128, 128, 3, 2, 1, drop=drop),
            SepConv2d(128, 256, 1, 4, 2),
            Flatten())
        # self.raw1 = SepConv2d(raw_ni, 32, (5, 1), (1, 1), 3, drop=drop)
        # self.raw2 = SepConv2d(32, 32, (3, 2), (2, 1), 1, drop=drop)
        # self.raw3 = SepConv2d(32, 64, (3, 3), (2, 2), 3, drop=drop)
        # self.raw4 = SepConv2d(64, 64, 3, 1, 1, drop=drop)
        # # self.maxpool = nn.MaxPool2d((2,2))
        # self.raw5 = SepConv2d(64, 128, 2, 1, 1, drop=drop)
        # self.raw6 = SepConv2d(128, 128, 3, 2, 1, drop=drop)
        # self.raw7 = SepConv2d(128, 256, 2, 2, 1)
        # self.flatten = Flatten()
        # self.linear1 = nn.Sequential(nn.Linear(3072, 1024), nn.PReLU())

        # self.raw = nn.Sequential(
        #     SepConv2d(raw_ni, 32, (5,5), (3,2), 3, drop=drop),
        #     # nn.MaxPool2d(2),
        #     SepConv2d(32, 32, (3,2), (2,1), 1, drop=drop),
        #     SepConv2d(32, 64, (5, 3), (4, 2), 3, drop=drop),
        #     # SepConv2d(64, 64, 3, 1, 1, drop=drop),
        #     # SepConv2d(64, 128, (5, 4), (4, 1), 1, drop=drop),
        #     # SepConv2d(128, 128, 3, 2, 1, drop=drop),
        #     # SepConv2d(128, 256, 1, 4, 2),
        #     Flatten())
        self.feat1 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(1024 + self.num_positions, 512), nn.PReLU(), nn.BatchNorm1d(512))
        self.feat2 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(512 + self.num_positions, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.PReLU())

        self.out = nn.Sequential(
            nn.Linear(128 + self.num_positions, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            out_counter = 0
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
                if isinstance(child, nn.Linear):
                    out_counter += 1
                    nn.init.uniform_(child.weights, a=0.0, b=1.0)
                    if out_counter > 4:
                        nn.init.normal_(child.weights, mean=0.0, std=1.0)

        init(self)

    def forward(self, t_raw, n_raw):
        raw_out = self.raw(t_raw)
        # x = self.raw1(t_raw)
        # x = self.raw2(x)
        # x = self.raw3(x)
        # x = self.raw4(x)
        # x = self.maxpool(x)
        # x = self.raw5(x)
        # x = self.raw6(x)
        # x = self.raw7(x)
        # raw_out = self.flatten(x)
        # raw_out = self.linear1(x)
        feat = torch.cat((raw_out, n_raw.float()), 1)
        f1 = self.feat1(feat)
        feat = torch.cat((f1, n_raw.float()), 1)
        f2 = self.feat2(feat)
        feat = torch.cat((f2, n_raw.float()), 1)
        return self.out(feat)


if __name__ == "__main__":
    import numpy as np
    from torchinfo import summary
    # for net_name in __all__:
    #     if net_name.startswith('resnet20'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()
    net = Classifier3L_2D(1, 2, num_positions=12)
    summary(net, input_size=[(2, 1, 286, 5), (2, 12)])

    # net = Classifier3LV1(1, 2, num_positions=12)
    # summary(net, input_size=[(2, 1, 286), (2, 12)])
