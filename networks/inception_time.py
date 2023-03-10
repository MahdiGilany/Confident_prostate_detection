import math
import torch
import torch.nn as nn


def noop(x):
    return x


def shortcut(c_in, c_out):
    return nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1),
                           nn.BatchNorm1d(c_out)])


class Inception(nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32):

        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1) if bottleneck and c_in > 1 else noop
        mts_feat = bottleneck or c_in
        conv_layers = []
        kss = [ks // (2 ** i) for i in range(3)]
        # ensure odd kss until nn.Conv1d with padding='same' is available in pytorch 1.3
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(mts_feat, nb_filters, kernel_size=kss[i], padding=kss[i] // 2))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = nn.Conv1d(c_in, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0:
                out = out_
            else:
                out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))


class InceptionBlock(nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32, residual=True, depth=6):

        super().__init__()
        self.residual = residual
        self.depth = depth

        # inception & residual layers
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(depth):
            inc_mods.append(
                Inception(c_in if d == 0 else nb_filters * 4, bottleneck=bottleneck if d > 0 else 0, ks=ks,
                          nb_filters=nb_filters))
            if self.residual and d % 3 == 2:
                res_layers.append(shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4))
                res += 1
            else:
                res_layer = res_layers.append(None)
        self.inc_mods = nn.ModuleList(inc_mods)
        self.res_layers = nn.ModuleList(res_layers)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x


class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, bottleneck=32, ks=40, nb_filters=32, residual=True, depth=6,
                 self_train=False, num_positions=0):
        super().__init__()
        self.self_train = self_train
        self.num_positions = num_positions
        self.block = InceptionBlock(c_in, bottleneck=bottleneck, ks=ks, nb_filters=nb_filters,
                                    residual=residual, depth=depth)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.feature_size = nb_filters
        if num_positions > 0:
            self.pos_encoder = nn.Sequential(
                nn.Dropout(.5), nn.Linear(nb_filters + num_positions, nb_filters), nn.PReLU(),
                nn.BatchNorm1d(nb_filters)
            )
        self.fc = nn.Linear(nb_filters * 4, c_out)

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
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        if self.num_positions > 0:
            x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    net = InceptionTime(c_in=1, c_out=2, bottleneck=32, ks=40, nb_filters=32, residual=True, depth=6)
    input_size = [(2, 1, 200), ]
    # input_size = [input_size, (2, 8)]
    summary(net, input_size=input_size)
