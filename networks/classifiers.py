
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_pred_classes, num_positions):
        super(Classifier, self).__init__()

        self.num_positions = num_positions
        if num_positions > 0:
            self.pos_encoder = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(in_channels + num_positions, out_channels), nn.PReLU(), nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.linear = nn.Linear(in_features=in_channels, out_features=num_pred_classes)

    def forward(self, x, *args):
        if self.num_positions > 0:
            x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        return self.linear(x)
