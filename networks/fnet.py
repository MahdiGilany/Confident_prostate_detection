import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dim_out=None, dropout=0.):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0., num_pred_classes=2, num_positions=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_positions = num_positions
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        if num_positions > 0:
            self.pos_encoder = nn.Sequential(
                PreNorm(dim + num_positions, FeedForward(dim + num_positions, mlp_dim, dropout=dropout, dim_out=dim)),
            )
        self.linear = nn.Linear(dim, num_pred_classes)

    def forward(self, x, *args):
        x = x.squeeze(dim=1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        if self.num_positions > 0:
            x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        return self.linear(x)


def main():
    from torchinfo import summary

    net = FNet(200, 5, 32, .5, num_positions=8)
    num_positions = 0
    input_size = [(2, 1, 200), (2, 8)]
    if num_positions > 0:
        input_size = [input_size, (2, num_positions)]
    summary(net, input_size=input_size)


if __name__ == '__main__':
    main()
