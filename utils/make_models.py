import torch
from torch import nn
from resnet1d import ResNet1D


def make_resnet_1d():
    # make model
    device_str = "cuda"
    device = torch.device('cpu')
    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    model = ResNet1D(
        in_channels=8,
        base_filters=128,  # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size,
        stride=stride,
        groups=32,
        n_block=n_block,
        n_classes=4,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap,
        use_do=True)
    model.to(device)
    return model


def make_transformer(manual=False):
    device_name = 'cpu'
    device = torch.device("cuda" if device_name == 'gpu' else 'cpu')
    num_layers = 6
    num_heads = 8
    dim_model = 512
    dim_ffw = 2048
    drop_out = .1
    src_vocab = 3000
    trg_vocab = 3000

    if manual:
        # from transformer_from_scratch import Transformer
        # transformer = Transformer(num_layers, num_layers, dim_model, num_heads, dim_ffw, drop_out, device).to(device)
        from transformer_from_scratch2 import make_model as mm
        transformer = mm(10, 10, num_layers).to(device)
    else:
        transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ffw,
            dropout=drop_out,
            activation='relu',
        )
    return transformer


def make_model(name):
    if 'resnet' in name:
        return make_resnet_1d()
    elif 'transformer' in name:
        if 'manual' in name:
            return make_transformer(manual=True)
        return make_transformer()
