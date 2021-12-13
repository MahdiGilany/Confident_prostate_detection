import torch
from torch import nn
from .patch_embedding import FFTPatchEmbedder, ResNet10PatchEmbedder
from .attention import SimpleGlobalAttention
from loss_functions.isomax import IsoMaxLossFirstPart


class ResViT(nn.Module):
    def __init__(self, hidden_dim, attn_key_dim, num_classes, input_height, input_width,
                 patch_embedder: str='resnet_10'):
        super(ResViT, self).__init__()
        if patch_embedder == 'resnet_10_FFT':
            self.patch_embedder = FFTPatchEmbedder(input_height, input_width, hidden_dim)
        elif patch_embedder == 'resnet_10':
            self.patch_embedder = ResNet10PatchEmbedder(input_height, input_width, hidden_dim)
        self.attention = SimpleGlobalAttention(hidden_dim, attn_key_dim)
        # self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc = IsoMaxLossFirstPart(hidden_dim, num_classes)

    def forward(self, x, *args):
        x = self.patch_embedder(x)
        x = self.attention(x)
        x = x[..., 0, :]
        x = self.fc(x)
        return x

def ResNet10_ViTv2(num_classes=2, in_channels=1):
    model = ResViT(
        input_height=256,
        input_width=256,
        num_classes=num_classes,
        hidden_dim=64,
        attn_key_dim=64,
        patch_embedder='resnet_10_FFT'
    )

    return model
