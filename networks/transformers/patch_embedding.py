import einops
import torch.fft
from torch import nn
from einops import rearrange
from .resnet_small import ResNet10


class PatchSplitter(nn.Module):
    def __init__(self, input_height, input_width, patch_height, patch_width):
        super(PatchSplitter, self).__init__()
        assert input_height % patch_height == 0
        assert input_width % patch_width == 0
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x):
        assert len(x.shape) == 4    # (b, c, h, w)
        return rearrange(x, 'b c (n_h patch_height) (n_w patch_width) -> b (n_h n_w) c patch_height patch_width',
                         patch_height=self.patch_height, patch_width=self.patch_width)


class AxialFFT(nn.Module):
    """invokes an FFT in the axial (height dimension , -2) of the input data patches.
    meant to be called on data of shape b, num_patches, patch_height, patch_width
    """
    def forward(self, x):
        fft = torch.fft.rfft(x, dim=-2)
        fft_magnitude = torch.abs(fft)
        return fft_magnitude[..., 1:, :]   # output only positive frequencies, output height is patch_height//2


class ResNet10PatchEmbedder(nn.Module):
    def __init__(self, input_height, input_width, embedding_dim):
        super(ResNet10PatchEmbedder, self).__init__()
        self.split_patches = PatchSplitter(input_height, input_width, patch_height=32, patch_width=32)
        self.patch_encoder = ResNet10(num_classes=embedding_dim, num_channels=1)

    def forward(self, x):
        x = self.split_patches(x)
        b, n, c, h, w = x.shape
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')  # fold n_patches into batch dimension
        x = self.patch_encoder(x)
        x = einops.rearrange(x, '(b n) emb_dim -> b n emb_dim', n=n)
        return x


class FFTPatchEmbedder(nn.Module):
    def __init__(self, input_height, input_width, embedding_dim):
        super(FFTPatchEmbedder, self).__init__()
        self.split_patches = PatchSplitter(input_height, input_width, patch_height=64, patch_width=32)
        self.fft = AxialFFT()
        self.patch_encoder = ResNet10(num_classes=embedding_dim, num_channels=1)

    def forward(self, x):
        x = self.split_patches(x)
        x = self.fft(x)
        b, n, c, h, w = x.shape
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w') # fold n_patches into batch dimension
        x = self.patch_encoder(x)
        x = einops.rearrange(x, '(b n) emb_dim -> b n emb_dim', n=n)
        return x

