from einops import rearrange
import torch
from torch import nn


class SimpleGlobalAttention(nn.Module):
    """an attention layer in which """
    def __init__(self, input_dim, key_dim):
        super(SimpleGlobalAttention, self).__init__()
        self.global_query = nn.Parameter(torch.randn(1, key_dim))
        self.to_keys = nn.Linear(input_dim, key_dim)
        self.softmax = nn.Softmax(dim=-2)
        self.scale = input_dim ** -0.5
        self.cached_attention = None

    def forward(self, x):
        keys = self.to_keys(x)
        keys_transpose = rearrange(keys, 'b n k -> b k n')
        dots = torch.matmul(self.global_query, keys_transpose)
        attn = self.softmax(dots*self.scale)
        self.cached_attention = attn
        x = torch.matmul(attn, x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)