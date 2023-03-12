import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


__all__ = ['ViT']

###############################################
#   The code consists of:
#   1. Patch Embedding
#       - CLS Token
#       - Position Embedding
#   2. Transformer
#       - Attention
#       - Residuals
#       - MLP
#       - TransformerEncoder
################################################


################################################
#   Implements:
#       Patch Embedding
#
################################################

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128, img_size: int = 32):
        super().__init__()
        self.patch_size = patch_size

        # naive implementation
        # self.projection = nn.Sequential(
        #     Rearrange(
        #         'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size
        #     ),
        #     nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # )

        # performance gain
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=emb_size,
                      kernel_size=patch_size,
                      stride=patch_size),
            Rearrange('b e h w -> b (h w) e')

        )

        # The CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        b = x.size(0)  # Batch size

        out = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        out = torch.cat([cls_tokens, out], dim=1)
        out += self.pos_emb
        return out


################################################
#   Implements:
#       MultiHead Attention
#
################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        # Split keys queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** 0.5
        # compute the attention score
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # compute the values
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)
        out = self.projection(out)
        return out


################################################
#   Implements:
#       Residual Block
#
################################################


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


################################################
#   Implements:
#       MLP
#
################################################

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.Linear(expansion * emb_size, emb_size)
        )


################################################
#   Implements:
#       TransformerEncoder
#
################################################

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 128,
                 dropout: float = 0.5,
                 expansion: int = 4,
                 num_heads: int = 8
                 ):
        super().__init__(
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size=emb_size, num_heads=num_heads, dropout=dropout),
                    nn.Dropout(dropout)
                )
            ),
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size=emb_size, expansion=expansion),
                    nn.Dropout(dropout)
                )
            )
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )


################################################
#   Implements:
#       MLP Head
#
################################################

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 128, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.GELU(),
            nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 32,
                 patch_size: int = 2,
                 num_heads: int = 8,
                 emb_size: int = 384,
                 transformer_depth: int = 8,
                 n_classes: int = 10,
                 dropout: float = 0.1
                 ):
        super().__init__(
            PatchEmbedding(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                emb_size=emb_size
            ),
            TransformerEncoder(
                emb_size=emb_size,
                dropout=dropout,
                num_heads=num_heads,
                depth=transformer_depth
            ),
            ClassificationHead(
                emb_size=emb_size,
                n_classes=n_classes
            )
        )

