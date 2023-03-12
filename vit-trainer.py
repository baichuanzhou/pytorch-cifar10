import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
import models

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import sampler
from utils import *

NUM_TRAIN = 45000
NUM = 50000


class ViT(pl.LightningModule):
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
        super().__init__()
        # assert (image_size % patch_size) == 0, "image_size must be divisible by patch_size"
        # num_patches = (image_size // patch_size) ** 2
        #
        # self.patch_embed = nn.Sequential(
        #     nn.Conv2d(in_channels=num_channels,
        #               out_channels=dim,
        #               kernel_size=patch_size,
        #               stride=patch_size),
        #     Rearrange('b e h w -> b (h w) e')
        # )
        #
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.dropout = nn.Dropout(dropout_rate)
        #
        # self.transformer = nn.ModuleList([
        #     nn.TransformerEncoderLayer(
        #         d_model=dim,
        #         nhead=heads,
        #         dim_feedforward=mlp_dim,
        #         dropout=dropout_rate
        #     ) for _ in range(depth)
        # ])
        #
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.vit = models.ViT(
            in_channels=in_channels,
            num_heads=num_heads,
            transformer_depth=transformer_depth,
            img_size=img_size,
            emb_size=emb_size,
            n_classes=n_classes,
            dropout=dropout,
            patch_size=patch_size
        )

    def forward(self, x):
        # x = self.patch_embed(x)
        # b, n, c = x.shape
        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        # x = self.dropout(x)
        # for transformer in self.transformer:
        #     x = transformer(x)
        # x = x.mean(dim=1)  # average over all tokens
        # x = self.mlp_head(x)
        x = self.vit(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=1e-3, weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    transform = T.Compose([
        T.ToTensor(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    )
    train_dataset = CIFAR10(root='./datasets', train=True, download=False, transform=transform)
    val_dataset = CIFAR10(root='./datasets', train=False, download=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=4,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=128,
    )

    model = ViT(
        n_classes=10,
        img_size=32,
        patch_size=4,
        in_channels=3,
        emb_size=384,
        transformer_depth=7,
        num_heads=8,
        dropout=0.1
    )

    logger = TensorBoardLogger("logs/", name="ViT")
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=logger)
    trainer.fit(model, train_loader, val_loader)
