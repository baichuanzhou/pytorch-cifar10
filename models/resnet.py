"""
    Implementation of ResNet and its variants using PyTorch.
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    [2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
    # BasicBlock of ResNet as described in the paper,
    # which has two layers.
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            stride=(stride, stride),
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        out = out + identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    # BottleNeck Block use three layers instead of two for computational purposes.
    # Here's a good visualization of ResNet50 that can explain a lot.
    # http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    expansion = 2

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1
    ) -> None:
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    # As describe in the paper, ResNet for CIFAR-10 dataset has 6n+2 layers.
    # First layer is a 3 x 3 convolution. Then it uses a stack of 6n layers.
    # It has 3 types of layers with filter size:{16, 32, 64}, with each type using 2n layers.
    # The net ends with a global average pooling and a 10-way fully connected layer.
    def __init__(
            self,
            block: Type[Union[BasicBlock, BottleNeck]],
            num_blocks: int,
            num_classes: int = 10
    ) -> None:
        self.in_planes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 2 * num_blocks, 16, 1)
        self.layer2 = self._make_layer(block, 2 * num_blocks, 32, 2)
        self.layer3 = self._make_layer(block, 2 * num_blocks, 64, 2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, BottleNeck]],
            num_blocks: int,
            planes: int,
            stride: int
    ) -> nn.Sequential:
        strides = [stride] + (num_blocks - 1) * [1]
        block_layers = []
        for stride in strides:
            block_layers.append(
                block(self.in_planes, planes, stride)
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*block_layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.linear(out)
        return out


def make_resnet(num_layers: int) -> ResNet:
    """
    A function that takes in a number as the number of layers of a ResNet.
    num_layers = 6 x num_blocks + 2
    Returns a ResNet of #num_layers layers.
    Input:
        - num_layers: The number of layers of a ResNet
    Output:
        - ResNet of given blocks
    """
    if (num_layers - 2) % 6 != 0:
        raise ValueError("%d of layers cannot be implemented!" % num_layers)
    num_blocks = int((num_layers - 2) / 6)
    return ResNet(BottleNeck, num_blocks)


def test():
    ResNet50 = ResNet(BottleNeck, 8)
    x = torch.ones((1, 3, 32, 32))
    out = ResNet50(x)
    print(out.size())

