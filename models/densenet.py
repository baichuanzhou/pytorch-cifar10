"""
    Implementation of DenseNet using PyTorch.
    Reference:
    [1]

"""
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Optional, Tuple

current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import *

__all__ = ["DenseNet", "densenet"]


class _BasicConv(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super(_BasicConv, self).__init__()
        self.add_module("bn", nn.BatchNorm2d(in_planes))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_channels=in_planes,
                                          out_channels=out_planes,
                                          kernel_size=3,
                                          padding=1))


class _BottleNeck(nn.Sequential):
    def __init__(self, in_planes: int, growth_rate: int, bn_size: int = 4) -> None:
        super(_BottleNeck, self).__init__()
        self.add_module("bottleneck_bn", nn.BatchNorm2d(in_planes))
        self.add_module("bottleneck_relu", nn.ReLU(inplace=True))
        self.add_module("bottleneck_conv", nn.Conv2d(in_channels=in_planes,
                                                     out_channels=bn_size * growth_rate,
                                                     kernel_size=1))


class _TransitionLayer(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super(_TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpooling = nn.AvgPool2d(kernel_size=2)


class _DenseLayer(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 growth_rate: int,
                 drop_rate: float = 0.5,
                 bottleneck: bool = True,
                 bn_size: int = 4
                 ) -> None:
        super(_DenseLayer, self).__init__()
        out_planes = in_planes
        if bottleneck:
            out_planes = bn_size * growth_rate
            self.add_module("bottleneck", _BottleNeck(in_planes, growth_rate, bn_size))
        self.add_module("basicConv", _BasicConv(out_planes, growth_rate))
        self.drop_rate = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([new_features, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self,
                 num_of_layers: int,
                 in_planes: int,
                 growth_rate: int,
                 drop_rate: float = 0.5,
                 bottleneck: bool = True,
                 bn_size: int = 4) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_of_layers):
            layer = _DenseLayer(
                in_planes=in_planes + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                bottleneck=bottleneck,
                bn_size=bn_size
            )
            self.add_module(f"denselayer{i+1}", layer)


class DenseNet(nn.Sequential):
    def __init__(self,
                 growth_rate: int = 12,
                 in_planes: int = 3,
                 reduction: float = 0.5,
                 bottleneck: bool = True,
                 drop_rate: float = 0.5,
                 block_config=(6, 12, 24, 16),
                 bn_size: int = 4,
                 num_classes=10
                 ):
        super(DenseNet, self).__init__()
        out_planes = 16
        if bottleneck and (reduction < 1):
            out_planes = 2 * growth_rate
        self.add_module("basicConv", _BasicConv(in_planes, out_planes))
        self.add_module("avgpool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        for i, num_of_layers in enumerate(block_config):
            self.add_module(f"denseblock{i+1}", _DenseBlock(
                num_of_layers=num_of_layers,
                in_planes=out_planes,
                growth_rate=growth_rate,
                bottleneck=bottleneck,
                bn_size=bn_size,
                drop_rate=drop_rate
            ))
            out_planes = out_planes + num_of_layers * growth_rate
            if i != len(block_config) - 1:
                in_planes, out_planes = out_planes, int(out_planes * reduction)
                self.add_module(f"transition{i+1}", _TransitionLayer(in_planes=in_planes,
                                                                     out_planes=out_planes))

        self.add_module("adaptivepool", nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module("flatten", nn.Flatten())
        self.add_module("linear", nn.Linear(out_planes, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


def densenet_bc100(pretrained=True):
    if not pretrained:
        return DenseNet(growth_rate=12, block_config=[32, 32, 32])
    else:
        model = load("DenseNetBC-100")
        return model


def densenet_bc250(pretrained=True):
    if not pretrained:
        return DenseNet(growth_rate=24, block_config=[82, 82, 82])
    else:
        model = load("DenseNetBC-250")
        return model


def densenet_bc190(pretrained=True):
    if not pretrained:
        return DenseNet(growth_rate=40, block_config=[62, 62, 62])
    else:
        model = load("DenseNetBC-190")
        return model


def densenet_100(pretrained=True):
    if not pretrained:
        return DenseNet(bottleneck=False, reduction=1, block_config=[32, 32, 32])
    else:
        model = load("DenseNet-100")
        return model


def densenet_40(pretrained=True):
    if not pretrained:
        return DenseNet(bottleneck=False, reduction=1, block_config=[12, 12, 12])
    else:
        model = load("DenseNet-40")
        return model


def densenet(num_of_layers, bottleneck=True, pretrained=False):
    block_layer = (num_of_layers - 4) // 3
    if not pretrained:
        if bottleneck:
            if num_of_layers == 100:
                return densenet_bc100(pretrained)
            elif num_of_layers == 190:
                return densenet_bc190(pretrained)
            elif num_of_layers == 250:
                return densenet_bc250(pretrained)
            return DenseNet(block_config=[block_layer] * 3)
        else:
            if num_of_layers == 40:
                return densenet_40(pretrained)
            elif num_of_layers == 100:
                return densenet_100(pretrained)
            return DenseNet(bottleneck=False, block_config=[block_layer] * 3)
    else:
        suffix = ""
        if bottleneck:
            suffix = "BC"
        model = load("DenseNet" + suffix + "-" + str(num_of_layers))
        return model

