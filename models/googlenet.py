from typing import Any, Callable, List, Optional, Tuple
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

_GoogLeNetOutputs = GoogLeNetOutputs

__all__ = ["GoogleNet"]


class GoogleConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(GoogleConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, **kwargs)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_planes, plane1x1, plane3x3_in, plane3x3, plane5x5_in, plane5x5, pool_planes):
        super(Inception, self).__init__()

        # 1x1 inception module (H, W unchanged)
        self.branch1x1 = GoogleConv2d(in_planes, plane1x1, kernel_size=1)

        # 3x3 inception module (1x1 Conv -> 3x3 Conv -> Out) (H, W Unchanged)
        self.branch3x3 = nn.Sequential(
            GoogleConv2d(in_planes, plane3x3_in, kernel_size=1),
            GoogleConv2d(plane3x3_in, plane3x3, kernel_size=3, padding=1)
        )

        # 5x5 inception module (1x1 Conv -> 5x5 Conv -> Out) (H, W Unchanged)
        self.branch5x5 = nn.Sequential(
            GoogleConv2d(in_planes, plane5x5_in, kernel_size=1),
            GoogleConv2d(plane5x5_in, plane5x5, kernel_size=5, padding=2)
        )

        # Pooling inception module (3x3 Maxpooling -> 1x1 Conv -> Out) (H, W Unchanged)
        self.branchmaxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            GoogleConv2d(in_planes, pool_planes, kernel_size=1)
        )

    def forward(self, x):
        branch1x1_out = self.branch1x1(x)
        branch3x3_out = self.branch3x3(x)
        branch5x5_out = self.branch5x5(x)
        branchmaxpool_out = self.branchmaxpool(x)

        out = torch.cat([branch1x1_out, branch3x3_out, branch5x5_out, branchmaxpool_out], 1)
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_planes):
        super(InceptionAux, self).__init__()

        self.conv = GoogleConv2d(in_planes, 128, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (4, 4))

        out = self.conv(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, 0.7, training=self.training)
        out = self.fc2(out)
        return out


class GoogleNet(nn.Module):
    def __init__(self, aux_classifier=False):
        super(GoogleNet, self).__init__()

        # Input layer (32 x 32 x 3) -> (32 x 32 x 192)
        self.conv1 = GoogleConv2d(3, 192, kernel_size=3, padding=1)

        # (32 x 32 x 192) -> (32 x 32 x 256(64 + 128 + 32 + 32)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)

        # (32 x 32 x 256) -> (32 x 32 x 480(128 + 192 + 96 + 64)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        # (32 x 32 x 480) -> (16 x 16 x 480)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        # (16 x 16 x 480) -> (16 x 16 x 512)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)

        # (16 x 16 x 512) -> (16 x 16 x 512)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)

        # (16 x 16 x 512) -> (16 x 16 x 512)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)

        # (16 x 16 x 512) -> (16 x 16 x 528)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)

        # (16 x 16 x 528) -> (16 x 16 x 832)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        # (16 x 16 x 832) -> (8 x 8 x 832)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        # (8 x 8 x 832) -> (8 x 8 x 832)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)

        # (8 x 8 x 832) -> (8 x 8 x 1024)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_classifier:
            self.aux1 = InceptionAux(512)
            self.aux2 = InceptionAux(528)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv1(x)

        out = self.inception3a(out)
        out = self.inception3b(out)

        out = self.maxpool3(out)

        out = self.inception4a(out)

        if self.aux1:
            if self.training:
                aux1 = self.aux1(out)
            else:
                aux1 = None

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        if self.aux2:
            if self.training:
                aux2 = self.aux2(out)
            else:
                aux2 = None

        out = self.inception4e(out)

        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)

        out = self.dropout(out)
        out = self.fc(out)

        if self.training and self.aux1 and self.aux2:
            return _GoogLeNetOutputs(out, aux2, aux1)
        else:
            return out

def test():
    x = torch.zeros((1, 3, 32, 32))
    googlenet = GoogleNet(aux_classifier=True)
    out = googlenet(x)
    print(out)