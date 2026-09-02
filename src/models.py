"""CNN architectures used in the receptive-field study."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F


class SamePadConv2d(nn.Module):
    """2D convolution with explicit asymmetric SAME padding.

    Explicit padding keeps the spatial resolution unchanged for both odd and
    even kernels (e.g. 2x2 and 28x28), which is important for this study.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        total_pad = self.kernel_size - 1
        left = total_pad // 2
        right = total_pad - left
        top = total_pad // 2
        bottom = total_pad - top
        return self.conv(F.pad(x, (left, right, top, bottom)))


def convolutional_receptive_field(kernel_sizes: Sequence[int]) -> int:
    """Return the receptive field of the stride-1 convolution stack only.

    This matches the metric reported in the original presentation:
    SimpleCNN: RF = k; DeepCNN/DepthwiseCNN: RF = 1 + 3(k - 1).

    Pooling layers are intentionally excluded so historical experiment values
    remain directly comparable with the presentation.
    """

    return 1 + sum(int(k) - 1 for k in kernel_sizes)


class SimpleCNN(nn.Module):
    """Shallow baseline with one kernel-size-dependent convolution."""

    def __init__(self, kernel_size: int, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.rf_kernel_sizes = [kernel_size]
        self.features = nn.Sequential(
            SamePadConv2d(in_channels, 32, kernel_size, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """Three-layer conventional CNN used as the higher-capacity model."""

    def __init__(self, kernel_size: int, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.rf_kernel_sizes = [kernel_size] * 3
        self.features = nn.Sequential(
            SamePadConv2d(in_channels, 32, kernel_size, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SamePadConv2d(32, 64, kernel_size, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SamePadConv2d(64, 128, kernel_size, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise spatial convolution followed by 1x1 channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            SamePadConv2d(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseCNN(nn.Module):
    """Three depthwise-separable blocks for parameter-efficient large kernels."""

    def __init__(self, kernel_size: int, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.rf_kernel_sizes = [kernel_size] * 3
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(in_channels, 32, kernel_size),
            DepthwiseSeparableBlock(32, 64, kernel_size),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(64, 128, kernel_size),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "DeepCNN": DeepCNN,
    "DepthwiseCNN": DepthwiseCNN,
}
