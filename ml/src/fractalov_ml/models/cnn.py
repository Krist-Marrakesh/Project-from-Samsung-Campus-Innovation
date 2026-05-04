"""Two-headed CNN baseline for fractalov.

Goal: from a 3-channel fractal image, predict
  (a) which family produced it — 4-way softmax classifier
  (b) for Julia, the value of c = (c_re, c_im) — 2D regression

A small ResNet-style backbone is plenty: the dataset is on the order of 10⁴
images, and adding capacity past ~1M params trades training speed for marginal
val-loss gains. The body uses an adaptive average pool, so the same checkpoint
runs on 96×96, 128×128, or 256×256 input without surgery.

Two heads share the backbone. The regression head's output is dimensionless;
training masks its loss to Julia rows so Mandelbrot/Burning-Ship/Multibrot do
not pull `c` toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelOutput:
    family_logits: torch.Tensor  # (B, num_families)
    c_pred: torch.Tensor          # (B, 2) — predicted (cRe, cIm)


def _norm(num_channels: int) -> nn.Module:
    """GroupNorm instead of BatchNorm.

    BatchNorm relies on running mean/var statistics that, on MPS (PyTorch 2.x),
    can desync between train and eval modes — a checkpoint that gets 0.93 val
    accuracy in train.py reloads as a 0.25 trivial classifier in eval.py.
    GroupNorm computes statistics per-sample, has no running buffers, and
    produces identical numerics on MPS and CPU. The cost is a small constant
    factor in throughput; the win is checkpoints that actually work.

    Group count = 8 (or fewer when num_channels < 8); 8 is the de facto default
    for ResNet-style stacks at the channel widths we use here.
    """
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ResidualBlock(nn.Module):
    """Two 3x3 convs with optional stride-2 downsample on the first conv."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(out_ch)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class FractalCNN(nn.Module):
    def __init__(
        self,
        num_families: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        # Stem: 3 -> c1, then halve spatially.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=False),
            _norm(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = _ResidualBlock(c1, c2, stride=2)
        self.layer2 = _ResidualBlock(c2, c3, stride=2)
        self.layer3 = _ResidualBlock(c3, c4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.family_head = nn.Linear(c4, num_families)
        # c regression: tanh keeps outputs in [-1, 1], we scale by 1.5 to match
        # the disk |c| <= 1.5 sampling range used by the dataset generator.
        self.c_head = nn.Sequential(
            nn.Linear(c4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return ModelOutput(
            family_logits=self.family_head(x),
            c_pred=self.c_head(x) * 1.5,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
