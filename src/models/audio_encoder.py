"""Ses kodlayıcı modülü: 128-bin Log-Mel spektrogram için 2D CNN ([B, 1, 128, 151] -> 256)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """4 konvolüsyon bloğu + Global Average Pooling ile 256 boyutlu akustik temsil üretir."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        channels = [1, 32, 64, 128, 256]
        blocks = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
        self.conv_net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_net(x)
        h = self.pool(h)
        return self.projection(h)
