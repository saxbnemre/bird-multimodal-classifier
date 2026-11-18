"""Görüntü kodlayıcı modülü: EfficientNet-B0 tabanlı görsel özellik çıkarıcı (1280 -> 256)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """[B, 3, 224, 224] RGB görüntüyü 256 boyutlu gömme vektörüne dönüştürür.

    torchvision mevcutsa ön-eğitimli EfficientNet-B0 omurgasını kullanır;
    aksi halde aynı 1280 boyutlu çıkış arayüzünü sunan hafif konvolüsyonel omurga kullanır.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features  # 1280
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        except ImportError:
            in_features = 1280
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
                nn.Conv2d(96, 320, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(320),
                nn.SiLU(inplace=True),
                nn.Conv2d(320, in_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_features),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.projection = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.projection(feats)
