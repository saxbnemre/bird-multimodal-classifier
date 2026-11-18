"""Multimodal füzyon modeli: Görüntü (256) + Ses (256) birleştirme (512 -> 256 -> num_classes)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.audio_encoder import AudioEncoder
from src.models.image_encoder import ImageEncoder


class MultimodalBirdClassifier(nn.Module):
    """Görsel ve işitsel gömme vektörlerini birleştirerek tür sınıflandırması yapar."""

    def __init__(
        self,
        num_classes: int = 33,
        embed_dim: int = 256,
        dropout: float = 0.3,
        pretrained_image: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained_image)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        fused_dim = embed_dim * 2  # 512

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_img = self.image_encoder(image)
        v_aud = self.audio_encoder(audio)
        fused = torch.cat([v_img, v_aud], dim=1)
        logits = self.fusion_head(fused)
        return logits, v_img, v_aud
