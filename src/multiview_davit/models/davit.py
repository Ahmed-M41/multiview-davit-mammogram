"""DaViT-Base classifier with configurable classification head.

Single consolidated class replacing three notebook variants.
Dropped from notebooks (not in paper):
  - tanh feature regularisation trick
  - logits/1.1 temperature scaling
  - SingleViewDaViT (dropout=0.7, freeze_stages=2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class DaViTClassifier(nn.Module):
    """DaViT-Base backbone with a task-specific classification head.

    Args:
        head: "single" (LayerNorm → Dropout(0.5) → Linear(1024→2))
              or "two_layer" (LayerNorm → Dropout(0.5) → Linear(1024→512) → GELU → Dropout(0.25) → Linear(512→2))
        freeze_stages: Number of early stages to freeze. 1 = freeze stem only.
        noise_std: Std of Gaussian noise added to features during training. 0 disables noise.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        head: str = "single",
        freeze_stages: int = 1,
        noise_std: float = 0.05,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "davit_base.msft_in1k", pretrained=True, num_classes=0
        )
        feat_dim: int = self.backbone.num_features  # 1024 for davit_base

        if freeze_stages >= 1:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False

        if head == "single":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(0.5),
                nn.Linear(feat_dim, num_classes),
            )
        elif head == "two_layer":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(0.5),
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head!r}. Use 'single' or 'two_layer'.")

        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.training and self.noise_std > 0:
            feats = feats + torch.randn_like(feats) * self.noise_std
        return self.head(feats)

    @classmethod
    def from_config(cls, cfg) -> "DaViTClassifier":
        return cls(
            head=cfg.model.head,
            freeze_stages=cfg.model.freeze_stages,
            noise_std=cfg.model.noise_std,
        )
