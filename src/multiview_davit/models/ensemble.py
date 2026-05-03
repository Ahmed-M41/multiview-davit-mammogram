"""Multi-view ensemble of DaViT models.

CBIS-DDSM: 4-model ensemble (DaViT-CC, DaViT-MLO, DaViT-ALL-CC, DaViT-ALL-MLO)
INbreast:   2-model ensemble (DaViT-CC, DaViT-MLO only)

DaViT-ALL is excluded from INbreast evaluation because it was trained on INbreast,
which would constitute data leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .davit import DaViTClassifier


class MultiViewEnsemble:
    """Ensemble that averages softmax probabilities across multiple DaViT models."""

    def __init__(
        self,
        cc_spec: DaViTClassifier,
        mlo_spec: DaViTClassifier,
        cc_all: DaViTClassifier | None = None,
        mlo_all: DaViTClassifier | None = None,
        device: str | torch.device = "cpu",
    ):
        self.cc_spec = cc_spec.to(device).eval()
        self.mlo_spec = mlo_spec.to(device).eval()
        self.cc_all = cc_all.to(device).eval() if cc_all is not None else None
        self.mlo_all = mlo_all.to(device).eval() if mlo_all is not None else None
        self.device = device

    @torch.inference_mode()
    def _probs(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(model(x.to(self.device)), dim=1)

    @torch.inference_mode()
    def predict_cbis(
        self, cc_img: torch.Tensor, mlo_img: torch.Tensor
    ) -> np.ndarray:
        """4-model ensemble: equal-weight average of CC-spec, MLO-spec, CC-all, MLO-all."""
        if self.cc_all is None or self.mlo_all is None:
            raise RuntimeError("cc_all and mlo_all checkpoints required for CBIS prediction.")
        p = (
            self._probs(self.cc_spec, cc_img)
            + self._probs(self.mlo_spec, mlo_img)
            + self._probs(self.cc_all, cc_img)
            + self._probs(self.mlo_all, mlo_img)
        ) / 4.0
        return p.cpu().numpy()

    @torch.inference_mode()
    def predict_inbreast(
        self, cc_img: torch.Tensor, mlo_img: torch.Tensor
    ) -> np.ndarray:
        """2-model ensemble: DaViT-CC and DaViT-MLO only.

        DaViT-ALL is explicitly excluded to prevent data leakage (it trained on INbreast).
        """
        p = (
            self._probs(self.cc_spec, cc_img)
            + self._probs(self.mlo_spec, mlo_img)
        ) / 2.0
        return p.cpu().numpy()

    @classmethod
    def from_config(cls, cfg, device: str = "cpu") -> "MultiViewEnsemble":
        """Load ensemble from checkpoint paths specified in ensemble.yaml."""

        def _load(path: str, head: str) -> DaViTClassifier:
            model = DaViTClassifier(head=head)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return model

        cc_spec = _load(cfg.checkpoints.cc_spec, head="single")
        mlo_spec = _load(cfg.checkpoints.mlo_spec, head="single")

        cc_all = mlo_all = None
        if hasattr(cfg.checkpoints, "cc_all") and cfg.checkpoints.cc_all:
            cc_all = _load(cfg.checkpoints.cc_all, head="two_layer")
            mlo_all = _load(cfg.checkpoints.mlo_all, head="two_layer")

        return cls(cc_spec=cc_spec, mlo_spec=mlo_spec, cc_all=cc_all, mlo_all=mlo_all, device=device)
