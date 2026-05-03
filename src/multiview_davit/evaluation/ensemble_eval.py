"""Ensemble evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.ensemble import MultiViewEnsemble
from ..training.metrics import compute_metrics


def evaluate_ensemble(
    ensemble: MultiViewEnsemble,
    loader: DataLoader,
    device: str | torch.device,
    mode: str = "cbis",
) -> dict:
    """Evaluate the ensemble on a paired-view DataLoader.

    Args:
        mode: "cbis" (4-model) or "inbreast" (2-model).
    Returns dict with keys: accuracy, f1, auc, sensitivity, specificity, y_true, y_pred, y_prob.
    """
    all_true, all_pred, all_prob = [], [], []
    predict_fn = ensemble.predict_cbis if mode == "cbis" else ensemble.predict_inbreast

    for batch in loader:
        cc_imgs, mlo_imgs, labels = batch
        probs = predict_fn(cc_imgs, mlo_imgs)
        preds = np.argmax(probs, axis=1)
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_prob.extend(probs)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics.update({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    return metrics


def dump_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save per-image predictions to a CSV file."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_benign": y_prob[:, 0],
        "prob_malignant": y_prob[:, 1],
    })
    df.to_csv(str(save_path), index=False)
