"""Confusion matrix evaluation and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

CLASS_NAMES = ["Benign", "Malignant"]


def confusion_matrix_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """Run inference and return confusion matrix metrics."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_true, all_pred)
    accuracy = accuracy_score(all_true, all_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(all_true, all_pred, target_names=class_names, digits=4))
    return {"cm": cm, "accuracy": accuracy, "y_true": all_true, "y_pred": all_pred}


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"}, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig
