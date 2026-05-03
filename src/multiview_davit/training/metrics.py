"""Classification metrics for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: list | np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def composite_score(metrics: dict[str, float], val_loss: float = 0.0) -> float:
    """Composite score for model selection and early stopping.

    Formula: 0.5*acc + 0.2*auc + 0.2*f1 + 0.1*(1 - loss)
    """
    return (
        0.5 * metrics["accuracy"]
        + 0.2 * metrics["auc"]
        + 0.2 * metrics["f1"]
        + 0.1 * (1.0 - val_loss)
    )
