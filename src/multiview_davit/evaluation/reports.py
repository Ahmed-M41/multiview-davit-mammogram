"""Training history visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_and_save_metrics(history: dict, save_dir: str | Path, case_name: str = "") -> None:
    """Plot train/val loss and metric curves and save to save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_keys = {k.replace("train_", "") for k in history if k.startswith("train_")}
    val_keys = {k.replace("val_", "") for k in history if k.startswith("val_")}
    metric_names = sorted(train_keys & val_keys)

    n = len(metric_names)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.ravel() if n > 1 else [axes]
    title = f"Training metrics — {case_name}" if case_name else "Training metrics"
    fig.suptitle(title, fontsize=14)

    for ax, metric in zip(axes, metric_names):
        ax.plot(history[f"train_{metric}"], label=f"Train {metric}")
        ax.plot(history[f"val_{metric}"], label=f"Val {metric}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    filename = f"{case_name}_metrics.png" if case_name else "metrics.png"
    fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
