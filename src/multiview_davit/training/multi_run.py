"""Multi-run training loop: run up to max_runs times, stop after max_no_improve consecutive non-improvements."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.davit import DaViTClassifier
from .early_stopping import CompositeEarlyStopper
from .loop import train_model, build_optimizer, build_scheduler


def multi_run_training(
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    checkpoint_path: str | Path | None = None,
    max_runs: int = 4,
    max_no_improve: int = 2,
) -> tuple[DaViTClassifier, dict, float]:
    """Train the model multiple times and keep the best run.

    Stops early after `max_no_improve` consecutive runs without improvement.
    Returns (best_model, best_history, best_score).
    """
    best_score = -float("inf")
    best_model = None
    best_history = None
    no_improve_count = 0

    for run in range(1, max_runs + 1):
        print(f"\n{'='*20} Run {run}/{max_runs} {'='*20}")

        model = DaViTClassifier.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        stopper = CompositeEarlyStopper(patience=cfg.early_stopping.patience)

        trained_model, history, score = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            stopper=stopper,
            cfg=cfg,
            device=device,
            checkpoint_path=checkpoint_path,
        )

        print(f"Run {run} score: {score:.4f} | Best so far: {best_score:.4f}")

        if score > best_score:
            best_score = score
            best_model = trained_model
            best_history = history
            no_improve_count = 0
            print(f"New best score: {best_score:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement ({no_improve_count}/{max_no_improve})")

        if no_improve_count >= max_no_improve:
            print("Stopping multi-run training.")
            break

    return best_model, best_history, best_score
