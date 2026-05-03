"""Core training loop with AMP, gradient clipping, and early stopping."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, composite_score
from .early_stopping import CompositeEarlyStopper


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    stopper: CompositeEarlyStopper,
    cfg,
    device: str | torch.device = "cuda",
    checkpoint_path: str | Path | None = None,
) -> tuple[nn.Module, dict, float]:
    """Train model with AMP, gradient clipping, and composite-score early stopping.

    Returns (model_with_best_weights, history, best_composite_score).
    """
    model = model.to(device)
    use_amp = str(device) != "cpu" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    history: dict[str, list] = defaultdict(list)
    max_epochs = cfg.training.max_epochs
    grad_clip = cfg.training.grad_clip_norm

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{max_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        train_loss = epoch_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(all_labels, all_preds, all_probs)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        # Scheduler step
        sched_type = cfg.scheduler.type
        if sched_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Record history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        score = composite_score(val_metrics, val_loss)

        print(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} | "
            f"Score={score:.4f}"
        )

        should_stop = stopper.step(score, model)
        if stopper.best_score == score and checkpoint_path is not None:
            torch.save(model.state_dict(), str(checkpoint_path))

        if should_stop:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    stopper.restore_best(model)
    return model, dict(history), stopper.best_score


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg.optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """Build LR scheduler from config."""
    sched_cfg = cfg.scheduler
    stype = sched_cfg.type

    if stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.factor,
            patience=sched_cfg.patience,
            min_lr=sched_cfg.min_lr,
        )
    elif stype == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.T_0,
            T_mult=sched_cfg.T_mult,
            eta_min=sched_cfg.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")
