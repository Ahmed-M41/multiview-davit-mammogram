"""CLI: Train a single DaViT model (CC, MLO, or ALL)."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..seed import set_seed
from ..data.datasets import MedicalImageDataset
from ..data.transforms import build_train_transform, build_inference_transform
from ..training.multi_run import multi_run_training
from ..evaluation.reports import plot_and_save_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a DaViT mammogram classifier")
    parser.add_argument("--config", required=True, help="Model config YAML (e.g. configs/davit_cc.yaml)")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--checkpoint-dir", default="checkpoints/", help="Directory to save .pth files")
    parser.add_argument("--results-dir", default="results/", help="Directory for training curves")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    input_size = cfg.training.input_size
    train_transform = build_train_transform(input_size)
    val_transform = build_inference_transform(input_size)

    train_ds = MedicalImageDataset(data_cfg.cbis_ddsm.train_csv, transform=train_transform)
    val_ds = MedicalImageDataset(data_cfg.cbis_ddsm.val_csv, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / cfg.checkpoint_name

    best_model, history, best_score = multi_run_training(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_path=checkpoint_path,
    )

    plot_and_save_metrics(history, save_dir=args.results_dir, case_name=cfg.checkpoint_name)
    print(f"\nBest composite score: {best_score:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
