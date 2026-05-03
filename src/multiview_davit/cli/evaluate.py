"""CLI: Evaluate the multi-view ensemble on CBIS-DDSM or INbreast."""

import argparse
from pathlib import Path

import torch

from ..config import load_config
from ..data.datasets import PairedViewDataset
from ..data.transforms import build_inference_transform
from ..models.ensemble import MultiViewEnsemble
from ..evaluation.ensemble_eval import evaluate_ensemble, dump_predictions_csv
from ..evaluation.confusion import plot_confusion_matrix
from ..evaluation.stats import wilson_ci
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-view DaViT ensemble")
    parser.add_argument("--ensemble-config", default="configs/ensemble.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset", choices=["cbis_test", "inbreast"], required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ens_cfg = load_config(args.ensemble_config)
    data_cfg = load_config(args.data_config)

    ensemble = MultiViewEnsemble.from_config(ens_cfg, device=args.device)

    input_size = ens_cfg.inference.input_size
    transform = build_inference_transform(input_size)

    csv_path = (
        data_cfg.cbis_ddsm.test_csv if args.dataset == "cbis_test"
        else data_cfg.inbreast.csv_path
    )
    mode = "cbis" if args.dataset == "cbis_test" else "inbreast"

    ds = PairedViewDataset(csv_path, transform=transform)
    loader = DataLoader(ds, batch_size=ens_cfg.inference.batch_size, shuffle=False, num_workers=4)

    print(f"Evaluating on {args.dataset} ({len(ds)} paired examples) ...")
    metrics = evaluate_ensemble(ensemble, loader, device=args.device, mode=mode)

    n_correct = int(round(metrics["accuracy"] * len(ds)))
    lo, hi = wilson_ci(n_correct, len(ds))

    print(f"\n{'='*40}")
    print(f"Accuracy:          {metrics['accuracy']*100:.2f}%  (95% CI: [{lo*100:.2f}%, {hi*100:.2f}%])")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"F1 (weighted):     {metrics['f1']:.4f}")
    print(f"AUC:               {metrics['auc']:.4f}")
    print(f"Sensitivity:       {metrics['sensitivity']:.4f}")
    print(f"Specificity:       {metrics['specificity']:.4f}")

    dump_predictions_csv(
        metrics["y_true"], metrics["y_pred"], metrics["y_prob"],
        save_path=out_dir / f"{args.dataset}_predictions.csv",
    )
    plot_confusion_matrix(
        metrics["cm"] if "cm" in metrics else None,
        save_path=out_dir / f"{args.dataset}_confusion.png",
    )
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
