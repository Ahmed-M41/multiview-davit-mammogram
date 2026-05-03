"""CLI: Run inference on a single CC+MLO image pair."""

import argparse

import numpy as np
import torch

from ..config import load_config
from ..data.transforms import build_inference_transform
from ..models.ensemble import MultiViewEnsemble

import cv2


def _load_image(path: str, transform) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = np.stack([img, img, img], axis=-1)
    tensor = transform(image=img)["image"]
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Single-pair DaViT ensemble inference")
    parser.add_argument("--cc-image", required=True, help="Path to CC-view image")
    parser.add_argument("--mlo-image", required=True, help="Path to MLO-view image")
    parser.add_argument("--ensemble-config", default="configs/ensemble.yaml")
    parser.add_argument("--dataset", choices=["cbis", "inbreast"], default="cbis",
                        help="'cbis' uses 4-model ensemble; 'inbreast' uses 2-model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.ensemble_config)
    ensemble = MultiViewEnsemble.from_config(cfg, device=args.device)
    transform = build_inference_transform(cfg.inference.input_size)

    cc_tensor = _load_image(args.cc_image, transform)
    mlo_tensor = _load_image(args.mlo_image, transform)

    predict_fn = ensemble.predict_cbis if args.dataset == "cbis" else ensemble.predict_inbreast
    probs = predict_fn(cc_tensor, mlo_tensor)[0]

    print(f"\nPrediction:")
    print(f"  Benign probability:    {probs[0]:.4f}")
    print(f"  Malignant probability: {probs[1]:.4f}")
    print(f"  Prediction: {'MALIGNANT' if probs[1] > 0.5 else 'BENIGN'}")


if __name__ == "__main__":
    main()
