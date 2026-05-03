# Multi-View Ensemble of Dual Attention Vision Transformers for Mammogram Classification

[![Paper](https://img.shields.io/badge/PeerJ_CS-AI_Application-blue)](https://peerj.com/computer-science/)
[![Data](https://img.shields.io/badge/FigShare-10.6084%2Fm9.figshare.32149885-orange)](https://doi.org/10.6084/m9.figshare.32149885)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the paper **"A multi-view ensemble of dual attention vision transformers for mammogram classification"** — PeerJ Computer Science (AI Application).

## Overview

A multi-view ensemble of DaViT-Base (Dual Attention Vision Transformer) models for binary mammogram classification (Benign vs. Malignant). CC-view and MLO-view mammograms are processed by specialist models whose softmax outputs are averaged at inference time.

### Results

| Dataset | Accuracy | Balanced Acc | F1 |
|---------|----------|--------------|----|
| CBIS-DDSM (4-model ensemble) | **83.33%** | — | **82.78%** |
| INbreast zero-shot (2-model) | **96.59%** | **95.35%** | — |

### Architecture

```
CC image ──► DaViT-CC  ──►─────────────────────► average ──► prediction
             DaViT-ALL(CC) ──►─────────────────►
MLO image ──► DaViT-MLO ──►────────────────────►
              DaViT-ALL(MLO) ──►────────────────►
```

- **DaViT-CC / DaViT-MLO**: View-specialist models, single-layer head, `betas=(0.9, 0.9)`
- **DaViT-ALL**: All-views + INbreast generalist, two-layer head, CosineAnnealingWarmRestarts
- **INbreast inference**: 2-model only (DaViT-ALL excluded to prevent data leakage)

---

## Installation

```bash
git clone https://github.com/Ahmed-M41/multiview-davit-mammogram.git
cd multiview-davit-mammogram
pip install -e .
```

---

## Data

| Dataset | Source | Licence |
|---------|--------|---------|
| CBIS-DDSM | [TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) — DOI: 10.7937/K9/TCIA.2016.7O02S9CY | CC BY 3.0 |
| INbreast | DOI: 10.1016/j.acra.2011.09.014 | Academic use |
| Preprocessed archives | [FigShare 10.6084/m9.figshare.32149885](https://doi.org/10.6084/m9.figshare.32149885) | CC BY 4.0 |

See [data/README.md](data/README.md) for directory layout and download instructions.

---

## Quickstart

### 1. Preprocess images

```bash
# Download YOLOv5 breast detector weights
bash scripts/download_yolo_weights.sh

# Run preprocessing pipeline
python -m multiview_davit.cli.preprocess \
  --src data/raw/cbis_ddsm \
  --dst data/processed/cbis_ddsm \
  --weights checkpoints/yolo/best.pt \
  --yolov5-path /path/to/yolov5
```

### 2. Build CSV metadata

```bash
python -m multiview_davit.cli.build_csvs \
  --cbis-raw data/raw/cbis_ddsm \
  --cbis-processed data/processed/cbis_ddsm \
  --out data/splits/
```

### 3. Create train/val split (patient-level)

```bash
python -m multiview_davit.cli.make_splits \
  --csv data/splits/cbis_train_full.csv \
  --out data/splits/
```

### 4. Train

```bash
python -m multiview_davit.cli.train --config configs/davit_cc.yaml
python -m multiview_davit.cli.train --config configs/davit_mlo.yaml
python -m multiview_davit.cli.train --config configs/davit_all.yaml
```

### 5. Evaluate

```bash
# 4-model ensemble on CBIS-DDSM test set
python -m multiview_davit.cli.evaluate \
  --ensemble-config configs/ensemble.yaml \
  --dataset cbis_test \
  --output results/

# 2-model zero-shot on INbreast
python -m multiview_davit.cli.evaluate \
  --ensemble-config configs/ensemble.yaml \
  --dataset inbreast \
  --output results/
```

### 6. Single-image inference

```bash
python -m multiview_davit.cli.inference \
  --cc-image path/to/cc.jpg \
  --mlo-image path/to/mlo.jpg \
  --ensemble-config configs/ensemble.yaml \
  --dataset cbis
```

---

## Repository Structure

```
multiview-davit-mammogram/
├── configs/                  # YAML hyperparameter configs (paper-accurate)
│   ├── davit_cc.yaml
│   ├── davit_mlo.yaml
│   ├── davit_all.yaml
│   ├── ensemble.yaml
│   ├── data.yaml
│   └── preprocess.yaml
├── codebook/                 # INbreast BI-RADS codebook and class distribution
├── data/                     # Data placement guide (images not included)
├── checkpoints/              # Model weights (download from Zenodo)
├── src/multiview_davit/
│   ├── config.py             # YAML loading
│   ├── seed.py               # Reproducibility
│   ├── data/                 # CSV building, patient splits, datasets, transforms
│   ├── preprocessing/        # YOLOv5 ROI, CLAHE, pipeline
│   ├── models/               # DaViTClassifier + MultiViewEnsemble
│   ├── training/             # Metrics, early stopping, training loop, multi-run
│   ├── evaluation/           # Confusion matrix, stats, reports
│   └── cli/                  # Command-line entry points
├── notebooks/                # Jupyter demonstration notebooks (01–09)
├── scripts/                  # Download utilities
├── tests/                    # Smoke tests (pytest)
└── docs/
    └── reproducibility.md    # Paper vs notebook discrepancies
```

---

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for a full list of 10 discrepancies between the original experimental notebooks and the paper. The repository implements the paper's specification in all cases.

---

## Citation

```bibtex
@article{multiview_davit_2025,
  title   = {A multi-view ensemble of dual attention vision transformers for mammogram classification},
  journal = {PeerJ Computer Science},
  year    = {2025},
  doi     = {10.5281/zenodo.20016511},
}
```

Preprocessed data:

```bibtex
@misc{figshare_mammogram_2025,
  title  = {Preprocessed mammogram archives and metadata for multi-view DaViT ensemble},
  doi    = {10.6084/m9.figshare.32149885},
  year   = {2025},
}
```

---

## License

[MIT](LICENSE)
