# Reproducibility Notes

This document lists all known discrepancies between the experimental notebooks and the paper. In every case, **the repository follows the paper**.

Expected metric differences when re-training from scratch are ±0.5–1% due to GPU non-determinism and data loading order.

---

## 1. Validation Split Strategy

**Notebooks**: Use the official CBIS-DDSM test set (361 images) as the validation set during training.

**Paper / Repo**: 80/20 patient-level stratified holdout from the training data, implemented via `GroupShuffleSplit` in `data/splits.py`.

*Impact*: Slightly different reported numbers expected when retraining.

---

## 2. DaViT-CC / DaViT-MLO Optimizer Betas

**Notebooks**: `betas=(0.9, 0.9999)`, `eps=1e-8`

**Paper / Repo**: `betas=(0.9, 0.9)`, `eps=1e-7` (configs/davit_cc.yaml, configs/davit_mlo.yaml)

---

## 3. DaViT-ALL Scheduler

**Notebooks (active code)**: `ReduceLROnPlateau`

**Paper / Repo**: `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-7)` (configs/davit_all.yaml)

---

## 4. Inference Resize Resolution

**Notebooks**: `test_transform` resizes to 288×288.

**Paper / Repo**: 224×224 (per paper §3.4 and `build_inference_transform(224)`).

---

## 5. Feature Regularisation Tricks

The following two tricks appear in notebook cells but are not described in the paper and were discarded before the final experiments:

- `feats = feats + torch.tanh(feats * 0.1) * 0.2`
- `logits = logits / 1.1`

Both are **absent** from `models/davit.py`.

---

## 6. DaViT-ALL Excluded from INbreast Evaluation

DaViT-ALL is trained partly on INbreast data. Including it in INbreast evaluation would constitute data leakage. `MultiViewEnsemble.predict_inbreast()` hard-enforces the 2-model (CC-spec + MLO-spec) path. See `tests/test_ensemble_shape.py`.

---

## 7. SingleViewDaViT Variant Omitted

A `SingleViewDaViT` variant with `dropout=0.7` and `freeze_stages=2` exists in the notebooks but is not described in the paper and produced the weakest checkpoints. It is **not** present in `models/davit.py`.

---

## 8. CLAHE Applied Online Only

In the notebooks, the `apply_clahe()` call inside `process_one` is commented out. This means offline preprocessing does NOT apply CLAHE. CLAHE is applied exclusively by the online albumentations transforms (`RandomCLAHE` during training, `FixedCLAHE` during inference). This behaviour is preserved exactly.

---

## 9. Albumentations API Update

Notebooks use Albumentations v1 keyword arguments that emit deprecation warnings (e.g. `always_apply`, old `ElasticTransform` signature). The repo uses the v1.4.x API (`p=` parameter, updated transform kwargs), pinned in `requirements.txt`.

---

## 10. calc + mass CSVs Both Included

Notebooks load only `mass_*` description CSVs for CBIS-DDSM. The paper states both masses and calcifications are included in evaluation. `data/csv_builder.py:build_cbis_full_csv` reads **both** `mass_*` and `calc_*` description CSVs and concatenates them before deduplication.
