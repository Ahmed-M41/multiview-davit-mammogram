"""PyTorch Dataset classes for single-view and paired-view mammogram loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    """Single-view grayscale mammogram dataset."""

    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        path_col: str = "image_path",
        label_col: str = "cancer",
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.at[idx, self.path_col]
        label = int(self.df.at[idx, self.label_col])

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read: {img_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Pseudo-RGB so ImageNet-pretrained models get 3-channel input
        img = np.stack([img, img, img], axis=-1)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)


class PairedViewDataset(Dataset):
    """Returns (cc_img, mlo_img, label) tuples for multi-view ensemble training."""

    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        cc_col: str = "cc_path",
        mlo_col: str = "mlo_path",
        label_col: str = "cancer",
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.cc_col = cc_col
        self.mlo_col = mlo_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)

    def __getitem__(self, idx: int):
        cc_img = self._load(self.df.at[idx, self.cc_col])
        mlo_img = self._load(self.df.at[idx, self.mlo_col])
        label = int(self.df.at[idx, self.label_col])

        if self.transform:
            cc_img = self.transform(image=cc_img)["image"]
            mlo_img = self.transform(image=mlo_img)["image"]

        return cc_img, mlo_img, torch.tensor(label, dtype=torch.long)


def restructure_per_image_to_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-image rows (one CC + one MLO per patient/breast) into paired rows.

    Input columns expected: patient_id, breast, view, image_path, cancer
    Output columns: patient_id, breast, cc_path, mlo_path, cancer
    """
    records = []
    group_keys = ["patient_id", "breast"] if "breast" in df.columns else ["patient_id"]
    for key, grp in df.groupby(group_keys):
        cc_rows = grp[grp["view"].str.upper() == "CC"]
        mlo_rows = grp[grp["view"].str.upper() == "MLO"]
        for _, cc_row in cc_rows.iterrows():
            for _, mlo_row in mlo_rows.iterrows():
                if cc_row["cancer"] == mlo_row["cancer"]:
                    rec = {
                        "patient_id": cc_row["patient_id"],
                        "cc_path": cc_row["image_path"],
                        "mlo_path": mlo_row["image_path"],
                        "cancer": cc_row["cancer"],
                    }
                    if "breast" in cc_row:
                        rec["breast"] = cc_row["breast"]
                    records.append(rec)
    return pd.DataFrame(records)
