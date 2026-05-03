"""Smoke test for MedicalImageDataset."""

import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from multiview_davit.data.datasets import MedicalImageDataset
from multiview_davit.data.transforms import build_inference_transform


def _create_fake_dataset(tmp_dir: str, n: int = 4):
    paths, labels = [], []
    for i in range(n):
        p = f"{tmp_dir}/img_{i}.jpg"
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(p)
        paths.append(p)
        labels.append(i % 2)
    df = pd.DataFrame({"image_path": paths, "cancer": labels})
    csv_path = f"{tmp_dir}/meta.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, len(paths)


def test_dataset_length_and_output():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, n = _create_fake_dataset(tmp)
        ds = MedicalImageDataset(csv_path, transform=build_inference_transform(224))
        assert len(ds) == n
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert label.item() in (0, 1)
