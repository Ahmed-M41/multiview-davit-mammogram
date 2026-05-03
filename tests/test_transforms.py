"""Test that inference transform is deterministic."""

import numpy as np
import pytest

from multiview_davit.data.transforms import build_inference_transform, build_train_transform


def _random_image(h: int = 512, w: int = 512) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return img


def test_inference_transform_deterministic():
    img = _random_image()
    transform = build_inference_transform(224)
    out1 = transform(image=img)["image"]
    out2 = transform(image=img)["image"]
    assert (out1 == out2).all(), "Inference transform should be deterministic"


def test_inference_output_shape():
    img = _random_image()
    transform = build_inference_transform(224)
    out = transform(image=img)["image"]
    assert out.shape == (3, 224, 224), f"Expected (3, 224, 224), got {out.shape}"


def test_train_transform_output_shape():
    img = _random_image()
    transform = build_train_transform(224)
    out = transform(image=img)["image"]
    assert out.shape == (3, 224, 224), f"Expected (3, 224, 224), got {out.shape}"
