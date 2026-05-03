"""Breast ROI extraction utilities."""

from __future__ import annotations

import cv2
import numpy as np


def read_img(path: str) -> np.ndarray:
    """Read a DICOM or JPEG image and return a numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            img = ds.pixel_array
        except Exception:
            raise FileNotFoundError(f"Could not read image: {path}")
    return img


def square_around(
    x1: int, y1: int, x2: int, y2: int,
    W: int, H: int,
    min_side: int | None = None,
) -> tuple[int, int, int, int]:
    """Expand a bounding box to a square centred on the same centroid."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    if min_side is not None:
        side = max(side, min_side)

    x1 = int(round(cx - side / 2))
    x2 = int(round(cx + side / 2))
    y1 = int(round(cy - side / 2))
    y2 = int(round(cy + side / 2))

    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)

    return x1, y1, x2, y2


def keep_breast_roi(
    img_or_path: str | np.ndarray,
    model,
    size: int = 640,
) -> np.ndarray:
    """Detect the breast ROI with YOLO and return the cropped region.

    Falls back to the full image if no detection is found.
    """
    img = read_img(img_or_path) if isinstance(img_or_path, str) else img_or_path
    H, W = img.shape[:2]
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    det = model(bgr, size=size).numpy()
    if det.shape[0] == 0:
        return img

    # Take highest-confidence detection
    det = det[det[:, 4].argsort()[::-1]]
    x1, y1, x2, y2 = map(int, det[0, :4])
    x1, y1, x2, y2 = square_around(x1, y1, x2, y2, W, H)
    return img[y1:y2, x1:x2]
