"""Image postprocessing utilities: normalisation, masking, CLAHE."""

import cv2
import numpy as np


def to_u8(img: np.ndarray) -> np.ndarray:
    """Normalise any dtype image to uint8 [0, 255]."""
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def keep_largest_object(img_u8: np.ndarray, p_low: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the largest connected bright component.

    Returns (masked_image, mask).
    """
    t = int(max(5, np.percentile(img_u8, p_low)))
    _, mask = cv2.threshold(img_u8, t, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        m = np.ones_like(mask) * 255
        largest_area = max(cv2.contourArea(c) for c in cnts)
        small = [c for c in cnts if cv2.contourArea(c) < largest_area]
        cv2.drawContours(m, small, -1, 0, -1)
    else:
        m = np.ones_like(mask) * 255

    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    return cv2.bitwise_and(img_u8, m), m


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    img = to_u8(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)
