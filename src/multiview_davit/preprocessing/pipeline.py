"""End-to-end preprocessing pipeline: YOLO crop → threshold → resize → save JPEG.

CLAHE is intentionally NOT applied here (offline). It is applied online during
training and inference via the albumentations transforms in data/transforms.py.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .naming import dst_from_src
from .postprocess import to_u8
from .roi import read_img, keep_breast_roi
from .yolo_runner import YoloV5Runner, load_yolov5

_model_lock = threading.Lock()


def process_one(
    src: str | Path,
    dst: str | Path,
    runner: YoloV5Runner,
    out_size: int = 512,
    threshold_value: int = 20,
) -> tuple[str, str]:
    """Process a single image through the full pipeline.

    Returns (src, status) where status is one of: 'ok', 'missing', 'read_fail', 'write_fail'.
    """
    src = str(src)
    dst = str(dst)

    if not os.path.exists(src):
        return src, "missing"

    try:
        with _model_lock:
            crop = keep_breast_roi(src, model=runner)

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        img = to_u8(img)
    except Exception as e:
        return src, "read_fail"

    # Threshold to suppress background noise — CLAHE is NOT applied offline
    img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TOZERO)[1]
    img_out = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = cv2.imwrite(dst, img_out)
    return src, "ok" if ok else "write_fail"


def run_pipeline(
    src_dir: str | Path,
    dst_dir: str | Path,
    weights_path: str | Path,
    out_size: int = 512,
    threshold_value: int = 20,
    yolov5_path: str | None = None,
    max_workers: int = 4,
) -> dict[str, list[str]]:
    """Run the preprocessing pipeline over all images in src_dir.

    Returns a dict with keys 'ok', 'missing', 'read_fail', 'write_fail'.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".dcm", ".jpg", ".jpeg", ".png"}
    src_paths = [p for p in src_dir.rglob("*") if p.suffix.lower() in extensions]

    runner = load_yolov5(weights_path, yolov5_path=yolov5_path)

    results: dict[str, list[str]] = {"ok": [], "missing": [], "read_fail": [], "write_fail": []}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _worker(src_path: Path):
        dst = dst_from_src(str(src_path), str(dst_dir))
        return process_one(str(src_path), dst, runner, out_size, threshold_value)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, p): p for p in src_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            src, status = fut.result()
            results[status].append(src)

    return results
