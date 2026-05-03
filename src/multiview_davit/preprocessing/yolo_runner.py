"""YOLOv5 breast ROI detector — hub-free loading via DetectMultiBackend.

Requires the YOLOv5 repository to be available on sys.path. Set the
YOLOV5_PATH environment variable or pass yolov5_path to load_yolov5().
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch


def _add_yolov5_to_path(yolov5_path: str | None = None) -> None:
    path = yolov5_path or os.environ.get("YOLOV5_PATH", "")
    if path and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _register_safe_globals() -> None:
    from torch.serialization import add_safe_globals
    from torch.nn import (
        Sequential, ModuleList, Identity, Dropout,
        Conv2d, BatchNorm2d, ReLU, LeakyReLU, SiLU,
        Upsample, MaxPool2d, AdaptiveAvgPool2d,
    )
    try:
        from models.yolo import DetectionModel, Detect
        from models.common import (
            Conv, Bottleneck, C3, SPPF, Concat,
            C3Ghost, GhostConv, GhostBottleneck,
        )
        add_safe_globals([
            Sequential, ModuleList, Identity, Dropout,
            Conv2d, BatchNorm2d, ReLU, LeakyReLU, SiLU,
            Upsample, MaxPool2d, AdaptiveAvgPool2d,
            DetectionModel, Detect,
            Conv, Bottleneck, C3, SPPF, Concat,
            C3Ghost, GhostConv, GhostBottleneck,
        ])
    except ImportError:
        pass  # YOLOv5 not yet on path; will fail later with a clear message


class YoloV5Runner:
    """Thin wrapper around YOLOv5 DetectMultiBackend for breast ROI detection."""

    def __init__(
        self,
        weights: str | Path,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        yolov5_path: str | None = None,
    ):
        _add_yolov5_to_path(yolov5_path)
        _register_safe_globals()

        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        self.device = select_device("" if device is None else device)
        self.backend = DetectMultiBackend(str(weights), device=self.device, dnn=False, fp16=False)
        self.backend.model.eval()
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    @torch.inference_mode()
    def __call__(self, im_bgr: np.ndarray, size: int | None = None) -> torch.Tensor:
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        use_size = int(size) if size is not None else self.img_size
        s = self.backend.stride
        stride = int(s if isinstance(s, (int, float)) else max(s))

        img = letterbox(im_bgr, new_shape=use_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        pred = self.backend(img)
        det = non_max_suppression(pred, self.conf, self.iou, agnostic=False, max_det=300)[0]
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im_bgr.shape).round()
        return det.cpu()


def load_yolov5(
    weights_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    yolov5_path: str | None = None,
) -> YoloV5Runner:
    return YoloV5Runner(
        weights=weights_path,
        device=device,
        conf=conf,
        iou=iou,
        yolov5_path=yolov5_path,
    )
