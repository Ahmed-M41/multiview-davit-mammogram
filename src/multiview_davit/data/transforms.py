"""Albumentations-based image transforms for training and inference.

CLAHE is applied ONLINE (here) only — the preprocessing pipeline does NOT
apply CLAHE offline so that the stochastic augmentation is always active.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomCLAHE(A.ImageOnlyTransform):
    """Stochastic CLAHE with randomised clip limit and tile size."""

    def __init__(
        self,
        clip_range: tuple[float, float] = (2.0, 5.0),
        tiles_choices: tuple[int, ...] = (4, 6, 8),
        p: float = 0.9,
    ):
        super().__init__(p=p)
        self.clip_range = clip_range
        self.tiles_choices = tiles_choices

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        clip = float(np.random.uniform(*self.clip_range))
        tile = int(np.random.choice(self.tiles_choices))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        if img.ndim == 2:
            return clahe.apply(img)
        channels = [clahe.apply(img[:, :, c]) for c in range(img.shape[2])]
        return cv2.merge(channels)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("clip_range", "tiles_choices")


class FixedCLAHE(A.ImageOnlyTransform):
    """Deterministic CLAHE for inference."""

    def __init__(self, clip: float = 3.0, tile_size: int = 8, p: float = 1.0):
        super().__init__(p=p)
        self.clip = clip
        self.tile_size = tile_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile_size, self.tile_size))
        if img.ndim == 2:
            return clahe.apply(img)
        channels = [clahe.apply(img[:, :, c]) for c in range(img.shape[2])]
        return cv2.merge(channels)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("clip", "tile_size")


def build_train_transform(input_size: int = 224) -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.GaussNoise(var_limit=(5, 25), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.3), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.Blur(blur_limit=(1, 2), p=0.4),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.3),
            A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(0.5, 2.0), alpha=(0.1, 0.3), p=0.3),
        ], p=0.2),
        RandomCLAHE(clip_range=(2.0, 5.0), tiles_choices=(4, 6, 8), p=1.0),
        A.Resize(input_size, input_size),
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.7, 1.0),
            ratio=(0.6, 0.85),
            interpolation=cv2.INTER_LINEAR,
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.RandomToneCurve(scale=0.2, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.08, 0.15),
                contrast_limit=(-0.3, 0.4),
                brightness_by_max=True,
                p=0.5,
            ),
        ], p=0.4),
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit_x=(-0.08, 0.08),
                shift_limit_y=(-0.15, 0.15),
                scale_limit=(-0.12, 0.12),
                rotate_limit=(-25, 25),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                rotate_method="largest_box",
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=0.8,
                sigma=18,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.15,
            ),
            A.GridDistortion(
                num_steps=4,
                distort_limit=0.25,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                normalized=True,
                p=0.15,
            ),
        ], p=0.5),
        A.CoarseDropout(
            max_holes=4,
            max_height=0.12,
            max_width=0.2,
            min_holes=1,
            min_height=0.04,
            min_width=0.08,
            fill_value=0,
            p=0.2,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_inference_transform(input_size: int = 224) -> A.Compose:
    return A.Compose([
        FixedCLAHE(clip=4.0, tile_size=6, p=1.0),
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
