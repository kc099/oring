"""
PatchCore Dataset
=================
PyTorch Dataset and DataLoader utilities for loading binned 720×720
O-ring BMP images, resizing to 256 and center-cropping to 224.

Author: GitHub Copilot
Date:   February 27, 2026
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SIZE,
    CENTER_CROP_SIZE,
    ModelConfig,
)


# ─── Transforms ──────────────────────────────────────────────────────────

def get_transform(resize: int = RESIZE_SIZE,
                  center_crop: int = CENTER_CROP_SIZE,
                  is_train: bool = True) -> transforms.Compose:
    """Standard PatchCore transform: resize → center-crop → normalize."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────

class OringDataset(Dataset):
    """Load O-ring images from a directory, with optional label.

    Parameters
    ----------
    root : Path
        Directory containing .bmp images.
    label : int
        0 = good (normal), 1 = defect (anomaly).
    label_name : str
        Human-readable label name for logging.
    transform : callable, optional
        Torchvision transform pipeline.  If ``None`` the default
        PatchCore transform is used.
    """

    def __init__(
        self,
        root: Path,
        label: int = 0,
        label_name: str = "good",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.label = label
        self.label_name = label_name
        self.transform = transform or get_transform()
        self.image_paths = self._collect_images()

    def _collect_images(self) -> List[Path]:
        paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.root} with extensions {IMAGE_EXTENSIONS}"
            )
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        # Read with OpenCV (BGR) → convert to RGB numpy uint8
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
        img = self.transform(img)                     # (3, 224, 224)
        return img, self.label, str(path)


# ─── DataLoader helpers ──────────────────────────────────────────────────

def get_train_loader(cfg: ModelConfig) -> DataLoader:
    """Training dataloader — only good/normal images."""
    tfm = get_transform(cfg.resize, cfg.center_crop)
    ds = OringDataset(cfg.train_good_dir, label=0, label_name="good", transform=tfm)
    print(f"  Train set ({cfg.name}): {len(ds)} good images from {cfg.train_good_dir}")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,         # order doesn't matter for feature extraction
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def get_test_loaders(cfg: ModelConfig) -> List[Tuple[str, DataLoader]]:
    """Test dataloaders — one per test directory (good + defect categories).

    Returns a list of (label_name, DataLoader) tuples.
    """
    tfm = get_transform(cfg.resize, cfg.center_crop)
    loaders: List[Tuple[str, DataLoader]] = []
    for label_name, dir_path in sorted(cfg.test_dirs.items()):
        is_good = "good" in label_name.lower()
        ds = OringDataset(
            dir_path,
            label=0 if is_good else 1,
            label_name=label_name,
            transform=tfm,
        )
        print(f"  Test set  ({cfg.name}/{label_name}): {len(ds)} images")
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        loaders.append((label_name, loader))
    return loaders
