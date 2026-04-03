"""
PatchCore Configuration
=======================
Central configuration for training PatchCore anomaly detection models
on binned 720x720 O-ring images (Model 1 and Model 2).

Author: GitHub Copilot
Date:   February 27, 2026
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── Workspace root ──────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent

# ─── Image settings ──────────────────────────────────────────────────────
ORIGINAL_CROP_SIZE = 720          # binned/cropped image size
RESIZE_SIZE = 384                 # bicubic resize all crops to this square
CENTER_CROP_SIZE = 384            # same as resize (no center crop)
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ─── ImageNet normalization ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@dataclass
class ModelConfig:
    """Configuration for a single PatchCore model (one O-ring model)."""

    name: str                          # human-readable name
    backbone: str = "resnet50"         # "resnet50" or "resnet101"

    # Feature extraction layers (ResNet block indices, 0-based)
    # Layer 2 → 512-d (28×28 @224 input), Layer 3 → 1024-d (14×14)
    feature_layers: Tuple[int, ...] = (2, 3)

    # Coreset subsampling ratio  (0.25 = keep 25% of patch features)
    coreset_ratio: float = 0.25

    # k-NN scoring
    n_neighbors: int = 9              # number of neighbors for anomaly score

    # Image resize / crop
    resize: int = RESIZE_SIZE
    center_crop: int = CENTER_CROP_SIZE

    # ── Paths (filled per-model) ──
    train_good_dir: Path = Path(".")
    test_dirs: dict = field(default_factory=dict)   # label → Path
    output_dir: Path = Path(".")

    # ── Training ──
    batch_size: int = 8
    num_workers: int = 4


def get_model1_config(backbone: str = "resnet50") -> ModelConfig:
    """Return config for Model 1 O-ring."""
    return ModelConfig(
        name=f"model1_{backbone}",
        backbone=backbone,
        train_good_dir=WORKSPACE / "binned" / "model1good",
        test_dirs={
            "good":    WORKSPACE / "binned" / "model1good",
            "defect":  WORKSPACE / "binned" / "model1defect",
            "defect2": WORKSPACE / "binned" / "model1defect2",
            "defect3": WORKSPACE / "binned" / "model1defect3",
        },
        output_dir=WORKSPACE / "patchcore" / "results" / f"model1_{backbone}",
    )


def get_model2_config(backbone: str = "resnet50") -> ModelConfig:
    """Return config for Model 2 O-ring."""
    return ModelConfig(
        name=f"model2_{backbone}",
        backbone=backbone,
        train_good_dir=WORKSPACE / "binned" / "good",
        test_dirs={
            "good":   WORKSPACE / "binned" / "good",
            "notok":  WORKSPACE / "binned" / "notok",
            "notok2": WORKSPACE / "binned" / "notok2",
            "notok3": WORKSPACE / "binned" / "notok3",
        },
        output_dir=WORKSPACE / "patchcore" / "results" / f"model2_{backbone}",
    )


def get_all_configs() -> List[ModelConfig]:
    """Return configs for all 4 combinations (2 models × 2 backbones)."""
    return [
        get_model1_config("resnet50"),
        get_model1_config("resnet101"),
        get_model2_config("resnet50"),
        get_model2_config("resnet101"),
    ]
