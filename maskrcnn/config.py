"""
Configuration for Mask R-CNN O-Ring Defect Detection.

Three model configurations:
  - Model 1 O-Ring: model1good (good) + model1defect (defect)
  - Model 2 O-Ring: good (good) + notok (defect)
  - Combined:       both model1 + model2 data merged together

Author: GitHub Copilot
Date: February 9, 2026
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"F:\standard elastomers")
BINNED_ROOT = PROJECT_ROOT / "binned"           # binned 720×720 images
MASKS_ROOT = BINNED_ROOT / "masks"              # labelled mask JSONs
ORIGINAL_ROOT = PROJECT_ROOT / "Original Data"
MASKRCNN_ROOT = PROJECT_ROOT / "maskrcnn"
OUTPUT_ROOT = MASKRCNN_ROOT / "dataset"


# ─── O-Ring model definitions ───────────────────────────────────────────────
@dataclass
class OringModelConfig:
    """Configuration for one o-ring type."""
    name: str                          # e.g. "model1" or "model2"
    good_folders: List[str]            # folders with good samples
    defect_folders: List[str]          # folders with defect samples
    mask_folders: Dict[str, str]       # original mask folder names (for full-res)
    description: str = ""

    @property
    def all_folders(self) -> List[str]:
        return self.good_folders + self.defect_folders


MODEL1_CONFIG = OringModelConfig(
    name="model1",
    good_folders=["model1good"],
    defect_folders=["model1defect", "model1defect2"],
    mask_folders={"model1defect": "model1defect", "model1defect2": "model1defect2"},
    description="Model 1 O-Ring (model1good + model1defect + model1defect2)"
)

MODEL2_CONFIG = OringModelConfig(
    name="model2",
    good_folders=["good"],
    defect_folders=["notok", "notok2"],
    mask_folders={"notok": "notok", "notok2": "notok2"},
    description="Model 2 O-Ring (good + notok + notok2)"
)

COMBINED_CONFIG = OringModelConfig(
    name="combined",
    good_folders=["model1good", "good"],
    defect_folders=["model1defect", "model1defect2", "notok", "notok2"],
    mask_folders={
        "model1defect": "model1defect",
        "model1defect2": "model1defect2",
        "notok": "notok",
        "notok2": "notok2",
    },
    description="Combined (model1 + model2 data)"
)

ORING_MODELS = {
    "model1": MODEL1_CONFIG,
    "model2": MODEL2_CONFIG,
    "combined": COMBINED_CONFIG,
}


# ─── Dataset / Training parameters ──────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Hyperparameters for Mask R-CNN training."""
    # Data
    image_size: int = 720               # input size (fits binned+cropped o-ring images)
    num_classes: int = 2                 # background + defect
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Training
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 50
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 15
    lr_gamma: float = 0.1

    # Backbone
    backbone: str = "resnet50"          # resnet50 or resnet101
    pretrained: bool = True
    trainable_backbone_layers: int = 3  # fine-tune last N layers

    # Augmentation
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_degrees: int = 15
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2

    # Inference
    score_threshold: float = 0.5        # min confidence for predictions
    nms_threshold: float = 0.3          # NMS IoU threshold
    mask_threshold: float = 0.5         # binarize predicted masks

    # Early stopping
    patience: int = 10

    # Misc
    seed: int = 42
    device: str = "cuda"                # "cuda" or "cpu"
    save_best_only: bool = True
    mixed_precision: bool = True         # use AMP for faster training


TRAINING_CONFIG = TrainingConfig()
