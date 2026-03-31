"""
Train PatchCore on YOLO-cropped O-ring images.

Trains two separate PatchCore models:
  - Model 1: from data/patchcore-model1-crops/
  - Model 2: from data/patchcore-model2-crops/

Since crops vary in size, images are resized (bicubic) and center-cropped
to a uniform size before feature extraction — same as existing PatchCore pipeline.

All cropped images are treated as "good" training data (PatchCore is
unsupervised — learns normal appearance only).

Usage:
    python train_patchcore.py                    # train both models
    python train_patchcore.py --model model1     # train one model
    python train_patchcore.py --backbone resnet101 --coreset 0.10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Import from the patchcore package (parent directory)
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from patchcore.config import ModelConfig, RESIZE_SIZE, CENTER_CROP_SIZE
from patchcore.dataset import get_train_loader
from patchcore.patchcore_model import PatchCore

DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results_cropped"

MODEL_CONFIGS = {
    "model1": {
        "train_dir": RESULTS_DIR / "patchcore-model1-crops",
        "name_prefix": "model1_cropped",
    },
    "model2": {
        "train_dir": RESULTS_DIR / "patchcore-model2-crops",
        "name_prefix": "model2_cropped",
    },
}


def train_single(
    model_key: str,
    backbone: str = "resnet50",
    coreset_ratio: float = 0.25,
    batch_size: int = 8,
    num_workers: int = 4,
    resize: int = RESIZE_SIZE,
    center_crop: int = CENTER_CROP_SIZE,
) -> dict:
    """Train a single PatchCore model on cropped images."""
    info = MODEL_CONFIGS[model_key]
    train_dir = info["train_dir"]
    name = f"{info['name_prefix']}_{backbone}"

    if not train_dir.exists():
        print(f"ERROR: Training directory not found: {train_dir}")
        print("       Run crop_with_yolo.py first.")
        return {}

    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        name=name,
        backbone=backbone,
        coreset_ratio=coreset_ratio,
        train_good_dir=train_dir,
        test_dirs={},  # no test dirs — train only
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        resize=resize,
        center_crop=center_crop,
    )

    print(f"\n{'='*70}")
    print(f"  Training PatchCore: {name}")
    print(f"  Train dir : {train_dir}")
    print(f"  Backbone  : {backbone}")
    print(f"  Coreset   : {coreset_ratio:.0%}")
    print(f"  Resize    : direct to {resize}x{resize}")
    print(f"{'='*70}")

    t0 = time.time()

    model = PatchCore(cfg)
    train_loader = get_train_loader(cfg)
    model.fit(train_loader)
    save_path = model.save()

    elapsed = time.time() - t0

    result = {
        "name": name,
        "backbone": backbone,
        "coreset_ratio": coreset_ratio,
        "memory_bank_shape": list(model.memory_bank.shape) if model.memory_bank is not None else [],
        "num_train_images": len(train_loader.dataset),
        "training_time_s": round(elapsed, 1),
        "model_path": str(save_path),
    }

    results_path = output_dir / f"{name}_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Model saved: {save_path}")
    print(f"  Training time: {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore on YOLO-cropped O-ring images"
    )
    parser.add_argument("--model", choices=["model1", "model2", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"],
                        default="resnet50", help="Backbone architecture")
    parser.add_argument("--coreset", type=float, default=0.03,
                        help="Coreset sampling ratio")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=RESIZE_SIZE,
                        help="Resize before center crop")
    parser.add_argument("--center-crop", type=int, default=CENTER_CROP_SIZE,
                        help="Center crop size")
    args = parser.parse_args()

    models_to_train = []
    if args.model in ("model1", "all"):
        models_to_train.append("model1")
    if args.model in ("model2", "all"):
        models_to_train.append("model2")

    all_results = []
    for model_key in models_to_train:
        result = train_single(
            model_key,
            backbone=args.backbone,
            coreset_ratio=args.coreset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            resize=args.resize,
            center_crop=args.center_crop,
        )
        if result:
            all_results.append(result)

    print("\n" + "=" * 70)
    print("PATCHCORE TRAINING COMPLETE")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['name']}:")
        print(f"    Bank: {r['memory_bank_shape']}  |  "
              f"Images: {r['num_train_images']}  |  "
              f"Time: {r['training_time_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
