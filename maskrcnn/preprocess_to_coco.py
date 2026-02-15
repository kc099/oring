"""
Preprocessing: Convert binned 720×720 dataset to COCO format + train/val/test split.

Reads 720×720 binned images from binned/<folder>/ and labels (mask JSONs)
from binned/masks/<folder>/. Defect folders must have non-empty mask JSONs;
good folders have no labels (empty annotations).

This script:
1) Reads 720×720 images from binned/<folder>/ for each folder in the model config
2) For defect folders: looks for _mask.json labels in binned/masks/<folder>/
3) For good folders: includes all images with no annotations
4) Balances: keeps ALL defect images, samples good images = same count as defect
5) Splits into train/val/test (80/10/10) per o-ring model
6) Generates COCO JSON annotations for each split
7) Copies images into organized folders

Usage:
    python preprocess_to_coco.py                      # process all models
    python preprocess_to_coco.py --model model1        # only model1
    python preprocess_to_coco.py --model model2        # only model2
    python preprocess_to_coco.py --model combined      # model1 + model2 merged

Author: GitHub Copilot
Date: February 14, 2026
"""

import json
import shutil
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from config import (
    BINNED_ROOT, MASKS_ROOT, MASKRCNN_ROOT, OUTPUT_ROOT,
    ORING_MODELS, TRAINING_CONFIG, OringModelConfig
)
from utils import (
    load_json_label, polygons_from_label,
    polygon_to_coco_segmentation, polygon_to_bbox, polygon_area
)


def collect_samples(
    model_cfg: OringModelConfig,
    binned_root: Path,
    masks_root: Path
) -> Tuple[List[dict], List[dict]]:
    """
    Collect defect and good samples for one o-ring model.

    Images: binned_root/<folder>/*.bmp|*.png
    Labels: masks_root/<folder>/<stem>_mask.json

    Returns:
        defect_samples: list of {"image_path": Path, "label_path": Path, "folder": str}
        good_samples: list of {"image_path": Path, "folder": str}
    """
    defect_samples = []
    good_samples = []

    # --- Defect folders ---
    for folder in model_cfg.defect_folders:
        img_dir = binned_root / folder
        lbl_dir = masks_root / folder
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping")
            continue

        for img_path in sorted(
                p for ext in ("*.bmp", "*.png") for p in img_dir.glob(ext)):
            # Labels use _mask.json suffix
            lbl_path = lbl_dir / (img_path.stem + "_mask.json")
            if not lbl_path.exists():
                # Also try without _mask suffix as fallback
                lbl_path = lbl_dir / (img_path.stem + ".json")
                if not lbl_path.exists():
                    continue

            # Only include if there are actual polygons
            label = load_json_label(lbl_path)
            polygons = polygons_from_label(label)
            if len(polygons) > 0:
                defect_samples.append({
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "folder": folder
                })

    # --- Good folders (no labels needed) ---
    for folder in model_cfg.good_folders:
        img_dir = binned_root / folder
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping")
            continue

        for img_path in sorted(
                p for ext in ("*.bmp", "*.png") for p in img_dir.glob(ext)):
            good_samples.append({
                "image_path": img_path,
                "folder": folder
            })

    return defect_samples, good_samples


def split_samples(
    samples: list,
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[list, list, list]:
    """Shuffle and split samples into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def build_coco_annotations(
    defect_samples: list,
    good_samples: list,
    split_name: str,
    output_dir: Path,
    category_name: str = "defect"
) -> dict:
    """
    Build COCO annotation JSON and copy images for one split.

    Good samples are included as images with no annotations (important
    for training — the model needs to learn that some images have no defects).
    """
    img_dir = output_dir / "images" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": category_name, "supercategory": "defect"}
        ]
    }

    ann_id = 1
    img_id = 1

    # --- Defect images ---
    for sample in defect_samples:
        img_path = sample["image_path"]
        lbl_path = sample["label_path"]

        # Copy image
        dst = img_dir / img_path.name
        if not dst.exists():
            shutil.copy2(str(img_path), str(dst))

        # Read dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        # Add annotations
        label = load_json_label(lbl_path)
        polygons = polygons_from_label(label)
        for poly in polygons:
            seg = polygon_to_coco_segmentation(poly)
            bbox = polygon_to_bbox(poly)
            area = polygon_area(poly)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [seg],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    # --- Good images (no annotations) ---
    for sample in good_samples:
        img_path = sample["image_path"]

        dst = img_dir / img_path.name
        if not dst.exists():
            shutil.copy2(str(img_path), str(dst))

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })
        img_id += 1

    return coco


def preprocess_model(model_cfg: OringModelConfig, cfg=TRAINING_CONFIG):
    """Full preprocessing pipeline for one o-ring model."""
    print(f"\n{'='*80}")
    print(f"PREPROCESSING: {model_cfg.description}")
    print(f"{'='*80}")

    # 1. Collect samples
    defect_samples, good_samples = collect_samples(model_cfg, BINNED_ROOT, MASKS_ROOT)
    print(f"\n  Defect samples: {len(defect_samples)}")
    print(f"  Good samples:   {len(good_samples)}")

    if len(defect_samples) == 0:
        print("  ERROR: No defect samples found. Check dataset paths.")
        return

    # 2. Balance: all defect images, good images = same count as defect
    n_defect = len(defect_samples)
    n_good_target = n_defect
    rng = random.Random(cfg.seed)

    if len(good_samples) > n_good_target:
        good_shuffled = list(good_samples)
        rng.shuffle(good_shuffled)
        good_samples = good_shuffled[:n_good_target]
        print(f"  Balanced good samples: {len(good_samples)} (target = defect count = {n_good_target})")
    else:
        print(f"  Good samples ({len(good_samples)}) already <= target ({n_good_target}), using all")

    # 3. Split defect and good samples independently (stratified)
    d_train, d_val, d_test = split_samples(
        defect_samples, cfg.train_ratio, cfg.val_ratio, cfg.seed
    )
    g_train, g_val, g_test = split_samples(
        good_samples, cfg.train_ratio, cfg.val_ratio, cfg.seed
    )

    print(f"\n  Split (defect): train={len(d_train)}, val={len(d_val)}, test={len(d_test)}")
    print(f"  Split (good):   train={len(g_train)}, val={len(g_val)}, test={len(g_test)}")

    # 3. Build COCO annotations for each split
    model_output = OUTPUT_ROOT / model_cfg.name
    model_output.mkdir(parents=True, exist_ok=True)

    for split_name, d_split, g_split in [
        ("train", d_train, g_train),
        ("val", d_val, g_val),
        ("test", d_test, g_test)
    ]:
        print(f"\n  Building {split_name} split...")
        coco = build_coco_annotations(d_split, g_split, split_name, model_output)

        # Save annotation JSON
        ann_dir = model_output / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / f"{split_name}.json"
        with open(ann_path, 'w') as f:
            json.dump(coco, f, indent=2)

        n_imgs = len(coco["images"])
        n_anns = len(coco["annotations"])
        print(f"    Images: {n_imgs}, Annotations: {n_anns}")
        print(f"    Saved: {ann_path}")

    # 4. Save a summary
    summary = {
        "model": model_cfg.name,
        "description": model_cfg.description,
        "defect_count": len(defect_samples),
        "good_count_used": len(good_samples),
        "good_target": n_good_target,
        "splits": {
            "train": {"defect": len(d_train), "good": len(g_train)},
            "val": {"defect": len(d_val), "good": len(g_val)},
            "test": {"defect": len(d_test), "good": len(g_test)},
        }
    }
    with open(model_output / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done! Output: {model_output}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess o-ring dataset to COCO format")
    parser.add_argument("--model", type=str, default=None,
                        choices=["model1", "model2", "combined"],
                        help="Process only this o-ring model (default: all)")
    args = parser.parse_args()

    if args.model:
        preprocess_model(ORING_MODELS[args.model])
    else:
        for model_cfg in ORING_MODELS.values():
            preprocess_model(model_cfg)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
