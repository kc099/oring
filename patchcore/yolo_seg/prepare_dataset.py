"""
Prepare YOLO segmentation dataset from patchcore data.

Reads images (2048x1536 BMP) and JSON polygon masks from:
  - data/patchcore-model1  +  data/patchcore-model1-label
  - data/patchcore-model2  +  data/patchcore-model2-label

Pipeline:
  1. Filter images that have non-zero masks (at least one polygon).
  2. Resize images from 2048x1536 → 640x480.
  3. Convert polygon points to normalised YOLO segmentation labels.
  4. Write data.yaml with train = val (all data used for training,
     same set reused as val for early-stopping metric).

Usage:
    python prepare_dataset.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import cv2

# ── Configuration ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# Source folders (image folder, label folder)
SOURCE_PAIRS = [
    (DATA_DIR / "patchcore-model1", DATA_DIR / "patchcore-model1-label"),
    (DATA_DIR / "patchcore-model2", DATA_DIR / "patchcore-model2-label"),
]

OUTPUT_DIR = SCRIPT_DIR / "dataset"
TARGET_W, TARGET_H = 512, 384   # 4x4 binning of 2048x1536
ORIG_W, ORIG_H = 2048, 1536


def load_mask_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def has_non_empty_mask(mask_data: dict) -> bool:
    polygons = mask_data.get("polygons", [])
    for poly in polygons:
        if len(poly.get("points", [])) >= 3:
            return True
    return False


def normalize_polygon(
    points: List[Dict], img_w: int, img_h: int
) -> List[Tuple[float, float]]:
    """Normalise polygon points to [0, 1] range."""
    normalised = []
    for pt in points:
        x = max(0.0, min(1.0, pt["x"] / img_w))
        y = max(0.0, min(1.0, pt["y"] / img_h))
        normalised.append((x, y))
    return normalised


def write_yolo_label(
    label_path: Path, polygons: List[Dict], img_w: int, img_h: int
) -> None:
    """Write YOLO segmentation label file (class 0 = defect)."""
    with open(label_path, "w") as f:
        for poly in polygons:
            pts = poly.get("points", [])
            if len(pts) < 3:
                continue
            norm_pts = normalize_polygon(pts, img_w, img_h)
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_pts)
            f.write(f"0 {coords}\n")


def main() -> None:
    images_dir = OUTPUT_DIR / "images" / "train"
    labels_dir = OUTPUT_DIR / "labels" / "train"

    # Clean previous output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    skipped_no_mask = 0
    skipped_no_image = 0

    for img_folder, label_folder in SOURCE_PAIRS:
        if not label_folder.exists():
            print(f"Warning: label folder not found: {label_folder}")
            continue
        if not img_folder.exists():
            print(f"Warning: image folder not found: {img_folder}")
            continue

        for mask_file in sorted(label_folder.glob("*_mask.json")):
            mask_data = load_mask_json(mask_file)

            # Skip images without polygons
            if not has_non_empty_mask(mask_data):
                skipped_no_mask += 1
                continue

            # Find corresponding image
            img_filename = mask_data.get("image_filename", "")
            if not img_filename:
                img_stem = mask_file.stem.replace("_mask", "")
                img_filename = img_stem + ".bmp"

            img_path = img_folder / img_filename
            if not img_path.exists():
                # Try other extensions
                for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                    candidate = img_folder / (img_path.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                else:
                    skipped_no_image += 1
                    continue

            # Read and resize image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Error reading: {img_path}")
                continue
            img_resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

            # Unique output name (folder prefix to avoid collisions)
            prefix = img_folder.name.replace("-", "_")
            out_stem = f"{prefix}_{img_path.stem}"

            # Save resized image as PNG
            out_img_path = images_dir / f"{out_stem}.png"
            cv2.imwrite(str(out_img_path), img_resized)

            # Write YOLO label (normalised coords are resolution-independent)
            orig_w = mask_data.get("image_width", ORIG_W)
            orig_h = mask_data.get("image_height", ORIG_H)
            out_label_path = labels_dir / f"{out_stem}.txt"
            write_yolo_label(out_label_path, mask_data["polygons"], orig_w, orig_h)

            total_processed += 1

    # ── Create data.yaml ─────────────────────────────────────────────────
    data_yaml_path = OUTPUT_DIR / "data.yaml"
    abs_dataset = OUTPUT_DIR.resolve().as_posix()
    yaml_content = (
        f"path: {abs_dataset}\n"
        f"train: images/train\n"
        f"val: images/train\n"  # same as train — all data for training
        f"\n"
        f"names:\n"
        f"  0: defect\n"
    )
    data_yaml_path.write_text(yaml_content, encoding="utf-8")

    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Images with masks processed : {total_processed}")
    print(f"  Skipped (no polygons)        : {skipped_no_mask}")
    print(f"  Skipped (image not found)    : {skipped_no_image}")
    print(f"  Output directory             : {OUTPUT_DIR}")
    print(f"  data.yaml                    : {data_yaml_path}")
    print(f"  Image size                   : {TARGET_W}x{TARGET_H}")
    print("=" * 70)


if __name__ == "__main__":
    main()
