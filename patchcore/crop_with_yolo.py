"""
YOLO-based Crop Pipeline for PatchCore Training Data.

For each BMP image in patchcore-model1 and patchcore-model2:
  1. Resize from 2048x1536 → 640x480
  2. Run YOLO11-seg inference to detect O-ring mask
  3. Compute bounding rectangle of the predicted mask
  4. Crop that region from the resized 640x480 image
  5. Save the crop to the output folder

Creates two output folders:
  - data/patchcore-model1-crops/
  - data/patchcore-model2-crops/

Usage:
    python crop_with_yolo.py
    python crop_with_yolo.py --model path/to/best.pt
    python crop_with_yolo.py --padding 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
YOLO_SEG_DIR = SCRIPT_DIR / "yolo_seg"

RESULTS_DIR = SCRIPT_DIR / "results_cropped"

SOURCE_FOLDERS = {
    "patchcore-model1": DATA_DIR / "patchcore-model1",
    "patchcore-model2": DATA_DIR / "patchcore-model2",
}

TARGET_W, TARGET_H = 640, 480
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_best_yolo() -> str:
    """Auto-detect best.pt from yolo_seg training runs."""
    runs_dir = YOLO_SEG_DIR / "runs"
    if not runs_dir.exists():
        return ""
    best_pts = sorted(
        runs_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(best_pts[0]) if best_pts else ""


def get_mask_bbox(result, img_h: int, img_w: int, padding: int = 10):
    """Get bounding rectangle of all predicted masks combined.

    Returns (x1, y1, x2, y2) or None if no masks found.
    """
    if result.masks is None or len(result.masks) == 0:
        return None

    # Combine all mask polygons
    all_pts = []
    for mask_xy in result.masks.xy:
        if len(mask_xy) > 0:
            all_pts.append(mask_xy)

    if not all_pts:
        return None

    all_pts = np.concatenate(all_pts, axis=0)
    x1 = max(0, int(all_pts[:, 0].min()) - padding)
    y1 = max(0, int(all_pts[:, 1].min()) - padding)
    x2 = min(img_w, int(all_pts[:, 0].max()) + padding)
    y2 = min(img_h, int(all_pts[:, 1].max()) + padding)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def process_folder(
    model: YOLO,
    src_folder: Path,
    out_folder: Path,
    padding: int = 10,
    conf: float = 0.25,
):
    """Process all images in a folder: resize → YOLO → crop → save."""
    out_folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in src_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"\n  Source: {src_folder}")
    print(f"  Output: {out_folder}")
    print(f"  Images found: {len(image_files)}")

    stats = {"total": 0, "cropped": 0, "no_mask": 0, "error": 0}

    for img_path in image_files:
        stats["total"] += 1

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ERROR reading: {img_path.name}")
            stats["error"] += 1
            continue

        # Resize to YOLO input size
        img_resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        # Run YOLO inference (GPU if available)
        results = model.predict(
            source=img_resized,
            imgsz=TARGET_W,
            conf=conf,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        result = results[0]

        # Get bounding rect of mask
        bbox = get_mask_bbox(result, TARGET_H, TARGET_W, padding=padding)

        if bbox is None:
            # No mask detected — skip this image (it may be good / no o-ring visible)
            # Save full resized image for good samples
            out_path = out_folder / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), img_resized)
            stats["no_mask"] += 1
            continue

        x1, y1, x2, y2 = bbox
        crop = img_resized[y1:y2, x1:x2]

        out_path = out_folder / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), crop)
        stats["cropped"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Crop images using YOLO masks")
    parser.add_argument("--model", type=str, default="",
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--padding", type=int, default=10,
                        help="Padding around mask bounding rect (pixels)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    args = parser.parse_args()

    model_path = args.model or find_best_yolo()
    if not model_path or not Path(model_path).exists():
        print("ERROR: No YOLO model found. Train YOLO first or provide --model.")
        sys.exit(1)

    print("=" * 70)
    print("YOLO CROP PIPELINE FOR PATCHCORE")
    print("=" * 70)
    print(f"  YOLO model : {model_path}")
    print(f"  Padding    : {args.padding}px")
    print(f"  Confidence : {args.conf}")

    model = YOLO(model_path)

    all_stats = {}
    for folder_name, src_folder in SOURCE_FOLDERS.items():
        if not src_folder.exists():
            print(f"\n  WARNING: {src_folder} not found, skipping.")
            continue

        out_folder = RESULTS_DIR / f"{folder_name}-crops"
        stats = process_folder(model, src_folder, out_folder,
                               padding=args.padding, conf=args.conf)
        all_stats[folder_name] = stats

    print("\n" + "=" * 70)
    print("CROP PIPELINE COMPLETE")
    print("=" * 70)
    for name, stats in all_stats.items():
        print(f"  {name}:")
        print(f"    Total: {stats['total']}  |  Cropped: {stats['cropped']}  "
              f"|  No mask (full): {stats['no_mask']}  |  Errors: {stats['error']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
