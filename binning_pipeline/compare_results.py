"""
Quick visual comparison of binning + cropping results.

Generates a side-by-side grid: original (scaled) | binned | binned+cropped
for a random sample of images from each folder.

Usage:
    python binning_pipeline/compare_results.py
    python binning_pipeline/compare_results.py --n 8  --folder model1defect2

Author: GitHub Copilot
Date: February 14, 2026
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent


def tile_comparison(original: np.ndarray,
                    binned: np.ndarray,
                    cropped: np.ndarray,
                    label: str,
                    tile_h: int = 350) -> np.ndarray:
    """Create a side-by-side comparison tile for one image."""

    def resize_to_height(img, h):
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    panels = []
    titles = [
        f"Original ({original.shape[1]}x{original.shape[0]})",
        f"Binned ({binned.shape[1]}x{binned.shape[0]})",
        f"Cropped ({cropped.shape[1]}x{cropped.shape[0]})",
    ]
    for img, title in zip([original, binned, cropped], titles):
        vis = resize_to_height(img, tile_h)
        # Add title bar
        bar = np.zeros((30, vis.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        vis = np.vstack([bar, vis])
        panels.append(vis)

    # Pad all to same height
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    # Add separator lines
    sep = np.full((max_h, 3, 3), 80, dtype=np.uint8)
    row = np.hstack([padded[0], sep, padded[1], sep, padded[2]])

    # Label
    label_bar = np.zeros((25, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_bar, label, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return np.vstack([label_bar, row])


def main():
    parser = argparse.ArgumentParser(description="Visual comparison of binning results")
    parser.add_argument("--folder", type=str, default=None,
                        help="Specific folder name (e.g. model1defect2)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of sample images per folder")
    args = parser.parse_args()

    os.chdir(WORKSPACE)

    from bin_and_crop import binning_2x2, crop_to_foreground

    pairs = [
        ("Original Data/model1defect2", "model1defect2"),
        ("Original Data/notok2", "notok2"),
    ]
    if args.folder:
        pairs = [(f"Original Data/{args.folder}", args.folder)]

    for input_dir, name in pairs:
        inp = Path(input_dir)
        if not inp.exists():
            print(f"Skipping {input_dir} (not found)")
            continue

        files = sorted([f for f in inp.iterdir() if f.suffix.lower() in {".bmp", ".png", ".jpg"}])
        sample = random.sample(files, min(args.n, len(files)))

        tiles = []
        for fpath in sample:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            binned = binning_2x2(img)
            cropped, _ = crop_to_foreground(binned)
            tile = tile_comparison(img, binned, cropped, fpath.name)
            tiles.append(tile)

        if not tiles:
            continue

        # Pad all tiles to same width
        max_w = max(t.shape[1] for t in tiles)
        padded = []
        for t in tiles:
            if t.shape[1] < max_w:
                pad = np.zeros((t.shape[0], max_w - t.shape[1], 3), dtype=np.uint8)
                t = np.hstack([t, pad])
            padded.append(t)

        grid = np.vstack(padded)

        out_path = f"binned/{name}_comparison.png"
        os.makedirs("binned", exist_ok=True)
        cv2.imwrite(out_path, grid)
        print(f"Saved comparison: {out_path}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
