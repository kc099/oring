"""
Camera-Binning Simulation + Background Crop Pipeline

Simulates 2×2 binning (averaging) to halve resolution from 2448×2048 → 1224×1024,
then crops away dark background pixels and resizes/pads to a fixed 720×720 output
so no further padding is needed during training.

Steps:
    1. 2×2 pixel binning (average pooling) — halves both dimensions
    2. Background subtraction → binary mask of foreground
    3. Morphological cleanup of the mask
    4. Bounding-box crop with configurable padding
    5. Resize to fit within 720×720 (maintaining aspect ratio)
    6. Pad with background color to exactly 720×720

Input:  Original 2448×2048 BMP images
Output: 720×720 binned + cropped images (BMP by default)

Usage:
    python binning_pipeline/bin_and_crop.py --input "Original Data/model1defect2" --output binned/model1defect2
    python binning_pipeline/bin_and_crop.py --input "Original Data/notok2" --output binned/notok2
    python binning_pipeline/bin_and_crop.py --all   # process all default folders

Author: GitHub Copilot
Date: February 14, 2026
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


# ─── Defaults ─────────────────────────────────────────────────────────────
BG_VALUE = 18           # background grey level
BG_THRESHOLD = 30      # pixels with |gray - bg| > this are foreground
PAD_PX = 20            # padding around bounding box (in binned pixels)
MORPH_KERNEL = 15      # morphological kernel size for mask cleanup
TARGET_SIZE = 720      # fixed output size (720×720)

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = [
    ("Original Data/model1defect2", "binned/model1defect2"),
    ("Original Data/notok2",        "binned/notok2"),
    ("Original Data/notok",        "binned/notok"),
    ("Original Data/model1good",   "binned/model1good"),
    ("Original Data/good",         "binned/good"),
    ("Original Data/model1defect", "binned/model1defect"),
]


# ─── Core functions ──────────────────────────────────────────────────────

def binning_2x2(image: np.ndarray) -> np.ndarray:
    """Simulate 2×2 camera binning via average pooling.

    Each 2×2 block of pixels is replaced by their average.
    This halves both width and height.
    """
    h, w = image.shape[:2]
    # Ensure dimensions are even
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    cropped = image[:h_even, :w_even]

    if len(cropped.shape) == 3:
        # Color image: reshape into 2×2 blocks per channel
        reshaped = cropped.reshape(h_even // 2, 2, w_even // 2, 2, -1)
        binned = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    else:
        # Grayscale
        reshaped = cropped.reshape(h_even // 2, 2, w_even // 2, 2)
        binned = reshaped.mean(axis=(1, 3)).astype(np.uint8)

    return binned


def foreground_bbox(gray: np.ndarray,
                    bg_value: int = BG_VALUE,
                    threshold: int = BG_THRESHOLD,
                    morph_k: int = MORPH_KERNEL) -> Optional[Tuple[int, int, int, int]]:
    """Find bounding box (x, y, w, h) of the o-ring foreground."""
    diff = cv2.absdiff(gray, np.full_like(gray, bg_value))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Clean up noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    return cv2.boundingRect(coords)


def crop_to_foreground(image: np.ndarray,
                       bg_value: int = BG_VALUE,
                       threshold: int = BG_THRESHOLD,
                       pad: int = PAD_PX,
                       target_size: int = TARGET_SIZE) -> Tuple[np.ndarray, dict]:
    """Crop image to foreground bounding box, then resize + pad to target_size×target_size.

    The image is first cropped to the o-ring bounding box (with padding),
    then resized to fit within target_size×target_size while maintaining
    aspect ratio, and finally padded with bg_value to exactly target_size×target_size.

    Returns (output_image, info_dict).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    bbox = foreground_bbox(gray, bg_value, threshold)
    if bbox is None:
        # Fallback: use entire image
        bbox = (0, 0, w, h)

    bx, by, bw, bh = bbox

    # Apply padding
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(w, bx + bw + pad)
    y2 = min(h, by + bh + pad)

    cropped = image[y1:y2, x1:x2]
    ch, cw = cropped.shape[:2]

    # Resize to fit within target_size×target_size, maintaining aspect ratio
    scale = min(target_size / cw, target_size / ch)
    if scale < 1.0:
        # Only downscale if needed (never upscale to avoid artifacts)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cropped
        new_w, new_h = cw, ch

    # Pad to exactly target_size×target_size with background color
    if len(image.shape) == 3:
        canvas = np.full((target_size, target_size, image.shape[2]),
                         bg_value, dtype=np.uint8)
    else:
        canvas = np.full((target_size, target_size), bg_value, dtype=np.uint8)

    # Center the resized image on the canvas
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    info = {
        "bbox": (bx, by, bw, bh),
        "crop_region": (x1, y1, x2, y2),
        "crop_size": (cw, ch),
        "scale": round(scale, 4),
        "resized_size": (new_w, new_h),
        "offset": (offset_x, offset_y),
        "final_size": (target_size, target_size),
    }
    return canvas, info


def process_image(input_path: str,
                  output_path: str,
                  bg_value: int = BG_VALUE,
                  threshold: int = BG_THRESHOLD,
                  pad: int = PAD_PX,
                  target_size: int = TARGET_SIZE,
                  save_mask: bool = False) -> Optional[dict]:
    """Full pipeline: load → bin → crop → resize/pad to 720×720 → save."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"  ✗ Cannot read: {input_path}")
        return None

    orig_h, orig_w = img.shape[:2]

    # Step 1: 2×2 binning
    binned = binning_2x2(img)
    bin_h, bin_w = binned.shape[:2]

    # Step 2: crop to foreground + resize/pad to target_size×target_size
    result, crop_info = crop_to_foreground(
        binned, bg_value, threshold, pad, target_size)
    final_h, final_w = result.shape[:2]

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)

    # Optionally save the binary mask for inspection
    if save_mask:
        mask_dir = os.path.join(os.path.dirname(output_path), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        gray_bin = cv2.cvtColor(binned, cv2.COLOR_BGR2GRAY)
        bbox = foreground_bbox(gray_bin, bg_value, threshold)
        if bbox is not None:
            diff = cv2.absdiff(gray_bin, np.full_like(gray_bin, bg_value))
            _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            mask_path = os.path.join(mask_dir, Path(output_path).stem + "_mask.png")
            cv2.imwrite(mask_path, mask)

    return {
        "filename": os.path.basename(input_path),
        "original_size": f"{orig_w}x{orig_h}",
        "binned_size": f"{bin_w}x{bin_h}",
        "final_size": f"{final_w}x{final_h}",
        "crop_size": f"{crop_info['crop_size'][0]}x{crop_info['crop_size'][1]}",
        "scale": crop_info["scale"],
        "offset": crop_info["offset"],
        "bbox": crop_info["bbox"],
    }


def process_folder(input_folder: str,
                   output_folder: str,
                   bg_value: int = BG_VALUE,
                   threshold: int = BG_THRESHOLD,
                   pad: int = PAD_PX,
                   target_size: int = TARGET_SIZE,
                   save_masks: bool = False,
                   output_ext: str = ".bmp") -> list:
    """Process all images in a folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"✗ Input folder not found: {input_folder}")
        return []

    exts = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    files = sorted([f for f in input_path.iterdir()
                    if f.suffix.lower() in exts])

    if not files:
        print(f"✗ No images found in: {input_folder}")
        return []

    print(f"\n{'='*60}")
    print(f"Processing: {input_folder}")
    print(f"Output:     {output_folder}")
    print(f"Images:     {len(files)}")
    print(f"BG={bg_value}  Threshold={threshold}  Pad={pad}  "
          f"Target={target_size}x{target_size}")
    print(f"{'='*60}")

    results = []
    for i, fpath in enumerate(files):
        out_name = fpath.stem + output_ext
        out_file = str(output_path / out_name)

        info = process_image(
            str(fpath), out_file,
            bg_value, threshold, pad, target_size,
            save_mask=save_masks)

        if info:
            results.append(info)
            status = "✓"
        else:
            status = "✗"

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {status} {fpath.name} → {info['final_size'] if info else 'FAILED'}")

    # Summary
    if results:
        print(f"\n--- Summary ---")
        print(f"  Processed: {len(results)}/{len(files)}")
        print(f"  All outputs: {target_size}x{target_size}")
        scales = [r["scale"] for r in results]
        print(f"  Scale range: {min(scales):.3f} – {max(scales):.3f}  (mean {np.mean(scales):.3f})")

        # Save CSV log
        csv_path = output_path / "processing_log.csv"
        os.makedirs(output_path, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "original_size", "binned_size", "crop_size",
                "scale", "offset", "final_size", "bbox"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Log saved: {csv_path}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate 2×2 camera binning + background crop")
    parser.add_argument("--input", type=str,
                        help="Input folder of full-res images")
    parser.add_argument("--output", type=str,
                        help="Output folder for binned+cropped images")
    parser.add_argument("--all", action="store_true",
                        help="Process default folders (model1defect2, notok2)")
    parser.add_argument("--bg", type=int, default=BG_VALUE,
                        help=f"Background grey value (default: {BG_VALUE})")
    parser.add_argument("--threshold", type=int, default=BG_THRESHOLD,
                        help=f"Foreground threshold (default: {BG_THRESHOLD})")
    parser.add_argument("--pad", type=int, default=PAD_PX,
                        help=f"Padding around crop box in px (default: {PAD_PX})")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE,
                        help=f"Fixed output size (default: {TARGET_SIZE}x{TARGET_SIZE})")
    parser.add_argument("--save-masks", action="store_true",
                        help="Also save binary masks for inspection")
    parser.add_argument("--ext", type=str, default=".bmp",
                        choices=[".bmp", ".png", ".jpg"],
                        help="Output image format (default: .bmp)")

    args = parser.parse_args()

    os.chdir(WORKSPACE)

    if args.all:
        for inp, out in DEFAULT_PAIRS:
            process_folder(inp, out,
                           args.bg, args.threshold, args.pad,
                           args.target_size,
                           args.save_masks, args.ext)
    elif args.input and args.output:
        process_folder(args.input, args.output,
                       args.bg, args.threshold, args.pad,
                       args.target_size,
                       args.save_masks, args.ext)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python binning_pipeline/bin_and_crop.py --all')
        print('  python binning_pipeline/bin_and_crop.py --all --save-masks')
        print('  python binning_pipeline/bin_and_crop.py '
              '--input "Original Data/model1defect2" '
              '--output binned/model1defect2')


if __name__ == "__main__":
    main()
