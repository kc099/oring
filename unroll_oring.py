"""
O-Ring Unrolling — Convert circular O-ring images to linear strips.

Reads full-resolution (2448×2048) or binned O-ring images, detects the
annular ring via background subtraction, fits inner/outer circles, and
uses OpenCV polar warp to produce a rectangular strip:

    width  = circumference at mean radius  (≈ 2π × mean_r)
    height = ring wall thickness + padding

The strip is saved as a PNG alongside a JSON metadata file that records
the transformation parameters so coordinates can be mapped back.

Folders processed (from Original Data/):
    good, notok, notok2, model1good, model1defect, model1defect2

Usage:
    python unroll_oring.py                       # process all
    python unroll_oring.py --folders good notok   # selective

Author: GitHub Copilot
Date: February 17, 2026
"""

import cv2
import numpy as np
import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_ROOT = Path(r"F:\standard elastomers\Original Data")
OUTPUT_ROOT = Path(r"F:\standard elastomers\oring_linear_patches")

FOLDERS = ["good", "notok", "notok2", "model1good", "model1defect", "model1defect2"]

# Extra pixels above/below the ring wall to include in the strip
RADIAL_PADDING = 20


# ═══════════════════════════════════════════════════════════════════════════
#  O-Ring detection helpers (adapted from rework/inspection_gui.py)
# ═══════════════════════════════════════════════════════════════════════════

def _largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


def auto_bg_value(image: np.ndarray, margin: int = 80) -> int:
    """Estimate background intensity from the four image corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = np.concatenate([
        gray[:margin, :margin].ravel(),
        gray[:margin, w - margin:].ravel(),
        gray[h - margin:, :margin].ravel(),
        gray[h - margin:, w - margin:].ravel(),
    ])
    return int(np.median(corners))


def build_mask(image: np.ndarray, bg_value: int = 20, threshold: int = 30) -> np.ndarray:
    """Background subtraction → binary mask of the o-ring."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, np.full_like(gray, bg_value))
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    if np.mean(binary == 255) > 0.75:
        binary = cv2.bitwise_not(binary)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    return _largest_component(binary)


def find_contours(mask: np.ndarray):
    """Return (outer, inner) contours of the ring, or (None, None)."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, None

    areas = [cv2.contourArea(c) for c in contours]
    outer_idx = int(np.argmax(areas))
    outer = contours[outer_idx]

    inner = None
    for i, h in enumerate(hierarchy[0]):
        if h[3] == outer_idx:
            if inner is None or cv2.contourArea(contours[i]) > cv2.contourArea(inner):
                inner = contours[i]
    return outer, inner


# ═══════════════════════════════════════════════════════════════════════════
#  Unrolling
# ═══════════════════════════════════════════════════════════════════════════

def detect_ring_geometry(image: np.ndarray) -> Optional[Dict]:
    """
    Detect the O-ring and return geometry dict:
        center_x, center_y, inner_radius, outer_radius
    Returns None on failure.
    """
    bg = auto_bg_value(image)
    mask = build_mask(image, bg_value=bg, threshold=30)
    outer, inner = find_contours(mask)

    if outer is None or inner is None:
        # Fallback: try Otsu on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = _largest_component(binary)
        outer, inner = find_contours(binary)
        if outer is None or inner is None:
            return None

    (ox, oy), orad = cv2.minEnclosingCircle(outer)
    (ix, iy), irad = cv2.minEnclosingCircle(inner)

    # Use midpoint of outer and inner centers as ring center
    cx = (ox + ix) / 2.0
    cy = (oy + iy) / 2.0

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "inner_radius": float(irad),
        "outer_radius": float(orad),
        "outer_center": (float(ox), float(oy)),
        "inner_center": (float(ix), float(iy)),
    }


def unroll_oring(image: np.ndarray,
                 cx: float, cy: float,
                 r_inner: float, r_outer: float,
                 radial_padding: int = RADIAL_PADDING,
                 angular_samples: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Convert the annular O-ring region into a rectangular strip using
    polar-to-Cartesian mapping.

    Parameters
    ----------
    image        : BGR input image
    cx, cy       : ring center
    r_inner      : inner radius in pixels
    r_outer      : outer radius in pixels
    radial_padding : extra pixels to include above/below ring walls
    angular_samples: width of output strip (0 = auto ≈ 2π × mean_r)

    Returns
    -------
    strip        : the unrolled strip (height = wall + 2*padding, width = angular_samples)
    meta         : dict with transformation parameters
    """
    r_min = max(0, r_inner - radial_padding)
    r_max = r_outer + radial_padding

    # Auto-compute angular samples if not specified
    mean_r = (r_inner + r_outer) / 2.0
    if angular_samples <= 0:
        angular_samples = int(round(2 * math.pi * mean_r))

    strip_height = int(round(r_max - r_min))
    strip_width = angular_samples

    # Build inverse map: for every (x, y) in strip, compute source (sx, sy)
    # x-axis → angle (0..2π), y-axis → radius (r_min..r_max)
    angles = np.linspace(0, 2 * math.pi, strip_width, endpoint=False).astype(np.float32)
    radii = np.linspace(r_min, r_max, strip_height, endpoint=True).astype(np.float32)

    # Create meshgrid: angle_map[y,x] = angle, radius_map[y,x] = radius
    angle_map, radius_map = np.meshgrid(angles, radii)

    # Convert polar → Cartesian  (source pixel locations)
    map_x = (cx + radius_map * np.cos(angle_map)).astype(np.float32)
    map_y = (cy + radius_map * np.sin(angle_map)).astype(np.float32)

    # Remap
    strip = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    meta = {
        "center_x": float(cx),
        "center_y": float(cy),
        "inner_radius": float(r_inner),
        "outer_radius": float(r_outer),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "radial_padding": radial_padding,
        "strip_width": strip_width,
        "strip_height": strip_height,
        "angular_samples": angular_samples,
        "angle_start": 0.0,
        "angle_end": 2 * math.pi,
    }
    return strip, meta


# ═══════════════════════════════════════════════════════════════════════════
#  Batch processing
# ═══════════════════════════════════════════════════════════════════════════

def process_image(image_path: Path, output_folder: Path,
                  radial_padding: int = RADIAL_PADDING) -> Optional[Dict]:
    """Process a single image: detect → unroll → save strip + metadata JSON."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Failed to load: {image_path.name}")
        return None

    geom = detect_ring_geometry(img)
    if geom is None:
        print(f"  ✗ Could not detect O-ring: {image_path.name}")
        return None

    strip, meta = unroll_oring(
        img,
        geom["center_x"], geom["center_y"],
        geom["inner_radius"], geom["outer_radius"],
        radial_padding=radial_padding,
    )

    # Save strip as PNG
    stem = image_path.stem
    strip_path = output_folder / f"{stem}.png"
    cv2.imwrite(str(strip_path), strip)

    # Save metadata JSON
    meta["source_filename"] = image_path.name
    meta["source_folder"] = image_path.parent.name
    meta["source_width"] = img.shape[1]
    meta["source_height"] = img.shape[0]
    meta["strip_file"] = strip_path.name

    json_path = output_folder / f"{stem}_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def process_folder(folder_name: str,
                   source_root: Path = SOURCE_ROOT,
                   output_root: Path = OUTPUT_ROOT,
                   radial_padding: int = RADIAL_PADDING):
    """Process all images in one Original Data subfolder."""
    src = source_root / folder_name
    dst = output_root / folder_name
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"\n⚠ Folder not found: {src}")
        return

    image_files = sorted(
        list(src.glob("*.bmp")) + list(src.glob("*.BMP")) +
        list(src.glob("*.png")) + list(src.glob("*.PNG")) +
        list(src.glob("*.jpg")) + list(src.glob("*.JPG"))
    )

    print(f"\n{'='*70}")
    print(f" {folder_name}:  {len(image_files)} images  →  {dst}")
    print(f"{'='*70}")

    success = 0
    fail = 0
    widths, heights = [], []

    for i, img_path in enumerate(image_files):
        meta = process_image(img_path, dst, radial_padding=radial_padding)
        if meta:
            success += 1
            widths.append(meta["strip_width"])
            heights.append(meta["strip_height"])
            if success <= 3 or (success % 50 == 0):
                print(f"  [{i+1}/{len(image_files)}] {img_path.name}  →  "
                      f"{meta['strip_width']}×{meta['strip_height']}")
        else:
            fail += 1

    print(f"\n  Summary: {success} OK, {fail} failed")
    if widths:
        print(f"  Strip sizes — "
              f"W: {min(widths)}–{max(widths)} (avg {int(np.mean(widths))}),  "
              f"H: {min(heights)}–{max(heights)} (avg {int(np.mean(heights))})")


def main():
    parser = argparse.ArgumentParser(description="Unroll O-ring images to linear strips")
    parser.add_argument("--folders", nargs="*", default=None,
                        help="Subset of folders to process (default: all)")
    parser.add_argument("--source", type=str, default=str(SOURCE_ROOT),
                        help="Source root with image subfolders")
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT),
                        help="Output root for linear patches")
    parser.add_argument("--padding", type=int, default=RADIAL_PADDING,
                        help="Radial padding above/below ring walls")
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)
    radial_padding = args.padding

    folders = args.folders if args.folders else FOLDERS

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       O-Ring Unrolling — Circular → Linear Strip       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Source : {source_root}")
    print(f"  Output : {output_root}")
    print(f"  Padding: {radial_padding} px")
    print(f"  Folders: {', '.join(folders)}")

    for folder in folders:
        process_folder(folder, source_root, output_root, radial_padding)

    print(f"\n{'='*70}")
    print("Done!  Linear patches saved to:", output_root)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
