"""
Compute statistical parameters for good Model 2 o-ring samples.

Reads all images from Original Data/good, applies background subtraction
(bg_value=20, threshold=30) to get clean masks, then measures geometric
parameters that capture the defects we want to detect:

Defects to detect:
    1. Outer radius too large  → outer_radius
    2. Inner radius too small  → inner_radius
    3. Not perfectly round     → circularity_outer, circularity_inner,
                                  outer_radial_std, inner_radial_std
    4. Cropped/cut at edges    → edge_clearance (min distance to image border)
    5. Non-uniform wall thickness → thickness_std, thickness_cv,
                                    thickness_range, thickness_ratio

Parameters measured per image:
    outer_radius          — min enclosing circle radius of outer contour
    inner_radius          — min enclosing circle radius of inner contour
    center_dist           — distance between outer & inner fitted centres
    ring_thickness        — outer_radius − inner_radius (fitted circles)
    mean_radius           — (outer_radius + inner_radius) / 2

    min_thickness         — minimum point-to-contour wall thickness
    max_thickness         — maximum point-to-contour wall thickness
    mean_thickness        — mean of sampled wall thickness measurements
    thickness_std         — standard deviation of wall thickness (uniformity)
    thickness_cv          — coefficient of variation (std / mean × 100 %)
    thickness_range       — max − min thickness
    thickness_ratio       — max / min thickness

    circularity_outer     — 4πA / P²  of outer contour (1.0 = perfect circle)
    circularity_inner     — 4πA / P²  of inner contour
    outer_radial_std      — std of radial distances from centroid to outer contour
    inner_radial_std      — std of radial distances from centroid to inner contour

    eccentricity_pct      — center_dist / mean_radius × 100
    annular_area          — pixel count of the ring mask
    edge_clearance        — min distance from any contour point to image border

Outputs:
    rework/good_measurements.csv         — per-image raw measurements
    rework/good_measurements_stats.csv   — mean, std, min, max, p5, p95 per metric

Usage:
    python rework/compute_good_model2_stats.py

Author: GitHub Copilot
Date: February 16, 2026
"""

import csv
import math
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────
BG_VALUE = 20
BG_THRESHOLD = 30
THICKNESS_SAMPLES = 360       # sample wall thickness at this many angles
MORPH_KERNEL = 7

WORKSPACE = Path(__file__).resolve().parent.parent
INPUT_DIR = WORKSPACE / "Original Data" / "good"
OUTPUT_DIR = Path(__file__).resolve().parent  # rework/

MEASUREMENTS_CSV = OUTPUT_DIR / "good_measurements.csv"
STATS_CSV = OUTPUT_DIR / "good_measurements_stats.csv"

# Metric columns (in order)
METRIC_COLS = [
    "outer_radius", "inner_radius", "center_dist",
    "ring_thickness", "mean_radius",
    "min_thickness", "max_thickness", "mean_thickness",
    "thickness_std", "thickness_cv",
    "thickness_range", "thickness_ratio",
    "circularity_outer", "circularity_inner",
    "outer_radial_std", "inner_radial_std",
    "eccentricity_pct", "annular_area", "edge_clearance",
]


# ─── Image processing ───────────────────────────────────────────────────

def build_mask(image: np.ndarray) -> np.ndarray:
    """Background subtraction → binary mask of the o-ring."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, np.full_like(gray, BG_VALUE))
    _, binary = cv2.threshold(diff, BG_THRESHOLD, 255, cv2.THRESH_BINARY)

    if np.mean(binary == 255) > 0.75:
        binary = cv2.bitwise_not(binary)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

    # Keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


def find_contours(mask: np.ndarray):
    """Return (outer, inner) contours of the ring."""
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


def radial_std(contour: np.ndarray) -> float:
    """Standard deviation of distances from centroid to contour points.

    Lower = more circular. A perfect circle has radial_std ≈ 0.
    """
    pts = contour.reshape(-1, 2).astype(float)
    cx, cy = pts.mean(axis=0)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.std(dists))


def sample_wall_thickness(outer, inner, n_samples: int = THICKNESS_SAMPLES) -> np.ndarray:
    """Sample wall thickness at n_samples evenly-spaced angles.

    For each angle, cast a ray from the centroid of the ring outward:
    - Find where the ray intersects the outer contour → outer_r
    - Find where the ray intersects the inner contour → inner_r
    - Wall thickness at that angle = outer_r − inner_r

    Falls back to point-polygon-test method if ray casting is tricky.
    """
    # Use centroid of outer contour as center
    M = cv2.moments(outer)
    if M["m00"] == 0:
        return np.array([])
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    thicknesses = []
    for i in range(n_samples):
        angle = 2.0 * math.pi * i / n_samples
        # Point far away along the ray
        far_x = cx + 2000 * math.cos(angle)
        far_y = cy + 2000 * math.sin(angle)

        # Distance from center to outer contour along this direction
        d_outer = _ray_contour_distance(cx, cy, far_x, far_y, outer)
        d_inner = _ray_contour_distance(cx, cy, far_x, far_y, inner)

        if d_outer is not None and d_inner is not None and d_outer > d_inner:
            thicknesses.append(d_outer - d_inner)

    return np.array(thicknesses)


def _ray_contour_distance(cx, cy, fx, fy, contour) -> Optional[float]:
    """Find distance from (cx, cy) to contour along the ray toward (fx, fy).

    Uses the closest contour point that lies near the ray direction.
    """
    pts = contour.reshape(-1, 2).astype(float)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dists = np.sqrt(dx ** 2 + dy ** 2)

    # Ray direction
    ray_dx = fx - cx
    ray_dy = fy - cy
    ray_len = math.hypot(ray_dx, ray_dy)
    if ray_len < 1e-6:
        return None
    ray_ux = ray_dx / ray_len
    ray_uy = ray_dy / ray_len

    # Dot product gives projection along ray
    dots = dx * ray_ux + dy * ray_uy
    # Cross product gives perpendicular distance from ray
    crosses = np.abs(dx * ray_uy - dy * ray_ux)

    # Only consider points in the forward direction with small perpendicular distance
    # Use an angular tolerance of ~3 degrees
    mask = (dots > 0) & (crosses < dists * 0.05 + 3)

    if not np.any(mask):
        return None

    # Closest point along the ray direction
    candidates = dists[mask]
    return float(np.median(candidates))  # median is more robust to noise


def edge_clearance(outer, inner, img_shape) -> float:
    """Minimum distance from any contour point to the image border."""
    h, w = img_shape[:2]
    min_dist = float("inf")

    for contour in [outer, inner]:
        pts = contour.reshape(-1, 2)
        # Distance to each edge
        d_left = pts[:, 0].min()
        d_right = w - 1 - pts[:, 0].max()
        d_top = pts[:, 1].min()
        d_bottom = h - 1 - pts[:, 1].max()
        min_dist = min(min_dist, d_left, d_right, d_top, d_bottom)

    return float(min_dist) if min_dist != float("inf") else 0.0


def measure_oring(image: np.ndarray, filename: str) -> Optional[Dict]:
    """Run full measurement pipeline on a BGR image."""
    mask = build_mask(image)
    outer, inner = find_contours(mask)

    if outer is None or inner is None:
        print(f"  ✗ Could not find inner/outer contours: {filename}")
        return None

    h, w = image.shape[:2]

    # Fitted circles
    (ox, oy), orad = cv2.minEnclosingCircle(outer)
    (ix, iy), irad = cv2.minEnclosingCircle(inner)
    cdist = math.hypot(ox - ix, oy - iy)
    rthick = float(orad) - float(irad)
    mrad = (float(orad) + float(irad)) / 2.0

    # Wall thickness sampling
    thicknesses = sample_wall_thickness(outer, inner, THICKNESS_SAMPLES)
    if len(thicknesses) < 10:
        # Fallback: use point-polygon-test based method
        min_t, max_t = _contour_distances_fallback(outer, inner)
        mean_t = (min_t + max_t) / 2.0
        std_t = (max_t - min_t) / 4.0  # rough estimate
    else:
        min_t = float(np.min(thicknesses))
        max_t = float(np.max(thicknesses))
        mean_t = float(np.mean(thicknesses))
        std_t = float(np.std(thicknesses))

    cv_t = (std_t / mean_t * 100) if mean_t > 0 else 0.0

    # Circularity
    o_area = cv2.contourArea(outer)
    o_peri = cv2.arcLength(outer, True)
    circ_outer = (4.0 * math.pi * o_area / (o_peri ** 2)) if o_peri > 0 else 0.0

    i_area = cv2.contourArea(inner)
    i_peri = cv2.arcLength(inner, True)
    circ_inner = (4.0 * math.pi * i_area / (i_peri ** 2)) if i_peri > 0 else 0.0

    # Radial std
    o_rstd = radial_std(outer)
    i_rstd = radial_std(inner)

    # Eccentricity
    ecc_pct = (cdist / mrad * 100) if mrad > 0 else 0.0

    # Annular area (pixel count)
    area_px = int(np.count_nonzero(mask))

    # Edge clearance
    ec = edge_clearance(outer, inner, image.shape)

    return {
        "image": filename,
        "outer_radius": round(float(orad), 2),
        "inner_radius": round(float(irad), 2),
        "center_dist": round(cdist, 2),
        "ring_thickness": round(rthick, 2),
        "mean_radius": round(mrad, 2),
        "min_thickness": round(min_t, 2),
        "max_thickness": round(max_t, 2),
        "mean_thickness": round(mean_t, 2),
        "thickness_std": round(std_t, 2),
        "thickness_cv": round(cv_t, 2),
        "thickness_range": round(max_t - min_t, 2),
        "thickness_ratio": round((max_t / min_t) if min_t > 0 else 99.0, 3),
        "circularity_outer": round(circ_outer, 4),
        "circularity_inner": round(circ_inner, 4),
        "outer_radial_std": round(o_rstd, 2),
        "inner_radial_std": round(i_rstd, 2),
        "eccentricity_pct": round(ecc_pct, 2),
        "annular_area": area_px,
        "edge_clearance": round(ec, 1),
    }


def _contour_distances_fallback(outer, inner, sample: int = 300):
    """Fallback min/max thickness via pointPolygonTest."""
    min_d, max_d = float("inf"), 0.0
    for contour_a, contour_b in [(outer, inner), (inner, outer)]:
        step = max(1, len(contour_a) // sample)
        for i in range(0, len(contour_a), step):
            pt = contour_a[i][0]
            d = abs(cv2.pointPolygonTest(
                contour_b, (float(pt[0]), float(pt[1])), True))
            min_d = min(min_d, d)
            max_d = max(max_d, d)
    return (0.0 if min_d == float("inf") else float(min_d)), float(max_d)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"Input folder:  {INPUT_DIR}")
    print(f"BG value: {BG_VALUE}, Threshold: {BG_THRESHOLD}")
    print(f"Wall thickness samples: {THICKNESS_SAMPLES}")

    # Collect images
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    files = sorted([f for f in INPUT_DIR.iterdir() if f.suffix.lower() in exts])
    print(f"Found {len(files)} images\n")

    if not files:
        print("No images found!")
        sys.exit(1)

    # Measure all images
    rows = []
    failed = 0
    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  ✗ Cannot read: {fpath.name}")
            failed += 1
            continue

        result = measure_oring(img, fpath.name)
        if result is None:
            failed += 1
            continue

        rows.append(result)

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {fpath.name}  "
                  f"outer_r={result['outer_radius']:.1f}  "
                  f"inner_r={result['inner_radius']:.1f}  "
                  f"thick_range={result['thickness_range']:.1f}  "
                  f"circ_o={result['circularity_outer']:.4f}")

    print(f"\nProcessed: {len(rows)}/{len(files)}  (failed: {failed})")

    if not rows:
        print("No valid measurements!")
        sys.exit(1)

    # ── Save per-image CSV ────────────────────────────────────────────────
    all_cols = ["image"] + METRIC_COLS
    with open(MEASUREMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved measurements: {MEASUREMENTS_CSV}")

    # ── Compute and print statistics ──────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GOOD MODEL 2 O-RING STATISTICS  (n = {len(rows)})")
    print(f"{'='*90}")
    header = f"{'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P5':>10} {'P95':>10}"
    print(header)
    print("-" * len(header))

    stats_rows = []
    for col in METRIC_COLS:
        vals = np.array([r[col] for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))

        print(f"{col:<22} {mean:>10.2f} {std:>10.2f} {vmin:>10.2f} {vmax:>10.2f} {p5:>10.2f} {p95:>10.2f}")

        stats_rows.append({
            "metric": col,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(vmin, 4),
            "max": round(vmax, 4),
            "p5": round(p5, 4),
            "p95": round(p95, 4),
            "n": len(vals),
        })

    # Save stats CSV
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "mean", "std", "min", "max", "p5", "p95", "n"])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"\nSaved statistics: {STATS_CSV}")

    # ── Suggested thresholds (mean ± 3σ) ─────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUGGESTED THRESHOLDS  (mean ± 3σ, clipped to observed range)")
    print(f"{'='*90}")
    print(f"{'Metric':<22} {'Low':>10} {'High':>10} {'Type':>8}  Description")
    print("-" * 90)

    threshold_defs = [
        # (metric, type, description)
        ("outer_radius",      "range", "Must be within good range; too large = swollen"),
        ("inner_radius",      "range", "Must be within good range; too small = closing"),
        ("center_dist",       "max",   "Concentricity — low is better"),
        ("ring_thickness",    "range", "Mean thickness from fitted circles"),
        ("min_thickness",     "min",   "Thinnest wall — must not be too thin"),
        ("max_thickness",     "max",   "Thickest wall — must not be too thick"),
        ("mean_thickness",    "range", "Average sampled wall thickness"),
        ("thickness_std",     "max",   "Uniformity — low is better"),
        ("thickness_cv",      "max",   "Uniformity (%) — low is better"),
        ("thickness_range",   "max",   "Max−min thickness — low is better"),
        ("thickness_ratio",   "max",   "Max/min thickness — close to 1.0 is better"),
        ("circularity_outer", "min",   "Roundness of outer — 1.0 = perfect circle"),
        ("circularity_inner", "min",   "Roundness of inner — 1.0 = perfect circle"),
        ("outer_radial_std",  "max",   "Shape regularity of outer — low is better"),
        ("inner_radial_std",  "max",   "Shape regularity of inner — low is better"),
        ("eccentricity_pct",  "max",   "Centers offset — low is better"),
        ("annular_area",      "range", "Total ring area in pixels"),
        ("edge_clearance",    "min",   "Distance to image border — high means fully visible"),
    ]

    sigma = 3.0
    for metric, ttype, desc in threshold_defs:
        vals = np.array([r[metric] for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        lo = mean - sigma * std
        hi = mean + sigma * std

        if ttype == "max":
            lo_str = "—"
            hi_val = max(hi, float(np.max(vals)))
            print(f"{metric:<22} {lo_str:>10} {hi_val:>10.2f} {'≤ hi':>8}  {desc}")
        elif ttype == "min":
            lo_val = min(lo, float(np.min(vals)))
            hi_str = "—"
            print(f"{metric:<22} {lo_val:>10.2f} {hi_str:>10} {'≥ lo':>8}  {desc}")
        else:  # range
            print(f"{metric:<22} {lo:>10.2f} {hi:>10.2f} {'range':>8}  {desc}")


if __name__ == "__main__":
    main()
