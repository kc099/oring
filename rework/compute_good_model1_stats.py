"""
Compute statistical parameters for good Model 1 o-ring samples.

Reads the image list from model1good_measurements.csv (only images where
bg_value=20, threshold=30 gives perfect masks), then measures the same
comprehensive parameter set as compute_good_model2_stats.py.

Outputs:
    rework/model1good_measurements.csv       — per-image raw measurements (overwrites)
    rework/model1good_measurements_stats.csv  — mean, std, min, max, p5, p95

Usage:
    python rework/compute_good_model1_stats.py

Author: GitHub Copilot
Date: February 16, 2026
"""

import csv
import math
import sys
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────
BG_VALUE = 20
BG_THRESHOLD = 30
THICKNESS_SAMPLES = 360
MORPH_KERNEL = 7

WORKSPACE = Path(__file__).resolve().parent.parent
INPUT_DIR = WORKSPACE / "Original Data" / "model1good"
OUTPUT_DIR = Path(__file__).resolve().parent  # rework/

# Old CSV that has the list of valid images
OLD_CSV = OUTPUT_DIR / "model1good_measurements.csv"
MEASUREMENTS_CSV = OUTPUT_DIR / "model1good_measurements.csv"
STATS_CSV = OUTPUT_DIR / "model1good_measurements_stats.csv"

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


# ─── Image processing (same as model2 script) ───────────────────────────

def build_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, np.full_like(gray, BG_VALUE))
    _, binary = cv2.threshold(diff, BG_THRESHOLD, 255, cv2.THRESH_BINARY)
    if np.mean(binary == 255) > 0.75:
        binary = cv2.bitwise_not(binary)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


def find_contours(mask):
    """Uses CHAIN_APPROX_NONE for uniform contour-point density."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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


def fit_circle_lsq(contour):
    """Least-squares circle fit (Kåsa method).  Returns (cx, cy, radius)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = res
    radius = math.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    return float(cx), float(cy), float(radius)


def radial_std(contour):
    """Std dev of distances from moments-based centroid to contour points."""
    pts = contour.reshape(-1, 2).astype(float)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        cx, cy = pts.mean(axis=0)
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.std(dists))


def _ray_contour_distance(cx, cy, fx, fy, contour):
    pts = contour.reshape(-1, 2).astype(float)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dists = np.sqrt(dx ** 2 + dy ** 2)
    ray_dx = fx - cx
    ray_dy = fy - cy
    ray_len = math.hypot(ray_dx, ray_dy)
    if ray_len < 1e-6:
        return None
    ray_ux = ray_dx / ray_len
    ray_uy = ray_dy / ray_len
    dots = dx * ray_ux + dy * ray_uy
    crosses = np.abs(dx * ray_uy - dy * ray_ux)
    mask = (dots > 0) & (crosses < dists * 0.05 + 3)
    if not np.any(mask):
        return None
    return float(np.median(dists[mask]))


def sample_wall_thickness(outer, inner, n_samples=THICKNESS_SAMPLES,
                          mask=None, center=None):
    """Mask-based ray tracing for rotation-invariant wall thickness."""
    if center is not None:
        cx, cy = center
    else:
        M = cv2.moments(outer)
        if M["m00"] == 0:
            return np.array([])
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    if mask is not None:
        h, w = mask.shape[:2]
        max_r = int(math.hypot(w, h))
        thicknesses = []
        for i in range(n_samples):
            angle = 2.0 * math.pi * i / n_samples
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            inner_r = None
            outer_r = None
            for r in range(1, max_r):
                px = int(round(cx + r * cos_a))
                py = int(round(cy + r * sin_a))
                if not (0 <= px < w and 0 <= py < h):
                    break
                val = mask[py, px]
                if inner_r is None:
                    if val > 0:
                        inner_r = r
                else:
                    if val == 0:
                        outer_r = r
                        break
            if inner_r is not None and outer_r is not None:
                thicknesses.append(float(outer_r - inner_r))
        return np.array(thicknesses)

    thicknesses = []
    for i in range(n_samples):
        angle = 2.0 * math.pi * i / n_samples
        far_x = cx + 2000 * math.cos(angle)
        far_y = cy + 2000 * math.sin(angle)
        d_outer = _ray_contour_distance(cx, cy, far_x, far_y, outer)
        d_inner = _ray_contour_distance(cx, cy, far_x, far_y, inner)
        if d_outer is not None and d_inner is not None and d_outer > d_inner:
            thicknesses.append(d_outer - d_inner)
    return np.array(thicknesses)


def _contour_distances_fallback(outer, inner, sample=300):
    min_d, max_d = float("inf"), 0.0
    for contour_a, contour_b in [(outer, inner), (inner, outer)]:
        step = max(1, len(contour_a) // sample)
        for i in range(0, len(contour_a), step):
            pt = contour_a[i][0]
            d = abs(cv2.pointPolygonTest(contour_b, (float(pt[0]), float(pt[1])), True))
            min_d = min(min_d, d)
            max_d = max(max_d, d)
    return (0.0 if min_d == float("inf") else float(min_d)), float(max_d)


def edge_clearance(outer, inner, img_shape):
    h, w = img_shape[:2]
    min_dist = float("inf")
    for contour in [outer, inner]:
        pts = contour.reshape(-1, 2)
        d_left = pts[:, 0].min()
        d_right = w - 1 - pts[:, 0].max()
        d_top = pts[:, 1].min()
        d_bottom = h - 1 - pts[:, 1].max()
        min_dist = min(min_dist, d_left, d_right, d_top, d_bottom)
    return float(min_dist) if min_dist != float("inf") else 0.0


def measure_oring(image, filename):
    mask = build_mask(image)
    outer, inner = find_contours(mask)
    if outer is None or inner is None:
        print(f"  ✗ Could not find contours: {filename}")
        return None

    h, w = image.shape[:2]
    ox, oy, orad = fit_circle_lsq(outer)
    ix, iy, irad = fit_circle_lsq(inner)
    cdist = math.hypot(ox - ix, oy - iy)
    rthick = float(orad) - float(irad)
    mrad = (float(orad) + float(irad)) / 2.0

    thicknesses = sample_wall_thickness(
        outer, inner, THICKNESS_SAMPLES, mask=mask, center=(ox, oy))
    if len(thicknesses) < 10:
        min_t, max_t = _contour_distances_fallback(outer, inner)
        mean_t = (min_t + max_t) / 2.0
        std_t = (max_t - min_t) / 4.0
    else:
        min_t = float(np.min(thicknesses))
        max_t = float(np.max(thicknesses))
        mean_t = float(np.mean(thicknesses))
        std_t = float(np.std(thicknesses))

    cv_t = (std_t / mean_t * 100) if mean_t > 0 else 0.0

    o_area = cv2.contourArea(outer)
    o_peri = cv2.arcLength(outer, True)
    circ_outer = (4.0 * math.pi * o_area / (o_peri ** 2)) if o_peri > 0 else 0.0

    i_area = cv2.contourArea(inner)
    i_peri = cv2.arcLength(inner, True)
    circ_inner = (4.0 * math.pi * i_area / (i_peri ** 2)) if i_peri > 0 else 0.0

    o_rstd = radial_std(outer)
    i_rstd = radial_std(inner)
    ecc_pct = (cdist / mrad * 100) if mrad > 0 else 0.0
    area_px = int(np.count_nonzero(mask))
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


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    # 1. Read valid image names from old CSV
    if not OLD_CSV.exists():
        print(f"ERROR: {OLD_CSV} not found")
        sys.exit(1)

    valid_images = []
    with open(OLD_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            valid_images.append(r["image"])

    print(f"Model 1 Good — using {len(valid_images)} validated images from {OLD_CSV.name}")
    print(f"Input folder:  {INPUT_DIR}")
    print(f"BG value: {BG_VALUE}, Threshold: {BG_THRESHOLD}\n")

    # 2. Measure each image
    rows = []
    failed = 0
    for i, fname in enumerate(valid_images):
        fpath = INPUT_DIR / fname
        if not fpath.exists():
            print(f"  ✗ File not found: {fname}")
            failed += 1
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  ✗ Cannot read: {fname}")
            failed += 1
            continue

        result = measure_oring(img, fname)
        if result is None:
            failed += 1
            continue

        rows.append(result)
        print(f"  [{i+1}/{len(valid_images)}] {fname}  "
              f"outer_r={result['outer_radius']:.1f}  "
              f"inner_r={result['inner_radius']:.1f}  "
              f"thick_range={result['thickness_range']:.1f}  "
              f"circ_o={result['circularity_outer']:.4f}")

    print(f"\nProcessed: {len(rows)}/{len(valid_images)}  (failed: {failed})")

    if not rows:
        print("No valid measurements!")
        sys.exit(1)

    # 3. Save per-image CSV
    all_cols = ["image"] + METRIC_COLS
    with open(MEASUREMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved measurements: {MEASUREMENTS_CSV}")

    # 4. Compute and print statistics
    print(f"\n{'='*90}")
    print(f"  GOOD MODEL 1 O-RING STATISTICS  (n = {len(rows)})")
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

    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "mean", "std", "min", "max", "p5", "p95", "n"])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"\nSaved statistics: {STATS_CSV}")

    # 5. Suggested thresholds
    print(f"\n{'='*90}")
    print(f"  SUGGESTED THRESHOLDS  (mean ± 3σ)")
    print(f"{'='*90}")
    print(f"{'Metric':<22} {'Low':>10} {'High':>10} {'Type':>8}")
    print("-" * 60)

    threshold_defs = [
        ("outer_radius",      "range"),
        ("inner_radius",      "range"),
        ("center_dist",       "max"),
        ("ring_thickness",    "range"),
        ("min_thickness",     "min"),
        ("max_thickness",     "max"),
        ("mean_thickness",    "range"),
        ("thickness_std",     "max"),
        ("thickness_cv",      "max"),
        ("thickness_range",   "max"),
        ("thickness_ratio",   "max"),
        ("circularity_outer", "min"),
        ("circularity_inner", "min"),
        ("outer_radial_std",  "max"),
        ("inner_radial_std",  "max"),
        ("eccentricity_pct",  "max"),
        ("annular_area",      "range"),
        ("edge_clearance",    "min"),
    ]

    sigma = 3.0
    for metric, ttype in threshold_defs:
        vals = np.array([r[metric] for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo = mean - sigma * std
        hi = mean + sigma * std

        if ttype == "max":
            print(f"{metric:<22} {'—':>10} {max(hi, float(np.max(vals))):>10.2f} {'≤ hi':>8}")
        elif ttype == "min":
            print(f"{metric:<22} {min(lo, float(np.min(vals))):>10.2f} {'—':>10} {'≥ lo':>8}")
        else:
            print(f"{metric:<22} {lo:>10.2f} {hi:>10.2f} {'range':>8}")


if __name__ == "__main__":
    main()
