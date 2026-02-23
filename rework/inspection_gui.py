"""
O-Ring Inspection GUI — Pass / Rework / Reject Classification

Loads a full-resolution (2448×2048) o-ring image, performs background
subtraction to segment the ring, detects inner/outer contours, computes
geometric measurements, and compares against statistical thresholds
derived from known-good samples.

Verdict logic (3-way, checked in this order):
    PASS    — all metrics within tolerance
    REWORK  — shape issues (circularity, radial_std failures)
              that are fixable by trimming sharp edges
    REJECT  — wall thickness / concentricity / area deviations
              or cut at edges (edge_clearance too low) — unfixable

Thresholds are auto-computed from per-model CSV files.
Every threshold is editable in the UI; the σ-multiplier controls how
many standard deviations from the good-sample mean define tolerance.

Usage:
    python rework/inspection_gui.py

Author: GitHub Copilot
Date: February 16, 2026
"""

import sys
import csv
import json
import math
import os
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import torch

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QComboBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# ── Hikrobot Camera SDK (optional — only needed for live camera) ─────────
try:
    _MV_DLL_PATH = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
    if os.path.isdir(_MV_DLL_PATH):
        os.add_dll_directory(_MV_DLL_PATH)
    from MvImport.MvCameraControl_class import *   # noqa
    HIKROBOT_AVAILABLE = True
except Exception:
    HIKROBOT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE  = SCRIPT_DIR.parent

# ── Mask R-CNN checkpoint (binary mode) ──────────────────────────────────
MASKRCNN_CHECKPOINT = (
    WORKSPACE / "maskrcnn" / "dataset" / "combined" / "checkpoints" / "best_model.pth"
)

# ── YOLO v11 segmentation checkpoint ─────────────────────────────────────
# Try yolo26n-seg first, fall back to yolo11n-seg
_YOLO_26_CKPT = WORKSPACE / "yolo_training" / "runs" / "yolo26n-seg_training" / "weights" / "best.pt"
_YOLO_11_CKPT = WORKSPACE / "yolo_training" / "runs" / "yolo11n-seg_training2" / "weights" / "best.pt"
YOLO_CHECKPOINT = _YOLO_26_CKPT if _YOLO_26_CKPT.exists() else _YOLO_11_CKPT

# Defect detection model options
DEFECT_MODEL_OPTIONS = ["Mask R-CNN", "YOLO v11"]

# Reference resolution — thresholds are calibrated at this size
REFERENCE_RESOLUTION = (2448, 2048)   # (width, height) of original camera images

# How each metric scales with resolution
#   "linear"  — proportional to pixel size  (divide by scale)
#   "area"    — proportional to pixel area   (divide by scale²)
#   "none"    — dimensionless ratio / percentage (no scaling)
METRIC_SCALE_TYPE = {
    "outer_radius":      "linear",
    "inner_radius":      "linear",
    "ring_thickness":    "linear",
    "mean_thickness":    "linear",
    "min_thickness":     "linear",
    "max_thickness":     "linear",
    "thickness_range":   "linear",
    "thickness_std":     "linear",
    "center_dist":       "linear",
    "edge_clearance":    "linear",
    "outer_radial_std":  "linear",
    "inner_radial_std":  "linear",
    "annular_area_k":    "area",
    "circularity_outer": "none",
    "circularity_inner": "none",
    "thickness_ratio":   "none",
    "thickness_cv":      "none",
    "eccentricity_pct":  "none",
}


def compute_resolution_scale(img_w: int, img_h: int) -> float:
    """Return scale factor relative to REFERENCE_RESOLUTION.

    scale = 1.0 for original resolution, 0.5 for 2×2 binned, etc.
    Uses the larger dimension to be robust to slight aspect changes.
    """
    ref = max(REFERENCE_RESOLUTION)
    cur = max(img_w, img_h)
    return cur / ref


def normalize_measurements(result: Dict, scale: float) -> Dict:
    """Scale pixel-based measurements back to reference resolution.

    Divides linear metrics by *scale* and area metrics by *scale²*
    so that the same thresholds work regardless of input resolution.
    Dimensionless metrics are left unchanged.
    Returns a **new** dict (original is not mutated).
    """
    if abs(scale - 1.0) < 1e-6:
        return result          # nothing to do at native resolution

    normed = dict(result)      # shallow copy
    for key, stype in METRIC_SCALE_TYPE.items():
        if key not in normed:
            continue
        if stype == "linear":
            normed[key] = normed[key] / scale
        elif stype == "area":
            normed[key] = normed[key] / (scale * scale)
    return normed


# Per-model CSV paths (new format with comprehensive metrics)
MODEL_CSV = {
    "Model 1": SCRIPT_DIR / "model1good_measurements.csv",
    "Model 2": SCRIPT_DIR / "good_measurements.csv",
}

# Per-model tuned threshold JSON (generated by tune_thresholds.py)
TUNED_JSON = {
    "Model 1": SCRIPT_DIR / "model1_tuned_thresholds.json",
    "Model 2": SCRIPT_DIR / "model2_tuned_thresholds.json",
}

DEFAULT_MODEL = "Model 2"

# ── Metric definitions ────────────────────────────────────────────────────
# (key, display_name, unit, thresh_type, decimals, spin_step, spin_lo, spin_hi, verdict_category)
#
#   thresh_type:  'max'   → value must be ≤ hi
#                 'min'   → value must be ≥ lo
#                 'range' → lo ≤ value ≤ hi
#
#   verdict_category:
#       "rework"  — failure → REWORK (shape fixable by trimming sharp edges)
#       "reject"  — failure → REJECT (size / thickness / concentricity — unfixable)

METRIC_DEFS = [
    # ── REWORK metrics (shape — sharp edges can be trimmed) ───────────────
    ("outer_radius",     "Outer Radius",          "px",   "range", 1, 1.0,  400, 1200, "rework"),
    ("inner_radius",     "Inner Radius",          "px",   "range", 1, 1.0,  200, 800,  "rework"),
    ("circularity_outer","Outer Circularity",     "",     "min",   3, 0.005,  0, 1,    "rework"),
    ("circularity_inner","Inner Circularity",     "",     "min",   3, 0.005,  0, 1,    "rework"),
    ("outer_radial_std", "Outer Radial Std",      "px",   "max",   1, 1.0,    0, 200,  "rework"),
    ("inner_radial_std", "Inner Radial Std",      "px",   "max",   1, 1.0,    0, 200,  "rework"),

    # ── REJECT metrics (thickness / concentricity / area — unfixable) ────
    ("ring_thickness",   "Ring Thickness (fitted)","px",  "range", 1, 1.0,  100, 600,  "reject"),
    ("mean_thickness",   "Mean Wall Thickness",   "px",   "range", 1, 1.0,  100, 600,  "reject"),
    ("min_thickness",    "Min Wall Thickness",    "px",   "min",   1, 1.0,    0, 500,  "reject"),
    ("max_thickness",    "Max Wall Thickness",    "px",   "max",   1, 1.0,    0, 600,  "reject"),
    ("thickness_range",  "Thickness Range",       "px",   "max",   1, 1.0,    0, 500,  "reject"),
    ("thickness_ratio",  "Thickness Ratio",       "",     "max",   2, 0.01,   1, 3,    "reject"),
    ("thickness_std",    "Thickness Std Dev",     "px",   "max",   1, 0.5,    0, 100,  "reject"),
    ("thickness_cv",     "Thickness CV",          "%",    "max",   2, 0.1,    0, 50,   "reject"),
    ("center_dist",      "Center Distance",       "px",   "max",   1, 1.0,    0, 500,  "reject"),
    ("eccentricity_pct", "Eccentricity",          "%",    "max",   2, 0.1,    0, 50,   "reject"),
    ("annular_area_k",   "Annular Area (×1000)",  "",     "range", 1, 5.0,    0, 5000, "reject"),
    ("edge_clearance",   "Edge Clearance",        "px",   "min",   0, 1.0,    0, 1000, "reject"),
]

# Fallback thresholds when CSV is missing
DEFAULT_THRESHOLDS = {
    "outer_radius":      {"lo": 650.0, "hi": 680.0},
    "inner_radius":      {"lo": 375.0, "hi": 400.0},
    "ring_thickness":    {"lo": 260.0, "hi": 295.0},
    "mean_thickness":    {"lo": 260.0, "hi": 295.0},
    "min_thickness":     {"lo": 230.0, "hi": 9999.0},
    "max_thickness":     {"lo": 0.0,   "hi": 325.0},
    "thickness_range":   {"lo": 0.0,   "hi": 60.0},
    "thickness_ratio":   {"lo": 1.0,   "hi": 1.25},
    "thickness_std":     {"lo": 0.0,   "hi": 15.0},
    "thickness_cv":      {"lo": 0.0,   "hi": 5.5},
    "center_dist":       {"lo": 0.0,   "hi": 35.0},
    "eccentricity_pct":  {"lo": 0.0,   "hi": 6.0},
    "annular_area_k":    {"lo": 780.0, "hi": 950.0},
    "circularity_outer": {"lo": 0.75,  "hi": 1.0},
    "circularity_inner": {"lo": 0.75,  "hi": 1.0},
    "outer_radial_std":  {"lo": 0.0,   "hi": 40.0},
    "inner_radial_std":  {"lo": 0.0,   "hi": 30.0},
    "edge_clearance":    {"lo": 5.0,   "hi": 9999.0},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics / threshold helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_good_stats(csv_path: Path) -> Optional[Dict]:
    """Return per-metric {mean, std} from the good-sample measurements CSV.

    Supports both old format (with 'method' column) and new format
    (direct metric columns from compute_good_model*_stats.py).
    """
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if len(rows) < 3:
        return None

    def _ms(vals):
        a = np.array(vals, dtype=float)
        return float(np.mean(a)), (float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)

    stats: Dict[str, Dict] = {}

    # Direct columns
    direct_cols = [
        "outer_radius", "inner_radius", "center_dist",
        "ring_thickness", "mean_radius",
        "min_thickness", "max_thickness",
    ]
    # New comprehensive columns
    new_cols = [
        "mean_thickness", "thickness_std", "thickness_cv",
        "thickness_range", "thickness_ratio",
        "circularity_outer", "circularity_inner",
        "outer_radial_std", "inner_radial_std",
        "eccentricity_pct", "edge_clearance",
    ]

    for key in direct_cols + new_cols:
        if key in rows[0]:
            vals = [float(r[key]) for r in rows if r.get(key)]
            if vals:
                m, s = _ms(vals)
                stats[key] = {"mean": m, "std": s}

    # Derived: thickness_range (if not directly available)
    if "thickness_range" not in stats and "max_thickness" in rows[0] and "min_thickness" in rows[0]:
        ranges = [float(r["max_thickness"]) - float(r["min_thickness"]) for r in rows]
        m, s = _ms(ranges)
        stats["thickness_range"] = {"mean": m, "std": s}

    # Derived: thickness_ratio (if not directly available)
    if "thickness_ratio" not in stats and "max_thickness" in rows[0] and "min_thickness" in rows[0]:
        ratios = [float(r["max_thickness"]) / float(r["min_thickness"])
                  for r in rows if float(r.get("min_thickness", 0)) > 0]
        if ratios:
            m, s = _ms(ratios)
            stats["thickness_ratio"] = {"mean": m, "std": s}

    # Derived: annular_area_k
    if "annular_area" in rows[0]:
        areas_k = [float(r["annular_area"]) / 1000.0 for r in rows]
        m, s = _ms(areas_k)
        stats["annular_area_k"] = {"mean": m, "std": s}

    # Derived: eccentricity_pct (if not directly available)
    if "eccentricity_pct" not in stats and "center_dist" in rows[0] and "mean_radius" in rows[0]:
        ecc = [float(r["center_dist"]) / float(r["mean_radius"]) * 100 for r in rows]
        m, s = _ms(ecc)
        stats["eccentricity_pct"] = {"mean": m, "std": s}

    return stats


def load_tuned_thresholds(json_path: Path) -> Optional[Dict[str, Dict]]:
    """Load tuned thresholds from JSON (generated by tune_thresholds.py)."""
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # JSON has {key: {lo, hi, tolerance_pct}} — keep only lo/hi
    thresholds = {}
    for key in data:
        thresholds[key] = {"lo": data[key]["lo"], "hi": data[key]["hi"]}
    return thresholds


def compute_thresholds(stats: Optional[Dict], sigma: float = 2.5) -> Dict[str, Dict]:
    """Compute {lo, hi} per metric from good-sample stats (fallback)."""
    if stats is None:
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}

    thresholds: Dict[str, Dict] = {}
    for key, _name, _unit, ttype, *_ in METRIC_DEFS:
        if key in stats:
            m, s = stats[key]["mean"], stats[key]["std"]
            lo = round(m - sigma * s, 4)
            hi = round(m + sigma * s, 4)
            if ttype == "max":
                thresholds[key] = {"lo": 0.0, "hi": hi}
            elif ttype == "min":
                thresholds[key] = {"lo": max(lo, 0.0), "hi": 9999.0}
            else:
                thresholds[key] = {"lo": lo, "hi": hi}
        else:
            thresholds[key] = dict(DEFAULT_THRESHOLDS.get(key, {"lo": 0, "hi": 9999}))
    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
#  Image processing
# ═══════════════════════════════════════════════════════════════════════════

def _largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


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
    """Return (outer, inner) contours of the ring, or (None, None).

    Uses CHAIN_APPROX_NONE to keep every boundary pixel so that
    contour-point-based metrics (radial_std, ray distances) are
    rotationally invariant — CHAIN_APPROX_SIMPLE compresses
    horizontal / vertical / diagonal runs to their endpoints,
    producing a point density that changes with orientation.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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


def fit_circle_lsq(contour: np.ndarray):
    """Least-squares circle fit (Kåsa method).

    All contour points contribute equally, making the result much
    more rotationally stable than ``cv2.minEnclosingCircle`` (which
    depends on only 2–3 extreme hull points that shift with pixel
    discretisation when the part is rotated).

    Returns (cx, cy, radius).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    # Solve  2·a·x + 2·b·y + c = x² + y²  in the least-squares sense
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = res
    radius = math.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    return float(cx), float(cy), float(radius)


def radial_std(contour: np.ndarray) -> float:
    """Std dev of distances from centroid to contour points.  0 = perfect circle.

    Uses the area-weighted centroid (``cv2.moments``) instead of the
    plain point-mean so the result does not depend on how contour
    points are distributed — only on the enclosed shape.
    """
    pts = contour.reshape(-1, 2).astype(float)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        cx, cy = pts.mean(axis=0)        # fallback
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.std(dists))


def _ray_contour_distance(cx, cy, fx, fy, contour):
    """Distance from (cx,cy) to contour along ray toward (fx,fy)."""
    pts = contour.reshape(-1, 2).astype(float)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dists = np.sqrt(dx ** 2 + dy ** 2)
    ray_dx, ray_dy = fx - cx, fy - cy
    ray_len = math.hypot(ray_dx, ray_dy)
    if ray_len < 1e-6:
        return None
    ray_ux, ray_uy = ray_dx / ray_len, ray_dy / ray_len
    dots = dx * ray_ux + dy * ray_uy
    crosses = np.abs(dx * ray_uy - dy * ray_ux)
    mask = (dots > 0) & (crosses < dists * 0.05 + 3)
    if not np.any(mask):
        return None
    return float(np.median(dists[mask]))


def sample_wall_thickness(outer, inner, n_samples: int = 360,
                          mask: np.ndarray = None,
                          center: tuple = None):
    """Sample wall thickness at evenly-spaced angles around the ring.

    Parameters
    ----------
    outer, inner : contours (used only as fallback when *mask* is None)
    n_samples    : number of angular samples
    mask         : binary mask of the ring (preferred — rotation-invariant)
    center       : (cx, cy) ray origin; defaults to outer moments centroid

    When *mask* is supplied the thickness at each angle is measured by
    walking along a ray on the mask and detecting the 0→255 (inner edge)
    and 255→0 (outer edge) transitions.  This is independent of contour
    point density and therefore fully rotation-invariant.
    """
    # Determine centre
    if center is not None:
        cx, cy = center
    else:
        M = cv2.moments(outer)
        if M["m00"] == 0:
            return np.array([])
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    # ── Mask-based ray tracing (preferred) ────────────────────────────
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

    # ── Fallback: contour-point ray casting ───────────────────────────
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


def contour_distances(outer, inner, sample: int = 300):
    """Fallback min / max wall thickness via pointPolygonTest."""
    min_d, max_d = float("inf"), 0.0
    for contour_a, contour_b in [(outer, inner), (inner, outer)]:
        step = max(1, len(contour_a) // sample)
        for i in range(0, len(contour_a), step):
            pt = contour_a[i][0]
            d = abs(cv2.pointPolygonTest(
                contour_b, (float(pt[0]), float(pt[1])), True))
            if d < min_d:
                min_d = d
            if d > max_d:
                max_d = d
    return (0.0 if min_d == float("inf") else float(min_d)), float(max_d)


def edge_clearance(outer, inner, img_shape) -> float:
    """Min distance from any contour point to the image border."""
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


def measure_oring(image: np.ndarray,
                  bg_value: int = 20,
                  threshold: int = 30) -> Optional[Dict]:
    """Run full measurement pipeline on a BGR image.

    Uses least-squares circle fitting (Kåsa method) instead of
    ``cv2.minEnclosingCircle`` so that radius / centre estimates
    are driven by *all* contour points rather than 2–3 extremes,
    making them robust to rotation in discrete pixel space.

    Wall thickness is sampled via mask-based ray tracing from the
    fitted outer-circle centre, eliminating any dependency on
    contour-point density.
    """
    mask = build_mask(image, bg_value, threshold)
    outer, inner = find_contours(mask)
    if outer is None or inner is None:
        return None

    h, w = image.shape[:2]

    # Least-squares circle fit (rotation-invariant)
    ox, oy, orad = fit_circle_lsq(outer)
    ix, iy, irad = fit_circle_lsq(inner)
    cdx, cdy = float(ox - ix), float(oy - iy)
    cdist = math.hypot(cdx, cdy)
    rthick = float(orad) - float(irad)
    mrad = (float(orad) + float(irad)) / 2.0

    # Wall thickness via mask-based ray sampling from fitted outer centre
    thicknesses = sample_wall_thickness(
        outer, inner, 360, mask=mask, center=(ox, oy))
    if len(thicknesses) < 10:
        min_t, max_t = contour_distances(outer, inner)
        mean_t = (min_t + max_t) / 2.0
        std_t = (max_t - min_t) / 4.0
    else:
        min_t = float(np.min(thicknesses))
        max_t = float(np.max(thicknesses))
        mean_t = float(np.mean(thicknesses))
        std_t = float(np.std(thicknesses))

    cv_t = (std_t / mean_t * 100) if mean_t > 0 else 0.0

    area_px = int(np.count_nonzero(mask))

    # Circularity: 4πA / P²
    o_area = cv2.contourArea(outer)
    o_peri = cv2.arcLength(outer, True)
    circ_o = (4.0 * math.pi * o_area / (o_peri ** 2)) if o_peri > 0 else 0.0

    i_area = cv2.contourArea(inner)
    i_peri = cv2.arcLength(inner, True)
    circ_i = (4.0 * math.pi * i_area / (i_peri ** 2)) if i_peri > 0 else 0.0

    o_rstd = radial_std(outer)
    i_rstd = radial_std(inner)

    ec = edge_clearance(outer, inner, image.shape)

    return {
        "outer_radius":      float(orad),
        "inner_radius":      float(irad),
        "center_dist":       cdist,
        "center_dx":         cdx,
        "center_dy":         cdy,
        "ring_thickness":    rthick,
        "mean_thickness":    mean_t,
        "min_thickness":     min_t,
        "max_thickness":     max_t,
        "thickness_std":     std_t,
        "thickness_cv":      cv_t,
        "thickness_range":   max_t - min_t,
        "thickness_ratio":   (max_t / min_t) if min_t > 0 else 99.0,
        "mean_radius":       mrad,
        "annular_area":      area_px,
        "annular_area_k":    area_px / 1000.0,
        "eccentricity_pct":  (cdist / mrad * 100) if mrad > 0 else 0,
        "circularity_outer": circ_o,
        "circularity_inner": circ_i,
        "outer_radial_std":  o_rstd,
        "inner_radial_std":  i_rstd,
        "edge_clearance":    ec,
        # keep for overlay drawing
        "mask":              mask,
        "outer_contour":     outer,
        "inner_contour":     inner,
        "outer_center":      (float(ox), float(oy)),
        "inner_center":      (float(ix), float(iy)),
    }


def draw_overlay(image: np.ndarray, result: Dict) -> np.ndarray:
    """Draw contours, centres and offset line on a copy of the image."""
    vis = image.copy()
    outer, inner = result["outer_contour"], result["inner_contour"]
    ox, oy = result["outer_center"]
    ix, iy = result["inner_center"]

    cv2.drawContours(vis, [outer], -1, (0, 255, 0), 3)
    cv2.drawContours(vis, [inner], -1, (0, 0, 255), 3)

    cv2.circle(vis, (int(ox), int(oy)), 10, (0, 255, 0), -1)
    cv2.circle(vis, (int(ix), int(iy)), 10, (0, 0, 255), -1)
    cv2.line(vis, (int(ox), int(oy)), (int(ix), int(iy)), (0, 255, 255), 2)

    cv2.circle(vis, (int(ox), int(oy)), int(result["outer_radius"]),
               (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(vis, (int(ix), int(iy)), int(result["inner_radius"]),
               (0, 0, 255), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "Green = Outer", (20, 40), font, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, "Red   = Inner", (20, 80), font, 1.0, (0, 0, 255), 2)
    cv2.putText(vis, "Yellow = Offset", (20, 120), font, 1.0, (0, 255, 255), 2)

    return vis


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════

class InspectionGUI(QMainWindow):
    """Main inspection window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("O-Ring Inspection — Pass / Rework / Reject")
        self.setGeometry(40, 40, 1600, 960)

        self.image: Optional[np.ndarray] = None
        self.overlay_image: Optional[np.ndarray] = None
        self.result: Optional[Dict] = None
        self.result_normed: Optional[Dict] = None
        self._resolution_scale: float = 1.0

        # Defect detection models (lazy-loaded on first PASS verdict)
        self._detector = None          # Mask R-CNN
        self._yolo_detector = None     # YOLO v11
        self._defect_model = "Mask R-CNN"  # Current selection
        self._pred_overlay: Optional[np.ndarray] = None
        self._pred_mask: Optional[np.ndarray] = None
        self._pred_result: Optional[Dict] = None

        # Camera (Hikrobot)
        self._camera = None
        self._camera_streaming = False
        self._stream_timer: Optional[QTimer] = None
        self._latest_frame: Optional[np.ndarray] = None

        # File navigation
        self._file_list: list = []   # sorted image paths in current folder
        self._file_index: int = -1   # current index in _file_list

        # Statistics & thresholds
        self.current_model = DEFAULT_MODEL
        self.good_stats = load_good_stats(MODEL_CSV[self.current_model])
        self.sigma = 2.5
        self.thresholds = self._load_best_thresholds()

        self.lo_spins: Dict[str, QDoubleSpinBox] = {}
        self.hi_spins: Dict[str, QDoubleSpinBox] = {}

        self._init_ui()
        self._populate_table()

        tuned_path = TUNED_JSON.get(self.current_model)
        if tuned_path and tuned_path.exists():
            self.info_label.setText(
                f"✓ Loaded tuned thresholds from {tuned_path.name}  "
                f"({self.current_model})")
        elif self.good_stats:
            csv_name = MODEL_CSV[self.current_model].name
            self.info_label.setText(
                f"✓ Loaded σ-based thresholds from {csv_name}  "
                f"({self.current_model}, σ = {self.sigma})")
        else:
            self.info_label.setText(
                f"⚠ No threshold data for {self.current_model} — using defaults")

    # ── UI setup ──────────────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ── LEFT: image + mask ────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self.img_label = QLabel("Load an image to begin inspection")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(700, 500)
        self.img_label.setStyleSheet(
            "background:#2b2b2b; color:#aaa; font-size:16px; border-radius:6px;")
        left_lay.addWidget(self.img_label, stretch=3)

        self.mask_label = QLabel("Binary mask preview")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setMinimumHeight(200)
        self.mask_label.setStyleSheet(
            "background:#1f1f1f; color:#888; font-size:13px; border-radius:6px;")
        left_lay.addWidget(self.mask_label, stretch=1)

        # Mask R-CNN prediction row (720×720 crop → model)
        pred_row = QHBoxLayout()
        self.pred_overlay_label = QLabel("Defect model prediction (PASS images)")
        self.pred_overlay_label.setAlignment(Qt.AlignCenter)
        self.pred_overlay_label.setMinimumHeight(180)
        self.pred_overlay_label.setStyleSheet(
            "background:#1a1a2e; color:#666; font-size:11px; border-radius:6px;")
        pred_row.addWidget(self.pred_overlay_label, stretch=1)

        self.pred_mask_label = QLabel("Predicted defect mask")
        self.pred_mask_label.setAlignment(Qt.AlignCenter)
        self.pred_mask_label.setMinimumHeight(180)
        self.pred_mask_label.setStyleSheet(
            "background:#1a1a2e; color:#666; font-size:11px; border-radius:6px;")
        pred_row.addWidget(self.pred_mask_label, stretch=1)
        left_lay.addLayout(pred_row, stretch=1)

        root.addWidget(left, stretch=3)

        # ── RIGHT: controls + results ─────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        # --- Detection settings -------------------------------------------
        sg = QGroupBox("Detection Settings")
        sl = QVBoxLayout()

        # Model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("O-Ring Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CSV.keys()))
        self.model_combo.setCurrentText(self.current_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_combo.setStyleSheet("padding:3px 8px; font-size:13px;")
        model_row.addWidget(self.model_combo)
        sl.addLayout(model_row)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.setStyleSheet("font-size:13px; padding:6px 14px;")
        self.load_btn.clicked.connect(self.load_image)
        btn_row.addWidget(self.load_btn)

        self.analyze_btn = QPushButton("🔍 Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(
            "font-size:13px; padding:6px 14px; "
            "background:#4CAF50; color:white; font-weight:bold;")
        self.analyze_btn.clicked.connect(self.analyze)
        btn_row.addWidget(self.analyze_btn)
        sl.addLayout(btn_row)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet("font-size:13px; padding:6px 14px;")
        self.prev_btn.clicked.connect(self._load_prev_image)
        nav_row.addWidget(self.prev_btn)

        self.nav_label = QLabel("")
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.setStyleSheet("color:#aaa; font-size:12px;")
        nav_row.addWidget(self.nav_label)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("font-size:13px; padding:6px 14px;")
        self.next_btn.clicked.connect(self._load_next_image)
        nav_row.addWidget(self.next_btn)
        sl.addLayout(nav_row)

        # Camera controls
        cam_row = QHBoxLayout()
        self.stream_btn = QPushButton("📷 Start Stream")
        self.stream_btn.setStyleSheet("font-size:13px; padding:6px 14px;")
        self.stream_btn.clicked.connect(self._toggle_stream)
        cam_row.addWidget(self.stream_btn)

        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet(
            "font-size:13px; padding:6px 14px; "
            "background:#1976D2; color:white; font-weight:bold;")
        self.capture_btn.clicked.connect(self._capture_frame)
        cam_row.addWidget(self.capture_btn)
        sl.addLayout(cam_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("BG Value:"))
        self.bg_spin = QSpinBox()
        self.bg_spin.setRange(0, 255)
        self.bg_spin.setValue(20)
        param_row.addWidget(self.bg_spin)

        self.auto_bg_btn = QPushButton("Auto")
        self.auto_bg_btn.setToolTip("Detect background from image corners")
        self.auto_bg_btn.setFixedWidth(50)
        self.auto_bg_btn.clicked.connect(self._auto_bg)
        param_row.addWidget(self.auto_bg_btn)

        param_row.addSpacing(12)
        param_row.addWidget(QLabel("Threshold:"))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(1, 255)
        self.thresh_spin.setValue(30)
        param_row.addWidget(self.thresh_spin)
        sl.addLayout(param_row)

        # Defect detection model selector
        defect_model_row = QHBoxLayout()
        defect_model_row.addWidget(QLabel("Defect Model:"))
        self.defect_model_combo = QComboBox()
        self.defect_model_combo.addItems(DEFECT_MODEL_OPTIONS)
        self.defect_model_combo.setCurrentText(self._defect_model)
        self.defect_model_combo.currentTextChanged.connect(self._on_defect_model_changed)
        self.defect_model_combo.setStyleSheet("padding:3px 8px; font-size:13px;")
        defect_model_row.addWidget(self.defect_model_combo)
        sl.addLayout(defect_model_row)

        sg.setLayout(sl)
        right_lay.addWidget(sg)

        # --- Verdict banner -----------------------------------------------
        self.verdict_frame = QFrame()
        self.verdict_frame.setMinimumHeight(110)
        self.verdict_frame.setStyleSheet(
            "background:#555; border-radius:12px; padding:8px;")
        vfl = QVBoxLayout(self.verdict_frame)
        self.verdict_label = QLabel("AWAITING")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.verdict_label.setStyleSheet("color:#ccc;")
        vfl.addWidget(self.verdict_label)
        self.verdict_detail = QLabel("Load an image and click Analyze")
        self.verdict_detail.setAlignment(Qt.AlignCenter)
        self.verdict_detail.setFont(QFont("Arial", 10))
        self.verdict_detail.setStyleSheet("color:#bbb;")
        self.verdict_detail.setWordWrap(True)
        vfl.addWidget(self.verdict_detail)
        right_lay.addWidget(self.verdict_frame)

        # --- Sigma / reset ------------------------------------------------
        tg = QGroupBox("Threshold Settings")
        tl = QHBoxLayout()
        tl.addWidget(QLabel("σ multiplier:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 5.0)
        self.sigma_spin.setSingleStep(0.25)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setValue(self.sigma)
        self.sigma_spin.valueChanged.connect(self._recompute_thresholds)
        tl.addWidget(self.sigma_spin)
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setToolTip("Reset all thresholds to σ = 2.5")
        reset_btn.clicked.connect(self._reset_thresholds)
        tl.addWidget(reset_btn)
        tg.setLayout(tl)
        right_lay.addWidget(tg)

        # --- Metrics table ------------------------------------------------
        mg = QGroupBox("Measurements && Thresholds  (editable)")
        ml = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Metric", "Measured", "Min", "Max", "Status", "Category"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3, 4, 5):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setRowCount(len(METRIC_DEFS))
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        ml.addWidget(self.table)
        mg.setLayout(ml)
        right_lay.addWidget(mg, stretch=1)

        # --- Extra info line ----------------------------------------------
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color:#888; font-size:10px;")
        self.info_label.setWordWrap(True)
        right_lay.addWidget(self.info_label)

        root.addWidget(right, stretch=2)

    def _populate_table(self):
        """Fill table rows with metric labels, threshold spinboxes."""
        for row, (key, name, unit, ttype, dec, step, s_lo, s_hi, category) in \
                enumerate(METRIC_DEFS):
            label = f"{name}" + (f"  ({unit})" if unit else "")

            # Col 0  – metric name
            item0 = QTableWidgetItem(label)
            item0.setFlags(item0.flags() & ~Qt.ItemIsEditable)
            if self.good_stats and key in self.good_stats:
                gs = self.good_stats[key]
                item0.setToolTip(
                    f"Good samples: {gs['mean']:.2f} ± {gs['std']:.2f}")
            self.table.setItem(row, 0, item0)

            # Col 1  – measured value (placeholder)
            item1 = QTableWidgetItem("—")
            item1.setFlags(item1.flags() & ~Qt.ItemIsEditable)
            item1.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, item1)

            # Col 2  – lo threshold spinbox
            lo_spin = QDoubleSpinBox()
            lo_spin.setDecimals(dec)
            lo_spin.setSingleStep(step)
            lo_spin.setRange(s_lo, s_hi)
            lo_val = self.thresholds[key]["lo"]
            lo_spin.setValue(max(s_lo, min(lo_val, s_hi)))
            lo_spin.valueChanged.connect(self._on_threshold_edited)
            if ttype == "max":
                lo_spin.setEnabled(False)
                lo_spin.setStyleSheet("color:#666; background:#3a3a3a;")
            self.table.setCellWidget(row, 2, lo_spin)
            self.lo_spins[key] = lo_spin

            # Col 3  – hi threshold spinbox
            hi_spin = QDoubleSpinBox()
            hi_spin.setDecimals(dec)
            hi_spin.setSingleStep(step)
            hi_spin.setRange(s_lo, s_hi)
            hi_val = self.thresholds[key]["hi"]
            hi_spin.setValue(max(s_lo, min(hi_val, s_hi)))
            hi_spin.valueChanged.connect(self._on_threshold_edited)
            if ttype == "min":
                hi_spin.setEnabled(False)
                hi_spin.setStyleSheet("color:#666; background:#3a3a3a;")
            self.table.setCellWidget(row, 3, hi_spin)
            self.hi_spins[key] = hi_spin

            # Col 4  – status
            item4 = QTableWidgetItem("—")
            item4.setFlags(item4.flags() & ~Qt.ItemIsEditable)
            item4.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, item4)

            # Col 5  – category (REWORK / REJECT)
            cat_label = "REWORK" if category == "rework" else "REJECT"
            cat_color = QColor(255, 152, 0) if category == "rework" else QColor(183, 28, 28)
            item5 = QTableWidgetItem(cat_label)
            item5.setFlags(item5.flags() & ~Qt.ItemIsEditable)
            item5.setTextAlignment(Qt.AlignCenter)
            item5.setForeground(cat_color)
            self.table.setItem(row, 5, item5)

    # ── Threshold helpers ────────────────────────────────────────────────

    def _sync_thresholds_to_table(self):
        """Push self.thresholds → spinboxes (without triggering signals)."""
        for key in self.lo_spins:
            self.lo_spins[key].blockSignals(True)
            self.hi_spins[key].blockSignals(True)
            lo_val = self.thresholds[key]["lo"]
            hi_val = self.thresholds[key]["hi"]
            sp_lo = self.lo_spins[key]
            sp_hi = self.hi_spins[key]
            sp_lo.setValue(max(sp_lo.minimum(), min(lo_val, sp_lo.maximum())))
            sp_hi.setValue(max(sp_hi.minimum(), min(hi_val, sp_hi.maximum())))
            self.lo_spins[key].blockSignals(False)
            self.hi_spins[key].blockSignals(False)

    def _read_thresholds_from_table(self):
        """Read spinbox values → self.thresholds."""
        for key in self.lo_spins:
            self.thresholds[key] = {
                "lo": self.lo_spins[key].value(),
                "hi": self.hi_spins[key].value(),
            }

    # ── Defect model selection ────────────────────────────────────────────

    def _on_defect_model_changed(self, model_name: str):
        """Called when user changes defect model dropdown."""
        self._defect_model = model_name
        print(f"Defect model changed to: {model_name}")

    # ── YOLO v11 helpers ─────────────────────────────────────────────────

    def _ensure_yolo_detector(self) -> bool:
        """Lazy-load the YOLO v11 model. Returns True if ready."""
        if self._yolo_detector is not None:
            return True
        if not YOLO_CHECKPOINT.exists():
            print(f"⚠ YOLO checkpoint not found: {YOLO_CHECKPOINT}")
            return False
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            self._yolo_detector = UltralyticsYOLO(str(YOLO_CHECKPOINT))
            return True
        except Exception as e:
            print(f"⚠ Failed to load YOLO v11: {e}")
            return False

    def _run_yolo_on_pass(self):
        """Called after geometric verdict is PASS (YOLO v11 mode).
        Crop to 720×720, run YOLO segmentation, display results.
        Returns True if defects found (override to REJECT).
        """
        if self.image is None:
            return False

        if not self._ensure_yolo_detector():
            self.pred_overlay_label.setText("⚠ YOLO v11 model not available")
            self.pred_mask_label.setText("")
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Bin + crop to 720×720
            img_720 = self._bin_crop_720(self.image)

            # Run YOLO inference
            results = self._yolo_detector.predict(
                img_720, conf=0.5, iou=0.45, verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            result = results[0]

            # Extract detections
            boxes = result.boxes
            num_det = len(boxes) if boxes is not None else 0
            has_defect = num_det > 0

            scores = boxes.conf.cpu().numpy().tolist() if num_det > 0 else []
            labels = boxes.cls.cpu().numpy().astype(int).tolist() if num_det > 0 else []

            # Build prediction result dict (compatible with existing code)
            self._pred_result = {
                "num_detections": num_det,
                "has_defect": has_defect,
                "scores": scores,
                "labels": labels,
            }

            # Draw overlay with predictions
            overlay = img_720.copy()
            h, w = img_720.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            if result.masks is not None and num_det > 0:
                masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
                for i in range(num_det):
                    mask_i = cv2.resize(
                        masks_data[i], (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    binary = (mask_i > 0.5).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, binary * 255)

                    # Draw mask overlay
                    color_mask = np.zeros_like(overlay)
                    color_mask[:, :, 2] = binary * 255  # Red channel
                    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

                    # Draw contours
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

                    # Draw score
                    if len(contours) > 0:
                        M = cv2.moments(contours[0])
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(overlay, f"{scores[i]:.0%}",
                                        (cx - 20, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 2)

            self._pred_overlay = overlay
            self._show_cv(overlay, self.pred_overlay_label)

            self._pred_mask = combined_mask
            self._show_cv(combined_mask, self.pred_mask_label)

            return has_defect
        except Exception as e:
            print(f"⚠ YOLO v11 inference failed: {e}")
            import traceback; traceback.print_exc()
            self.pred_overlay_label.setText(f"⚠ YOLO inference error: {e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    # ── Mask R-CNN helpers ───────────────────────────────────────────────

    def _ensure_detector(self) -> bool:
        """Lazy-load the Mask R-CNN model. Returns True if ready."""
        if self._detector is not None:
            return True
        if not MASKRCNN_CHECKPOINT.exists():
            print(f"⚠ Checkpoint not found: {MASKRCNN_CHECKPOINT}")
            return False
        try:
            # Temporarily add maskrcnn dir to path for imports
            maskrcnn_dir = str(WORKSPACE / "maskrcnn")
            if maskrcnn_dir not in sys.path:
                sys.path.insert(0, maskrcnn_dir)
            from inference import OringDefectDetector
            self._detector = OringDefectDetector(
                model_name="combined",
                checkpoint_path=str(MASKRCNN_CHECKPOINT),
                device="cuda" if torch.cuda.is_available() else "cpu",
                score_threshold=0.5,
                mask_threshold=0.5,
            )
            return True
        except Exception as e:
            print(f"⚠ Failed to load Mask R-CNN: {e}")
            return False

    def _bin_crop_720(self, image: np.ndarray) -> np.ndarray:
        """2×2 bin + BG crop + resize/pad to 720×720 (same as training pipeline)."""
        # Import from binning_pipeline
        binning_dir = str(WORKSPACE / "binning_pipeline")
        if binning_dir not in sys.path:
            sys.path.insert(0, binning_dir)
        from bin_and_crop import binning_2x2, crop_to_foreground

        binned = binning_2x2(image)
        cropped, _info = crop_to_foreground(binned, bg_value=20, threshold=30,
                                            pad=10, target_size=720)
        return cropped

    def _run_maskrcnn(self, image_720: np.ndarray) -> Dict:
        """Run Mask R-CNN prediction on a 720×720 image.
        Returns prediction dict with boxes, masks, scores, labels, has_defect.
        """
        return self._detector.predict(image_720)

    def _run_maskrcnn_on_pass(self):
        """Called after geometric verdict is PASS.
        Crop to 720×720, run binary Mask R-CNN, display results.
        Returns True if defects found (override to REJECT).
        """
        if self.image is None:
            return False

        if not self._ensure_detector():
            self.pred_overlay_label.setText("⚠ Model not available")
            self.pred_mask_label.setText("")
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Bin + crop to 720×720
            img_720 = self._bin_crop_720(self.image)

            # Run Mask R-CNN
            pred = self._run_maskrcnn(img_720)
            self._pred_result = pred

            # Draw prediction overlay
            maskrcnn_dir = str(WORKSPACE / "maskrcnn")
            if maskrcnn_dir not in sys.path:
                sys.path.insert(0, maskrcnn_dir)
            from utils import draw_predictions

            self._pred_overlay = draw_predictions(
                img_720, pred["boxes"], pred["masks"],
                pred["scores"], pred["labels"],
                score_threshold=0.5, mask_alpha=0.4)
            self._show_cv(self._pred_overlay, self.pred_overlay_label)

            # Build combined defect mask image
            h, w = img_720.shape[:2]
            if pred["num_detections"] > 0:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for m in pred["masks"]:
                    combined_mask = np.maximum(combined_mask, m * 255)
                self._pred_mask = combined_mask
            else:
                self._pred_mask = np.zeros((h, w), dtype=np.uint8)
            self._show_cv(self._pred_mask, self.pred_mask_label)

            return pred["has_defect"]
        except Exception as e:
            print(f"⚠ Mask R-CNN inference failed: {e}")
            import traceback; traceback.print_exc()
            self.pred_overlay_label.setText(f"⚠ Inference error: {e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    # ── Actions ──────────────────────────────────────────────────────────

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select O-Ring Image", "",
            "Images (*.bmp *.jpg *.jpeg *.png *.tiff);;All Files (*)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{path}")
            return

        self.image = img
        self.overlay_image = None
        self.result = None
        self._show_cv(img, self.img_label)
        self.mask_label.setText("Click  🔍 Analyze  to process")
        self.mask_label.setPixmap(QPixmap())
        self.analyze_btn.setEnabled(True)
        self._clear_results()

        # Build file list for navigation
        folder = str(Path(path).parent)
        exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff'}
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if Path(f).suffix.lower() in exts],
            key=lambda p: Path(p).name.lower()
        )
        self._file_list = files
        try:
            self._file_index = files.index(os.path.normpath(path))
        except ValueError:
            self._file_index = 0
        self._update_nav_buttons()

        h, w = img.shape[:2]
        self.info_label.setText(f"Loaded: {Path(path).name}  ({w}×{h})")
        self.setWindowTitle(f"O-Ring Inspection — {Path(path).name}")

    def _auto_bg(self):
        if self.image is None:
            QMessageBox.information(self, "No Image", "Load an image first.")
            return
        val = auto_bg_value(self.image)
        self.bg_spin.setValue(val)
        self.info_label.setText(
            f"Auto BG: median corner intensity = {val}")

    # ── File navigation ──────────────────────────────────────────────────

    def _update_nav_buttons(self):
        """Enable/disable prev/next buttons and update counter label."""
        n = len(self._file_list)
        idx = self._file_index
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < n - 1)
        if n > 0:
            self.nav_label.setText(f"{idx + 1} / {n}")
        else:
            self.nav_label.setText("")

    def _load_image_at_index(self, index: int):
        """Load an image by index from the current file list."""
        if index < 0 or index >= len(self._file_list):
            return
        path = self._file_list[index]
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{path}")
            return

        self._file_index = index
        self.image = img
        self.overlay_image = None
        self.result = None
        self._show_cv(img, self.img_label)
        self.mask_label.setText("Click  🔍 Analyze  to process")
        self.mask_label.setPixmap(QPixmap())
        self.analyze_btn.setEnabled(True)
        self._clear_results()
        self._update_nav_buttons()

        h, w = img.shape[:2]
        self.info_label.setText(f"Loaded: {Path(path).name}  ({w}×{h})")
        self.setWindowTitle(f"O-Ring Inspection — {Path(path).name}")

    def _load_prev_image(self):
        """Navigate to the previous image in the folder."""
        self._load_image_at_index(self._file_index - 1)

    def _load_next_image(self):
        """Navigate to the next image in the folder."""
        self._load_image_at_index(self._file_index + 1)

    def analyze(self):
        if self.image is None:
            return
        bg = self.bg_spin.value()
        th = self.thresh_spin.value()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.result = measure_oring(self.image, bg, th)
        finally:
            QApplication.restoreOverrideCursor()

        if self.result is None:
            QMessageBox.warning(
                self, "Detection Failed",
                "Could not detect inner/outer ring contours.\n"
                "Try adjusting BG value or threshold.")
            return

        # Auto-detect resolution scale and normalize to reference
        h, w = self.image.shape[:2]
        self._resolution_scale = compute_resolution_scale(w, h)
        self.result_normed = normalize_measurements(self.result, self._resolution_scale)

        # Display overlay
        self.overlay_image = draw_overlay(self.image, self.result)
        self._show_cv(self.overlay_image, self.img_label)

        # Display mask with contour outline
        mask_vis = cv2.cvtColor(self.result["mask"], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(
            mask_vis, [self.result["outer_contour"]], -1, (0, 255, 0), 2)
        cv2.drawContours(
            mask_vis, [self.result["inner_contour"]], -1, (0, 0, 255), 2)
        self._show_cv(mask_vis, self.mask_label)

        # Evaluate against thresholds
        self._evaluate()

        # Show extra informational metrics (normalized values)
        r = self.result_normed
        scale_str = f"scale={self._resolution_scale:.2f}" if abs(self._resolution_scale - 1.0) > 0.01 else ""
        self.info_label.setText(
            f"center_dx={r['center_dx']:+.1f}  center_dy={r['center_dy']:+.1f}  "
            f"max_thick={r['max_thickness']:.1f}  "
            f"thick_std={r['thickness_std']:.1f}  "
            f"circ_o={r['circularity_outer']:.4f}  "
            f"edge_clr={r['edge_clearance']:.0f}  "
            f"area={self.result['annular_area']:,}  {scale_str}".rstrip())

    def _evaluate(self):
        """Compare each metric against thresholds.

        Verdict priority:
            1. REWORK  — any "rework"-category metric fails (shape — fixable)
            2. REJECT  — any "reject"-category metric fails (thickness / size)
            3. PASS    — everything within tolerance

        Uses resolution-normalized measurements so that thresholds
        (calibrated at REFERENCE_RESOLUTION) apply at any input size.
        """
        if self.result is None:
            return

        # Use normalized measurements for threshold comparison
        normed = getattr(self, "result_normed", self.result)

        self._read_thresholds_from_table()
        rework_fails = []
        reject_fails = []

        for row, (key, name, _unit, ttype, dec, *_, category) in enumerate(METRIC_DEFS):
            val = normed.get(key)
            if val is None:
                continue

            lo = self.thresholds[key]["lo"]
            hi = self.thresholds[key]["hi"]

            # Update measured value
            fmt = f"{{:.{dec}f}}"
            val_item = self.table.item(row, 1)
            val_item.setText(fmt.format(val))

            # Check pass / fail
            passed = True
            # outer_radius: only flag if TOO LARGE (ignore min — smaller outer is not rework)
            # inner_radius: only flag if TOO SMALL (ignore max — larger inner is not rework)
            if key == "outer_radius":
                if val > hi:
                    passed = False
            elif key == "inner_radius":
                if val < lo:
                    passed = False
            else:
                if ttype in ("range", "min") and val < lo:
                    passed = False
                if ttype in ("range", "max") and val > hi:
                    passed = False

            status_item = self.table.item(row, 4)
            if passed:
                status_item.setText("✅ PASS")
                status_item.setForeground(QColor(76, 175, 80))
                val_item.setForeground(QColor(200, 200, 200))
            else:
                status_item.setText("❌ FAIL")
                status_item.setForeground(QColor(244, 67, 54))
                val_item.setForeground(QColor(244, 67, 54))
                if category == "rework":
                    rework_fails.append(name)
                else:
                    reject_fails.append(name)

        # ── Overall verdict ──────────────────────────────────────────────
        # Check REWORK first (fixable), then REJECT (unfixable)
        # Clear prediction panels (model only runs on PASS)
        self.pred_overlay_label.setText(f"{self._defect_model} prediction (PASS images)")
        self.pred_overlay_label.setPixmap(QPixmap())
        self.pred_mask_label.setText("Predicted defect mask")
        self.pred_mask_label.setPixmap(QPixmap())
        self._pred_overlay = None
        self._pred_mask = None
        self._pred_result = None
        if rework_fails:
            # REWORK — shape issues fixable by trimming
            self.verdict_label.setText("🔧  REWORK")
            self.verdict_label.setStyleSheet("color: white;")
            self.verdict_frame.setStyleSheet(
                "background:#C62828; border-radius:12px; padding:8px;")
            detail_parts = []
            detail_parts.append(
                f"Shape ({len(rework_fails)}): " + ", ".join(rework_fails))
            if reject_fails:
                detail_parts.append(
                    f"Also failing ({len(reject_fails)}): " + ", ".join(reject_fails))
            self.verdict_detail.setText(" | ".join(detail_parts))
            self.verdict_detail.setStyleSheet("color:#FFCDD2;")

        elif reject_fails:
            # REJECT — thickness / concentricity / area issues (unfixable)
            self.verdict_label.setText("⛔  REJECT")
            self.verdict_label.setStyleSheet("color: white;")
            self.verdict_frame.setStyleSheet(
                "background:#4A148C; border-radius:12px; padding:8px;")
            self.verdict_detail.setText(
                f"{len(reject_fails)} metric(s): " + ", ".join(reject_fails))
            self.verdict_detail.setStyleSheet("color:#CE93D8;")

        else:
            # Geometric verdict = PASS — run defect model for confirmation
            model_name = self._defect_model
            self.pred_overlay_label.setText(f"Running {model_name}…")
            self.pred_mask_label.setText("")
            QApplication.processEvents()  # update UI before blocking inference

            if model_name == "YOLO v11":
                has_defect = self._run_yolo_on_pass()
            else:
                has_defect = self._run_maskrcnn_on_pass()

            if has_defect:
                # Override: model found defects on a geometrically-passing image
                n_det = self._pred_result["num_detections"]
                top_score = max(self._pred_result["scores"]) if n_det > 0 else 0
                self.verdict_label.setText("⛔  REJECT (AI)")
                self.verdict_label.setStyleSheet("color: white;")
                self.verdict_frame.setStyleSheet(
                    "background:#4A148C; border-radius:12px; padding:8px;")
                self.verdict_detail.setText(
                    f"Geometry OK, but {model_name} detected {n_det} defect(s)  "
                    f"(top score {top_score:.0%})")
                self.verdict_detail.setStyleSheet("color:#CE93D8;")
            else:
                # Confirmed PASS
                self.verdict_label.setText("✅  PASS")
                self.verdict_label.setStyleSheet("color: white;")
                self.verdict_frame.setStyleSheet(
                    "background:#2E7D32; border-radius:12px; padding:8px;")
                self.verdict_detail.setText(
                    f"All metrics within tolerance • {model_name}: no defects")
                self.verdict_detail.setStyleSheet("color:#C8E6C9;")

    def _clear_results(self):
        for row in range(self.table.rowCount()):
            self.table.item(row, 1).setText("—")
            self.table.item(row, 1).setForeground(QColor(170, 170, 170))
            self.table.item(row, 4).setText("—")
            self.table.item(row, 4).setForeground(QColor(170, 170, 170))
        self.verdict_label.setText("AWAITING")
        self.verdict_label.setStyleSheet("color:#ccc;")
        self.verdict_frame.setStyleSheet(
            "background:#555; border-radius:12px; padding:8px;")
        self.verdict_detail.setText("Load an image and click Analyze")
        self.verdict_detail.setStyleSheet("color:#bbb;")
        # Clear prediction panels
        self._pred_overlay = None
        self._pred_mask = None
        self._pred_result = None
        self.pred_overlay_label.setText(f"{self._defect_model} prediction (PASS images)")
        self.pred_overlay_label.setPixmap(QPixmap())
        self.pred_mask_label.setText("Predicted defect mask")
        self.pred_mask_label.setPixmap(QPixmap())

    def _on_threshold_edited(self, _=None):
        if self.result is not None:
            self._evaluate()

    def _load_best_thresholds(self) -> Dict[str, Dict]:
        """Load tuned JSON if available, else fall back to σ-based.

        After loading, widen all REJECT-category thresholds by 10 %.
        """
        tuned_path = TUNED_JSON.get(self.current_model)
        if tuned_path:
            tuned = load_tuned_thresholds(tuned_path)
            if tuned:
                # Fill any missing metrics from σ-based
                sigma_t = compute_thresholds(self.good_stats, self.sigma)
                for key in sigma_t:
                    if key not in tuned:
                        tuned[key] = sigma_t[key]
                thresholds = tuned
            else:
                thresholds = compute_thresholds(self.good_stats, self.sigma)
        else:
            thresholds = compute_thresholds(self.good_stats, self.sigma)

        # Widen REJECT thresholds by 10 %
        reject_keys = {m[0] for m in METRIC_DEFS if m[8] == "reject"}
        for key in reject_keys:
            if key not in thresholds:
                continue
            lo = thresholds[key]["lo"]
            hi = thresholds[key]["hi"]
            # lo gets 10 % lower, hi gets 10 % higher
            if lo > 0:
                thresholds[key]["lo"] = round(lo * 0.9, 4)
            if hi < 9999:
                thresholds[key]["hi"] = round(hi * 1.1, 4)

        return thresholds

    def _recompute_thresholds(self, sigma):
        self.sigma = sigma
        self.thresholds = self._load_best_thresholds()
        self._sync_thresholds_to_table()
        if self.result is not None:
            self._evaluate()

    def _on_model_changed(self, model_name: str):
        self.current_model = model_name
        csv_path = MODEL_CSV[model_name]
        self.good_stats = load_good_stats(csv_path)
        self.thresholds = self._load_best_thresholds()
        self._sync_thresholds_to_table()
        if self.result is not None:
            self._evaluate()
        tuned_path = TUNED_JSON.get(model_name)
        if tuned_path and tuned_path.exists():
            self.info_label.setText(
                f"✓ Switched to {model_name} — tuned thresholds from {tuned_path.name}")
        elif self.good_stats:
            self.info_label.setText(
                f"✓ Switched to {model_name} — σ-based from {csv_path.name}  "
                f"(σ = {self.sigma})")
        else:
            self.info_label.setText(
                f"⚠ No threshold data for {model_name} — using defaults")

    def _reset_thresholds(self):
        self.thresholds = self._load_best_thresholds()
        self._sync_thresholds_to_table()
        if self.result is not None:
            self._evaluate()

    # ── Camera helpers (Hikrobot) ─────────────────────────────────────────

    def _init_camera(self) -> bool:
        """Enumerate Hikrobot devices, create handle, open the first camera."""
        if not HIKROBOT_AVAILABLE:
            QMessageBox.warning(
                self, "Camera Error",
                "Hikrobot SDK not available.\n"
                "Install the MVS SDK and ensure MvImport is on the Python path.")
            return False
        if self._camera is not None:
            return True  # already initialised

        try:
            deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0 or deviceList.nDeviceNum == 0:
                QMessageBox.warning(self, "Camera Error", "No Hikrobot cameras found!")
                return False

            self._camera = MvCamera()
            stDevInfo = cast(
                deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self._camera.MV_CC_CreateHandle(stDevInfo)
            if ret != 0:
                self._camera = None
                QMessageBox.warning(
                    self, "Camera Error", f"CreateHandle failed (0x{ret:08X})")
                return False

            ret = self._camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self._camera.MV_CC_DestroyHandle()
                self._camera = None
                QMessageBox.warning(
                    self, "Camera Error", f"OpenDevice failed (0x{ret:08X})")
                return False

            # Free-run mode (no hardware trigger)
            self._camera.MV_CC_SetEnumValue("TriggerMode", 0)
            return True

        except Exception as e:
            self._camera = None
            QMessageBox.warning(
                self, "Camera Error", f"Camera initialisation failed:\n{e}")
            return False

    def _toggle_stream(self):
        """Toggle camera live stream on / off."""
        if self._camera_streaming:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self):
        """Open camera (if needed), start grabbing, start preview timer."""
        if not self._init_camera():
            return
        if self._camera_streaming:
            return

        ret = self._camera.MV_CC_StartGrabbing()
        if ret != 0:
            QMessageBox.warning(
                self, "Camera Error", f"StartGrabbing failed (0x{ret:08X})")
            return

        self._camera_streaming = True
        self.stream_btn.setText("⏹ Stop Stream")
        self.capture_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)

        if self._stream_timer is None:
            self._stream_timer = QTimer(self)
            self._stream_timer.timeout.connect(self._update_stream_preview)
        self._stream_timer.start(66)   # ~15 fps preview
        self.info_label.setText("Camera streaming — press Capture to grab a frame")

    def _stop_stream(self):
        """Stop preview timer and camera grabbing."""
        if self._stream_timer is not None:
            self._stream_timer.stop()
        if self._camera is not None and self._camera_streaming:
            self._camera.MV_CC_StopGrabbing()
        self._camera_streaming = False
        self.stream_btn.setText("📷 Start Stream")
        self.capture_btn.setEnabled(False)
        self.load_btn.setEnabled(True)

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from the camera.

        Returns a BGR numpy array or None on failure.
        """
        if self._camera is None or not self._camera_streaming:
            return None

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        ret = self._camera.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            return None

        try:
            buf_len = stOutFrame.stFrameInfo.nFrameLen
            pData = (c_ubyte * buf_len).from_address(stOutFrame.pBufAddr)
            raw = np.frombuffer(pData, dtype=np.uint8).copy()

            h = stOutFrame.stFrameInfo.nHeight
            w = stOutFrame.stFrameInfo.nWidth
            px = stOutFrame.stFrameInfo.enPixelType

            # ── Pixel-format conversion to BGR ────────────────────────────
            if px == 0x01080001:          # Mono8
                frame = raw.reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif px == 0x01080009:        # BayerRG8
                frame = raw.reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            elif px == 0x0108000A:        # BayerGB8
                frame = raw.reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
            elif px == 0x0108000B:        # BayerGR8
                frame = raw.reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
            elif px == 0x01080008:        # BayerBG8
                frame = raw.reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            elif px == 0x02180014:        # RGB8
                frame = raw.reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif px == 0x02180015:        # BGR8
                frame = raw.reshape(h, w, 3)
            else:
                # Fallback — treat as single-channel
                try:
                    frame = raw.reshape(h, w)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                except ValueError:
                    frame = raw.reshape(h, w, -1)[:, :, :3].copy()
            return frame
        finally:
            self._camera.MV_CC_FreeImageBuffer(stOutFrame)

    def _update_stream_preview(self):
        """QTimer callback — grab a frame and show it in the preview."""
        frame = self._grab_frame()
        if frame is not None:
            self._latest_frame = frame
            self._show_cv(frame, self.img_label)

    def _capture_frame(self):
        """Freeze the current live frame and load it for analysis."""
        if not self._camera_streaming:
            QMessageBox.information(
                self, "No Stream", "Start the camera stream first.")
            return

        # Stop the stream so the user can work with a still image
        self._stop_stream()

        frame = self._latest_frame
        if frame is None:
            QMessageBox.warning(
                self, "Capture Error", "No frame available from the camera.")
            return

        self.image = frame
        self.overlay_image = None
        self.result = None
        self._show_cv(frame, self.img_label)
        self.mask_label.setText("Click  🔍 Analyze  to process")
        self.mask_label.setPixmap(QPixmap())
        self.analyze_btn.setEnabled(True)
        self._clear_results()

        # Clear file navigation (this is from camera, not a file)
        self._file_list = []
        self._file_index = -1
        self._update_nav_buttons()

        h, w = frame.shape[:2]
        self.info_label.setText(f"Captured from camera  ({w}×{h})")
        self.setWindowTitle("O-Ring Inspection — Camera Capture")

    def _release_camera(self):
        """Release all camera resources."""
        self._stop_stream()
        if self._camera is not None:
            try:
                self._camera.MV_CC_CloseDevice()
                self._camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._camera = None

    def closeEvent(self, event):
        """Ensure camera is released when the window is closed."""
        self._release_camera()
        super().closeEvent(event)

    # ── Display helpers ──────────────────────────────────────────────────

    def _show_cv(self, cv_img: np.ndarray, label: QLabel):
        if cv_img is None:
            return
        if len(cv_img.shape) == 2:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.overlay_image is not None:
            self._show_cv(self.overlay_image, self.img_label)
        elif self.image is not None:
            self._show_cv(self.image, self.img_label)
        if self.result is not None and "mask" in self.result:
            mask_vis = cv2.cvtColor(
                self.result["mask"], cv2.COLOR_GRAY2BGR)
            cv2.drawContours(
                mask_vis, [self.result["outer_contour"]], -1, (0, 255, 0), 2)
            cv2.drawContours(
                mask_vis, [self.result["inner_contour"]], -1, (0, 0, 255), 2)
            self._show_cv(mask_vis, self.mask_label)
        if self._pred_overlay is not None:
            self._show_cv(self._pred_overlay, self.pred_overlay_label)
        if self._pred_mask is not None:
            self._show_cv(self._pred_mask, self.pred_mask_label)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)

    # Dark palette
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(53, 53, 53))
    pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            QColor(35, 35, 35))
    pal.setColor(QPalette.AlternateBase,   QColor(53, 53, 53))
    pal.setColor(QPalette.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.Button,          QColor(53, 53, 53))
    pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       QColor(42, 130, 218))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ToolTipBase,     QColor(42, 42, 42))
    pal.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
    app.setPalette(pal)

    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #555;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QSpinBox, QDoubleSpinBox {
            background: #3a3a3a;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
            color: #ddd;
        }
        QPushButton {
            border: 1px solid #666;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background: #4a4a4a;
        }
        QTableWidget {
            gridline-color: #444;
        }
        QHeaderView::section {
            background: #3a3a3a;
            border: 1px solid #555;
            padding: 4px;
            font-weight: bold;
        }
    """)

    window = InspectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
