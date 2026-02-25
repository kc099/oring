"""
O-Ring Inspection GUI — Pass / Rework / Reject Classification  (Dear PyGui)

Single-image display (defect overlay shown only on FAIL).
Mask R-CNN model pre-loaded at startup with a status indicator.
Model 1 / Model 2 toggle buttons.
Hikrobot camera integration (optional).
Cycle time displayed after each analysis.

Metrics (6 total):
  REWORK:  outer_radius, inner_radius, circularity_outer/inner
  REJECT:  center_dist, eccentricity_pct

Usage:
    python rework/inspection_gui.py
"""

from __future__ import annotations

import math
import os
import sys
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import dearpygui.dearpygui as dpg

# ── Hikrobot Camera SDK (optional) ───────────────────────────────────────
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

MASKRCNN_CHECKPOINT = (
    WORKSPACE / "maskrcnn" / "dataset" / "combined" / "checkpoints" / "best_model.pth"
)

DEFAULT_BG_VALUE = 20

REFERENCE_RESOLUTION = (2448, 2048)

METRIC_SCALE_TYPE = {
    "outer_radius":      "linear",
    "inner_radius":      "linear",
    "center_dist":       "linear",
    "circularity_outer": "none",
    "circularity_inner": "none",
    "eccentricity_pct":  "none",
}

MODEL_CSV = {
    "Model 1": SCRIPT_DIR / "model1good_measurements.csv",
    "Model 2": SCRIPT_DIR / "good_measurements.csv",
}

TUNED_JSON = {
    "Model 1": SCRIPT_DIR / "model1_tuned_thresholds.json",
    "Model 2": SCRIPT_DIR / "model2_tuned_thresholds.json",
}

DEFAULT_MODEL = "Model 2"

# (key, display_name, unit, thresh_type, decimals, spin_step, spin_lo, spin_hi, verdict_category)
METRIC_DEFS = [
    ("outer_radius",      "Outer Radius",      "px", "range", 1, 1.0,   400, 1200, "rework"),
    ("inner_radius",      "Inner Radius",      "px", "range", 1, 1.0,   200, 800,  "rework"),
    ("circularity_outer", "Outer Circularity",  "",   "min",   3, 0.005,   0, 1,    "rework"),
    ("circularity_inner", "Inner Circularity",  "",   "min",   3, 0.005,   0, 1,    "rework"),
    ("center_dist",       "Center Distance",   "px",  "max",   1, 1.0,     0, 500,  "reject"),
    ("eccentricity_pct",  "Eccentricity",       "%",  "max",   2, 0.1,     0, 50,   "reject"),
]

_METRIC_KEYS = {m[0] for m in METRIC_DEFS}

DEFAULT_THRESHOLDS = {
    "outer_radius":      {"lo": 650.0, "hi": 680.0},
    "inner_radius":      {"lo": 375.0, "hi": 400.0},
    "center_dist":       {"lo": 0.0,   "hi": 35.0},
    "eccentricity_pct":  {"lo": 0.0,   "hi": 6.0},
    "circularity_outer": {"lo": 0.75,  "hi": 1.0},
    "circularity_inner": {"lo": 0.75,  "hi": 1.0},
}

IMG_W, IMG_H = 800, 700   # display texture size


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics / threshold helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_resolution_scale(img_w: int, img_h: int) -> float:
    ref = max(REFERENCE_RESOLUTION)
    cur = max(img_w, img_h)
    return cur / ref


def normalize_measurements(result: Dict, scale: float) -> Dict:
    if abs(scale - 1.0) < 1e-6:
        return result
    normed = dict(result)
    for key, stype in METRIC_SCALE_TYPE.items():
        if key not in normed:
            continue
        if stype == "linear":
            normed[key] = normed[key] / scale
    return normed


def load_good_stats(csv_path: Path) -> Optional[Dict]:
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
    for key in _METRIC_KEYS:
        if key in rows[0]:
            vals = [float(r[key]) for r in rows if r.get(key)]
            if vals:
                m, s = _ms(vals)
                stats[key] = {"mean": m, "std": s}

    if "eccentricity_pct" not in stats and "center_dist" in rows[0] and "mean_radius" in rows[0]:
        ecc = [float(r["center_dist"]) / float(r["mean_radius"]) * 100 for r in rows]
        m, s = _ms(ecc)
        stats["eccentricity_pct"] = {"mean": m, "std": s}

    return stats


def load_tuned_thresholds(json_path: Path) -> Optional[Dict[str, Dict]]:
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    thresholds = {}
    for key in data:
        if key in _METRIC_KEYS:
            thresholds[key] = {"lo": data[key]["lo"], "hi": data[key]["hi"]}
    return thresholds


def compute_thresholds(stats: Optional[Dict], sigma: float = 2.5) -> Dict[str, Dict]:
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


def load_best_thresholds(model_name: str, good_stats, sigma: float) -> Dict[str, Dict]:
    """Load tuned JSON if available, else sigma-based. Widen REJECT by 10%."""
    tuned_path = TUNED_JSON.get(model_name)
    if tuned_path:
        tuned = load_tuned_thresholds(tuned_path)
        if tuned:
            sigma_t = compute_thresholds(good_stats, sigma)
            for key in sigma_t:
                if key not in tuned:
                    tuned[key] = sigma_t[key]
            thresholds = tuned
        else:
            thresholds = compute_thresholds(good_stats, sigma)
    else:
        thresholds = compute_thresholds(good_stats, sigma)

    reject_keys = {m[0] for m in METRIC_DEFS if m[8] == "reject"}
    for key in reject_keys:
        if key not in thresholds:
            continue
        lo = thresholds[key]["lo"]
        hi = thresholds[key]["hi"]
        if lo > 0:
            thresholds[key]["lo"] = round(lo * 0.9, 4)
        if hi < 9999:
            thresholds[key]["hi"] = round(hi * 1.1, 4)

    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
#  Image processing  (only the 8 metrics)
# ═══════════════════════════════════════════════════════════════════════════

def _largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


def build_mask(image: np.ndarray, bg_value: int = 20,
               threshold: int = 30) -> np.ndarray:
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
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = res
    radius = math.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    return float(cx), float(cy), float(radius)


def auto_bg_value(image: np.ndarray, margin: int = 80) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = np.concatenate([
        gray[:margin, :margin].ravel(),
        gray[:margin, w - margin:].ravel(),
        gray[h - margin:, :margin].ravel(),
        gray[h - margin:, w - margin:].ravel(),
    ])
    return int(np.median(corners))


def measure_oring(image: np.ndarray, bg_value: int = 20,
                  threshold: int = 30) -> Optional[Dict]:
    mask = build_mask(image, bg_value, threshold)
    outer, inner = find_contours(mask)
    if outer is None or inner is None:
        return None

    ox, oy, orad = fit_circle_lsq(outer)
    ix, iy, irad = fit_circle_lsq(inner)
    cdist = math.hypot(ox - ix, oy - iy)
    mrad = (orad + irad) / 2.0

    o_area = cv2.contourArea(outer)
    o_peri = cv2.arcLength(outer, True)
    circ_o = (4.0 * math.pi * o_area / (o_peri ** 2)) if o_peri > 0 else 0.0

    i_area = cv2.contourArea(inner)
    i_peri = cv2.arcLength(inner, True)
    circ_i = (4.0 * math.pi * i_area / (i_peri ** 2)) if i_peri > 0 else 0.0

    return {
        "outer_radius":      float(orad),
        "inner_radius":      float(irad),
        "center_dist":       cdist,
        "eccentricity_pct":  (cdist / mrad * 100) if mrad > 0 else 0,
        "circularity_outer": circ_o,
        "circularity_inner": circ_i,
        "mask":              mask,
        "outer_contour":     outer,
        "inner_contour":     inner,
        "outer_center":      (float(ox), float(oy)),
        "inner_center":      (float(ix), float(iy)),
    }


def draw_overlay(image: np.ndarray, result: Dict) -> np.ndarray:
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

    return vis


# ── Eagerly import bin_and_crop (cached at module load) ──────────────────
_binning_dir = str(WORKSPACE / "binning_pipeline")
if _binning_dir not in sys.path:
    sys.path.insert(0, _binning_dir)
from bin_and_crop import binning_2x2, crop_to_foreground   # noqa: E402


def bin_crop_720(image: np.ndarray) -> np.ndarray:
    """2x2 bin + BG crop + resize/pad to 720x720."""
    binned = binning_2x2(image)
    cropped, _info = crop_to_foreground(
        binned, bg_value=20, threshold=30, pad=10, target_size=720)
    return cropped


# ═══════════════════════════════════════════════════════════════════════════
#  Application state
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    """Mutable application state — plain object, no signals."""
    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.overlay_image: Optional[np.ndarray] = None
        self.result: Optional[Dict] = None
        self.result_normed: Optional[Dict] = None
        self.resolution_scale: float = 1.0
        self.verdict: str = "AWAITING"

        # DL models (preloaded)
        self.detector = None           # Mask R-CNN
        self.maskrcnn_ready = False
        self.maskrcnn_status = "loading..."

        # Current model / thresholds
        self.current_model: str = DEFAULT_MODEL
        self.good_stats = load_good_stats(MODEL_CSV[DEFAULT_MODEL])
        self.sigma: float = 2.5
        self.thresholds = load_best_thresholds(DEFAULT_MODEL, self.good_stats, 2.5)

        # Camera
        self.camera = None
        self.camera_streaming = False
        self.latest_frame: Optional[np.ndarray] = None

        # File navigation
        self.file_list: list = []
        self.file_index: int = -1

        # DL prediction results
        self.pred_overlay: Optional[np.ndarray] = None
        self.pred_result: Optional[Dict] = None


# ═══════════════════════════════════════════════════════════════════════════
#  DL model preloading (runs in background threads at startup)
# ═══════════════════════════════════════════════════════════════════════════

def preload_models(state: AppState):
    """Spawn daemon thread to load Mask R-CNN (FP32) with GPU warmup."""

    def _load_maskrcnn():
        try:
            if not MASKRCNN_CHECKPOINT.exists():
                state.maskrcnn_status = "checkpoint not found"
                return
            maskrcnn_dir = str(WORKSPACE / "maskrcnn")
            if maskrcnn_dir not in sys.path:
                sys.path.insert(0, maskrcnn_dir)
            from inference import OringDefectDetector

            use_cuda = torch.cuda.is_available()
            device_str = "cuda" if use_cuda else "cpu"

            state.maskrcnn_status = "loading weights..."
            state.detector = OringDefectDetector(
                model_name="combined",
                checkpoint_path=str(MASKRCNN_CHECKPOINT),
                device=device_str,
                score_threshold=0.5,
                mask_threshold=0.5,
            )

            # ── GPU warmup (3 dummy forward passes for consistent timing) ──
            state.maskrcnn_status = "warming up GPU..."
            dummy = torch.zeros(3, 720, 720, dtype=torch.float32)
            if use_cuda:
                dummy = dummy.cuda()
            for _ in range(3):
                with torch.no_grad():
                    state.detector.model([dummy])
            if use_cuda:
                torch.cuda.synchronize()
            del dummy

            state.maskrcnn_ready = True
            state.maskrcnn_status = "ready"
        except Exception as e:
            state.maskrcnn_status = f"error: {e}"

    t1 = threading.Thread(target=_load_maskrcnn, daemon=True)
    t1.start()


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers for Dear PyGui textures
# ═══════════════════════════════════════════════════════════════════════════

def cv_to_dpg(cv_img: np.ndarray, target_w: int = IMG_W,
              target_h: int = IMG_H) -> np.ndarray:
    """Convert BGR cv2 image → RGBA float32 flat array for DPG raw texture."""
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    h, w = cv_img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) / 255.0).flatten()


# ═══════════════════════════════════════════════════════════════════════════
#  Build GUI
# ═══════════════════════════════════════════════════════════════════════════

def build_gui(state: AppState, default_font=None, large_font=None,
              heading_font=None, status_font=None):
    """Create the entire Dear PyGui UI. Returns the _frame_update callback."""

    # ── Themes ────────────────────────────────────────────────────────────
    with dpg.theme() as theme_btn_active:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (33, 150, 243))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 165, 245))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 118, 210))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (100, 200, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)

    with dpg.theme() as theme_btn_inactive:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 65, 78))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (160, 160, 165))

    with dpg.theme() as theme_btn_disabled:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))

    with dpg.theme() as theme_pass:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (27, 94, 32))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)

    with dpg.theme() as theme_rework:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (230, 126, 34))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)

    with dpg.theme() as theme_reject:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (211, 47, 47))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)

    with dpg.theme() as theme_awaiting:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (55, 55, 64))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)

    # Green-tinted theme for "ready" status indicators
    with dpg.theme() as theme_status_ready:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 220, 100))

    with dpg.theme() as theme_status_loading:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 193, 7))

    with dpg.theme() as theme_status_error:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (244, 67, 54))

    # Accent button theme (for Load / Analyze)
    with dpg.theme() as theme_btn_green:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (56, 142, 60))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (76, 175, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (46, 125, 50))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))

    with dpg.theme() as theme_btn_orange:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (230, 126, 34))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (243, 156, 18))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (211, 84, 0))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))

    themes = {
        "active": theme_btn_active,
        "inactive": theme_btn_inactive,
        "disabled": theme_btn_disabled,
        "pass": theme_pass,
        "rework": theme_rework,
        "reject": theme_reject,
        "awaiting": theme_awaiting,
        "status_ready": theme_status_ready,
        "status_loading": theme_status_loading,
        "status_error": theme_status_error,
        "btn_green": theme_btn_green,
        "btn_orange": theme_btn_orange,
    }

    # ── Texture registry ─────────────────────────────────────────────────
    blank = [0.0] * (IMG_W * IMG_H * 4)
    with dpg.texture_registry():
        dpg.add_raw_texture(IMG_W, IMG_H, blank, tag="tex_main",
                            format=dpg.mvFormat_Float_rgba)

    # ══════════════════════════════════════════════════════════════════════
    #  Define ALL helper / callback functions BEFORE creating widgets
    # ══════════════════════════════════════════════════════════════════════

    # -- Display helpers --------------------------------------------------

    def _show_image(cv_img: np.ndarray):
        """Push a cv2 BGR image to the DPG texture."""
        data = cv_to_dpg(cv_img)
        dpg.set_value("tex_main", data)

    # -- Model button helpers ---------------------------------------------

    def _update_model_btns():
        m = state.current_model
        dpg.bind_item_theme("btn_model1",
                            themes["active"] if m == "Model 1" else themes["inactive"])
        dpg.bind_item_theme("btn_model2",
                            themes["active"] if m == "Model 2" else themes["inactive"])

    # -- Threshold helpers ------------------------------------------------

    def _sync_thresholds_to_ui():
        """Push state.thresholds → DPG input widgets."""
        for key, *_ in METRIC_DEFS:
            lo_tag = f"lo_{key}"
            hi_tag = f"hi_{key}"
            if dpg.does_item_exist(lo_tag):
                dpg.set_value(lo_tag, state.thresholds[key]["lo"])
            if dpg.does_item_exist(hi_tag):
                dpg.set_value(hi_tag, state.thresholds[key]["hi"])

    def _read_thresholds_from_ui():
        """Read DPG input widget values → state.thresholds."""
        for key, *_ in METRIC_DEFS:
            lo_tag = f"lo_{key}"
            hi_tag = f"hi_{key}"
            if dpg.does_item_exist(lo_tag):
                state.thresholds[key]["lo"] = dpg.get_value(lo_tag)
            if dpg.does_item_exist(hi_tag):
                state.thresholds[key]["hi"] = dpg.get_value(hi_tag)

    # -- Analyze / navigation state helpers --------------------------------

    def _update_analyze_btn():
        has_img = state.image is not None
        dpg.bind_item_theme("btn_analyze",
                            themes["active"] if has_img else themes["disabled"])

    def _update_nav_btns():
        n = len(state.file_list)
        idx = state.file_index
        dpg.configure_item("btn_prev", enabled=(idx > 0))
        dpg.configure_item("btn_next", enabled=(idx < n - 1))
        dpg.set_value("nav_label", f"{idx + 1} / {n}" if n > 0 else "")

    # -- Clear results ----------------------------------------------------

    def _clear_results():
        for key, name, unit, *_ in METRIC_DEFS:
            dpg.set_value(f"val_{key}", "-")
            dpg.set_value(f"status_{key}", "-")
        dpg.set_value("verdict_text", "AWAITING")
        dpg.bind_item_theme("verdict_box", themes["awaiting"])
        dpg.set_value("cycle_time_text", "")
        state.verdict = "AWAITING"
        state.pred_overlay = None
        state.pred_result = None

    # -- DL inference helpers (Mask R-CNN) ---------------------------------

    def _predict_maskrcnn(img_720: np.ndarray, det_thresh: float) -> Dict:
        """Run Mask R-CNN inference (FP32) with CUDA sync for consistent timing."""
        img_rgb = cv2.cvtColor(img_720, cv2.COLOR_BGR2RGB)
        t = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        t = t.to(state.detector.device)

        with torch.no_grad():
            outputs = state.detector.model([t])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        keep = outputs["scores"] >= det_thresh
        scores = outputs["scores"][keep].cpu().numpy()
        if "masks" in outputs and keep.sum() > 0:
            masks = (outputs["masks"][keep].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        else:
            masks = np.zeros((0, img_720.shape[0], img_720.shape[1]), dtype=np.uint8)
        return {
            "masks": masks,
            "scores": scores,
            "num_detections": len(scores),
            "has_defect": len(scores) > 0,
        }

    def _run_maskrcnn_on_pass(precomputed_720: Optional[np.ndarray] = None) -> bool:
        if state.image is None or not state.maskrcnn_ready:
            return False
        try:
            det_thresh = dpg.get_value("spin_det_thresh")
            img_720 = precomputed_720 if precomputed_720 is not None else bin_crop_720(state.image)
            pred = _predict_maskrcnn(img_720, det_thresh)
            state.pred_result = pred

            # Draw mask-only overlay (no bounding boxes)
            overlay = img_720.copy()
            masks = pred["masks"]
            scores = pred["scores"]
            for i in range(len(scores)):
                if masks is not None and i < len(masks):
                    mask_i = masks[i]
                    if mask_i.shape != img_720.shape[:2]:
                        mask_i = cv2.resize(mask_i.astype(np.uint8),
                                            (img_720.shape[1], img_720.shape[0]))
                    color_overlay = overlay.copy()
                    color_overlay[mask_i > 0.5] = (0, 0, 255)
                    overlay = cv2.addWeighted(overlay, 0.6, color_overlay, 0.4, 0)
                    # Draw mask contour
                    contours, _ = cv2.findContours(
                        (mask_i > 0.5).astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                    # Confidence label at mask centroid
                    if len(contours) > 0:
                        M = cv2.moments(contours[0])
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(overlay, f"{scores[i]:.0%}",
                                        (cx - 20, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255), 2)
            state.pred_overlay = overlay
            return pred["has_defect"]
        except Exception as e:
            print(f"Mask R-CNN inference failed: {e}")
            return False

    # -- Evaluate metrics -------------------------------------------------

    def _evaluate():
        """Compare each metric against thresholds and issue verdict."""
        if state.result is None:
            return

        normed = state.result_normed if state.result_normed else state.result
        _read_thresholds_from_ui()
        rework_fails = []
        reject_fails = []

        for key, name, _unit, ttype, dec, *rest in METRIC_DEFS:
            category = rest[-1]  # last element is verdict_category
            val = normed.get(key)
            if val is None:
                continue

            lo = state.thresholds[key]["lo"]
            hi = state.thresholds[key]["hi"]

            fmt = f"{{:.{dec}f}}"
            dpg.set_value(f"val_{key}", fmt.format(val))

            passed = True
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

            if passed:
                dpg.set_value(f"status_{key}", "PASS")
                dpg.configure_item(f"status_{key}", color=(76, 175, 80))
                dpg.configure_item(f"val_{key}", color=(200, 200, 200))
            else:
                dpg.set_value(f"status_{key}", "FAIL")
                dpg.configure_item(f"status_{key}", color=(244, 67, 54))
                dpg.configure_item(f"val_{key}", color=(244, 67, 54))
                if category == "rework":
                    rework_fails.append(name)
                else:
                    reject_fails.append(name)

        # -- Overall verdict -----------------------------------------------

        state.pred_overlay = None
        state.pred_result = None

        if rework_fails:
            dpg.set_value("verdict_text", "REWORK")
            dpg.bind_item_theme("verdict_box", themes["rework"])
            state.verdict = "REWORK"
            # Show overlay on FAIL
            if state.overlay_image is not None:
                _show_image(state.overlay_image)

        elif reject_fails:
            dpg.set_value("verdict_text", "REJECT")
            dpg.bind_item_theme("verdict_box", themes["reject"])
            state.verdict = "REJECT"
            # Show overlay on FAIL
            if state.overlay_image is not None:
                _show_image(state.overlay_image)

        else:
            # Geometric PASS → run Mask R-CNN defect model
            dpg.set_value("info_text", "Running Mask R-CNN...")
            precomputed = getattr(state, '_precomputed_720', None)
            has_defect = _run_maskrcnn_on_pass(precomputed_720=precomputed)

            if has_defect:
                dpg.set_value("verdict_text", "REJECT")
                dpg.bind_item_theme("verdict_box", themes["reject"])
                state.verdict = "REJECT"
                # Show defect overlay on FAIL
                if state.pred_overlay is not None:
                    _show_image(state.pred_overlay)
                elif state.overlay_image is not None:
                    _show_image(state.overlay_image)
            else:
                dpg.set_value("verdict_text", "PASS")
                dpg.bind_item_theme("verdict_box", themes["pass"])
                state.verdict = "PASS"
                # PASS — show original image (no overlay)
                if state.image is not None:
                    _show_image(state.image)

    # -- Load image at path -----------------------------------------------

    def _load_image_path(path: str):
        img = cv2.imread(path)
        if img is None:
            dpg.set_value("info_text", f"Cannot read: {path}")
            return

        state.image = img
        state.overlay_image = None
        state.result = None
        state.result_normed = None
        _show_image(img)
        _clear_results()
        _update_analyze_btn()

        folder = str(Path(path).parent)
        exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff'}
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if Path(f).suffix.lower() in exts],
            key=lambda p: Path(p).name.lower())
        state.file_list = files
        try:
            state.file_index = files.index(os.path.normpath(path))
        except ValueError:
            state.file_index = 0
        _update_nav_btns()

        h, w = img.shape[:2]
        dpg.set_value("info_text", f"Loaded: {Path(path).name}  ({w}x{h})")

    # -- Navigation -------------------------------------------------------

    def _navigate(delta: int):
        idx = state.file_index + delta
        if 0 <= idx < len(state.file_list):
            _load_image_path(state.file_list[idx])

    # -- Model button callback --------------------------------------------

    def _on_model_btn(model_name: str):
        state.current_model = model_name
        state.good_stats = load_good_stats(MODEL_CSV[model_name])
        state.thresholds = load_best_thresholds(
            model_name, state.good_stats, state.sigma)
        _update_model_btns()
        _sync_thresholds_to_ui()
        if state.result is not None:
            _evaluate()
        tuned_path = TUNED_JSON.get(model_name)
        if tuned_path and tuned_path.exists():
            dpg.set_value("info_text", f"{model_name} - tuned thresholds loaded")
        elif state.good_stats:
            dpg.set_value("info_text", f"{model_name} - sigma={state.sigma}")
        else:
            dpg.set_value("info_text", f"{model_name} - using defaults")

    # -- Analyze callback -------------------------------------------------

    def _on_analyze(sender=None, app_data=None):
        if state.image is None:
            return
        t_start = time.perf_counter()
        th = dpg.get_value("spin_thresh")

        dpg.set_value("info_text", "Analyzing...")

        # ── Run measure_oring and bin_crop_720 concurrently ───────
        # They're independent: measure_oring uses original image,
        # bin_crop_720 also uses original image but produces a different
        # output.  Both are CPU-bound so threading overlaps their work.
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_geo = pool.submit(measure_oring, state.image,
                                  DEFAULT_BG_VALUE, th)
            fut_720 = pool.submit(bin_crop_720, state.image)
            state.result = fut_geo.result()
            state._precomputed_720 = fut_720.result()

        if state.result is None:
            dpg.set_value("info_text",
                          "Detection failed - adjust threshold")
            dpg.set_value("cycle_time_text", "")
            return

        h, w = state.image.shape[:2]
        state.resolution_scale = compute_resolution_scale(w, h)
        state.result_normed = normalize_measurements(
            state.result, state.resolution_scale)
        state.overlay_image = draw_overlay(state.image, state.result)

        _evaluate()

        t_end = time.perf_counter()
        cycle_ms = (t_end - t_start) * 1000
        dpg.set_value("cycle_time_text", f"Cycle Time: {cycle_ms:.0f} ms")
        dpg.set_value("info_text", f"Analysis complete  ({cycle_ms:.0f} ms)")

    # -- File dialog callback ---------------------------------------------

    def _file_selected(sender, app_data):
        selections = app_data.get("selections", {})
        if selections:
            path = list(selections.values())[0]
            _load_image_path(path)

    # -- Threshold edit callback ------------------------------------------

    def _on_threshold_edited(sender=None, app_data=None):
        if state.result is not None:
            _evaluate()

    # -- Camera callbacks -------------------------------------------------

    def _on_toggle_stream(sender=None, app_data=None):
        if state.camera_streaming:
            _stop_stream()
        else:
            _start_stream()

    def _start_stream():
        if not HIKROBOT_AVAILABLE:
            dpg.set_value("info_text", "Hikrobot SDK not available")
            return
        if state.camera is None:
            try:
                deviceList = MV_CC_DEVICE_INFO_LIST()
                tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
                ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
                if ret != 0 or deviceList.nDeviceNum == 0:
                    dpg.set_value("info_text", "No cameras found")
                    return
                state.camera = MvCamera()
                stDevInfo = cast(
                    deviceList.pDeviceInfo[0],
                    POINTER(MV_CC_DEVICE_INFO)).contents
                state.camera.MV_CC_CreateHandle(stDevInfo)
                state.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                state.camera.MV_CC_SetEnumValue("TriggerMode", 0)
            except Exception as e:
                state.camera = None
                dpg.set_value("info_text", f"Camera error: {e}")
                return

        ret = state.camera.MV_CC_StartGrabbing()
        if ret != 0:
            dpg.set_value("info_text",
                          f"StartGrabbing failed (0x{ret:08X})")
            return

        state.camera_streaming = True
        dpg.configure_item("btn_stream", label="Stop Stream")
        dpg.configure_item("btn_capture", enabled=True)
        dpg.set_value("info_text", "Camera streaming - press Capture")

    def _stop_stream():
        if state.camera is not None and state.camera_streaming:
            state.camera.MV_CC_StopGrabbing()
        state.camera_streaming = False
        dpg.configure_item("btn_stream", label="Start Stream")
        dpg.configure_item("btn_capture", enabled=False)

    def _grab_frame() -> Optional[np.ndarray]:
        if state.camera is None or not state.camera_streaming:
            return None
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = state.camera.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            return None
        try:
            buf_len = stOutFrame.stFrameInfo.nFrameLen
            pData = (c_ubyte * buf_len).from_address(stOutFrame.pBufAddr)
            raw = np.frombuffer(pData, dtype=np.uint8).copy()
            fh = stOutFrame.stFrameInfo.nHeight
            fw = stOutFrame.stFrameInfo.nWidth
            px = stOutFrame.stFrameInfo.enPixelType
            if px == 0x01080001:
                frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_GRAY2BGR)
            elif px == 0x01080009:
                frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_BayerRG2BGR)
            elif px == 0x0108000A:
                frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_BayerGB2BGR)
            elif px == 0x0108000B:
                frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_BayerGR2BGR)
            elif px == 0x01080008:
                frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_BayerBG2BGR)
            elif px == 0x02180014:
                frame = cv2.cvtColor(raw.reshape(fh, fw, 3), cv2.COLOR_RGB2BGR)
            elif px == 0x02180015:
                frame = raw.reshape(fh, fw, 3)
            else:
                try:
                    frame = cv2.cvtColor(raw.reshape(fh, fw), cv2.COLOR_GRAY2BGR)
                except ValueError:
                    frame = raw.reshape(fh, fw, -1)[:, :, :3].copy()
            return frame
        finally:
            state.camera.MV_CC_FreeImageBuffer(stOutFrame)

    def _on_capture(sender=None, app_data=None):
        if not state.camera_streaming:
            return
        _stop_stream()
        frame = state.latest_frame
        if frame is None:
            dpg.set_value("info_text", "No frame available")
            return
        state.image = frame
        state.overlay_image = None
        state.result = None
        _show_image(frame)
        _clear_results()
        _update_analyze_btn()
        state.file_list = []
        state.file_index = -1
        _update_nav_btns()
        fh, fw = frame.shape[:2]
        dpg.set_value("info_text", f"Captured from camera ({fw}x{fh})")

    # -- Per-frame update (camera streaming) ------------------------------

    def _frame_update():
        """Called every frame from the render loop."""
        # Update model status indicator with color
        mrcnn_txt = f"Mask R-CNN: {state.maskrcnn_status}"
        dpg.set_value("status_maskrcnn", mrcnn_txt)

        if state.maskrcnn_ready:
            dpg.bind_item_theme("status_maskrcnn", themes["status_ready"])
        elif "error" in state.maskrcnn_status:
            dpg.bind_item_theme("status_maskrcnn", themes["status_error"])
        else:
            dpg.bind_item_theme("status_maskrcnn", themes["status_loading"])

        # Camera streaming preview
        if state.camera_streaming:
            frame = _grab_frame()
            if frame is not None:
                state.latest_frame = frame
                _show_image(frame)

    # ══════════════════════════════════════════════════════════════════════
    #  Build widgets (all functions are now defined)
    # ══════════════════════════════════════════════════════════════════════

    # -- File dialog (hidden until triggered) -----------------------------
    with dpg.file_dialog(directory_selector=False, show=False,
                         callback=_file_selected,
                         tag="file_dialog", width=700, height=400):
        dpg.add_file_extension(".bmp")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".tiff")
        dpg.add_file_extension(".*")

    # -- Main window ------------------------------------------------------
    with dpg.window(tag="primary_window"):

        # ── TOP STATUS BAR: DL model readiness ────────────────────────
        with dpg.child_window(height=38, tag="status_bar", border=False):
            with dpg.group(horizontal=True):
                dpg.add_text("DL Model:", color=(160, 160, 170))
                dpg.add_spacer(width=10)
                dpg.add_text("Mask R-CNN: loading...",
                             tag="status_maskrcnn",
                             color=(255, 193, 7))
                dpg.add_spacer(width=40)
                dpg.add_text("", tag="info_text", color=(140, 160, 190),
                             wrap=500)
        if status_font:
            dpg.bind_item_font("status_bar", status_font)

        dpg.add_spacer(height=2)

        with dpg.group(horizontal=True):

            # ── LEFT PANEL: image display ─────────────────────────────────
            with dpg.child_window(width=830, tag="left_panel"):
                dpg.add_image("tex_main", tag="img_display")

            # ── RIGHT PANEL: controls + results ───────────────────────────
            with dpg.child_window(tag="right_panel"):

                # --- O-Ring Model toggle buttons --------------------------
                _hdr = dpg.add_text("O-RING MODEL", color=(130, 170, 220))
                if heading_font:
                    dpg.bind_item_font(_hdr, heading_font)
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Model 1", tag="btn_model1",
                                   width=200, height=50,
                                   callback=lambda: _on_model_btn("Model 1"))
                    dpg.add_spacer(width=10)
                    dpg.add_button(label="Model 2", tag="btn_model2",
                                   width=200, height=50,
                                   callback=lambda: _on_model_btn("Model 2"))

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=6)

                # --- Detection controls ----------------------------------
                _hdr2 = dpg.add_text("DETECTION", color=(130, 170, 220))
                if heading_font:
                    dpg.bind_item_font(_hdr2, heading_font)
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    btn_ld = dpg.add_button(label="Load Image", tag="btn_load",
                                   width=120, height=36,
                                   callback=lambda: dpg.show_item("file_dialog"))
                    dpg.bind_item_theme(btn_ld, themes["btn_green"])
                    dpg.add_spacer(width=4)
                    dpg.add_button(label="Analyze", tag="btn_analyze",
                                   width=120, height=36,
                                   callback=_on_analyze)

                dpg.add_spacer(height=6)

                # Navigation
                with dpg.group(horizontal=True):
                    dpg.add_button(label="< Prev", tag="btn_prev",
                                   width=90, height=30, enabled=False,
                                   callback=lambda: _navigate(-1))
                    dpg.add_spacer(width=8)
                    dpg.add_text("", tag="nav_label")
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Next >", tag="btn_next",
                                   width=90, height=30, enabled=False,
                                   callback=lambda: _navigate(1))

                dpg.add_spacer(height=6)

                # Camera controls
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start Stream", tag="btn_stream",
                                   width=130, height=30,
                                   callback=_on_toggle_stream)
                    dpg.add_spacer(width=4)
                    dpg.add_button(label="Capture", tag="btn_capture",
                                   width=100, height=30, enabled=False,
                                   callback=_on_capture)

                # Threshold
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_text("Thresh:", color=(160, 160, 170))
                    dpg.add_input_int(tag="spin_thresh", default_value=30,
                                      min_value=1, max_value=255,
                                      min_clamped=True, max_clamped=True,
                                      width=70, step=0)

                # Detection threshold
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_text("Detection Threshold:", color=(160, 160, 170))
                    dpg.add_input_float(tag="spin_det_thresh",
                                        default_value=0.5,
                                        min_value=0.1, max_value=0.8,
                                        min_clamped=True, max_clamped=True,
                                        width=80, step=0.05,
                                        format="%.2f")

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=6)

                # --- Verdict banner (BIG) ─────────────────────────────────
                with dpg.child_window(height=100, tag="verdict_box",
                                      border=False):
                    dpg.add_spacer(height=16)
                    _vt = dpg.add_text("AWAITING", tag="verdict_text",
                                 color=(204, 204, 204), indent=16)
                    if large_font:
                        dpg.bind_item_font(_vt, large_font)

                dpg.add_spacer(height=4)
                dpg.add_text("", tag="cycle_time_text",
                             color=(140, 200, 255))

                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=6)

                # --- Metrics table ----------------------------------------
                _hdr3 = dpg.add_text("MEASUREMENTS & THRESHOLDS",
                             color=(130, 170, 220))
                if heading_font:
                    dpg.bind_item_font(_hdr3, heading_font)
                dpg.add_spacer(height=4)

                with dpg.table(tag="metrics_table", header_row=True,
                               borders_innerH=True, borders_outerH=True,
                               borders_innerV=True, borders_outerV=True,
                               resizable=True, pad_outerX=True):
                    dpg.add_table_column(label="Metric", width_fixed=True,
                                         init_width_or_weight=180)
                    dpg.add_table_column(label="Measured", width_fixed=True,
                                         init_width_or_weight=90)
                    dpg.add_table_column(label="Min", width_fixed=True,
                                         init_width_or_weight=90)
                    dpg.add_table_column(label="Max", width_fixed=True,
                                         init_width_or_weight=90)
                    dpg.add_table_column(label="Status", width_fixed=True,
                                         init_width_or_weight=70)

                    for key, name, unit, ttype, dec, step, s_lo, s_hi, category in METRIC_DEFS:
                        label_text = f"{name}" + (f" ({unit})" if unit else "")
                        cat_tag_s = "[R]" if category == "rework" else "[X]"
                        cat_color = (255, 183, 77) if category == "rework" else (239, 83, 80)

                        with dpg.table_row():
                            dpg.add_text(f"{cat_tag_s} {label_text}",
                                         color=cat_color)
                            dpg.add_text("-", tag=f"val_{key}",
                                         color=(170, 170, 175))
                            dpg.add_input_float(
                                tag=f"lo_{key}",
                                default_value=state.thresholds[key]["lo"],
                                min_value=s_lo, max_value=s_hi,
                                min_clamped=True, max_clamped=True,
                                width=80, step=0,
                                format=f"%.{dec}f",
                                enabled=(ttype != "max"),
                                callback=_on_threshold_edited)
                            dpg.add_input_float(
                                tag=f"hi_{key}",
                                default_value=state.thresholds[key]["hi"],
                                min_value=s_lo, max_value=s_hi,
                                min_clamped=True, max_clamped=True,
                                width=80, step=0,
                                format=f"%.{dec}f",
                                enabled=(ttype != "min"),
                                callback=_on_threshold_edited)
                            dpg.add_text("-", tag=f"status_{key}",
                                         color=(170, 170, 175))

    # ── Initial UI state ─────────────────────────────────────────────────
    _update_model_btns()
    _update_analyze_btn()
    _update_nav_btns()
    dpg.bind_item_theme("verdict_box", themes["awaiting"])

    # Set initial info text
    dpg.set_value("info_text", "Ready")

    return _frame_update


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    state = AppState()

    # Start loading DL models in background immediately
    preload_models(state)

    dpg.create_context()
    dpg.create_viewport(title="O-Ring Inspection - Pass / Rework / Reject",
                        width=1580, height=1000)

    # ── Font registry ─────────────────────────────────────────────────
    with dpg.font_registry():
        # Try system fonts; fall back to DPG default
        _font_candidates = [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]
        _font_path = None
        for _fp in _font_candidates:
            if os.path.isfile(_fp):
                _font_path = _fp
                break

        if _font_path:
            default_font = dpg.add_font(_font_path, 18)
            large_font   = dpg.add_font(_font_path, 44)
            heading_font = dpg.add_font(_font_path, 22)
            status_font  = dpg.add_font(_font_path, 15)
        else:
            default_font = None
            large_font   = None
            heading_font = None
            status_font  = None

    if default_font:
        dpg.bind_font(default_font)

    # ── Dark theme ────────────────────────────────────────────────────
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (32, 32, 38))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 235))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 52))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (60, 60, 70))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (50, 50, 60))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 65, 78))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (65, 65, 78))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (50, 50, 58))
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (40, 40, 48))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, (60, 60, 70))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, (48, 48, 56))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (70, 70, 82))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (60, 60, 70))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (80, 80, 95))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (90, 90, 108))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (35, 35, 42))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)

    dpg.bind_theme(global_theme)

    frame_update = build_gui(state, default_font, large_font, heading_font, status_font)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    while dpg.is_dearpygui_running():
        frame_update()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
