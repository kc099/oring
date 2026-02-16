"""
O-Ring Inspection GUI — Pass / Rework Classification

Loads a full-resolution (2448×2048) o-ring image, performs background
subtraction to segment the ring, detects inner/outer contours, computes
geometric measurements, and compares them against statistical thresholds
derived from known-good samples to produce a PASS / REWORK verdict.

Thresholds are auto-computed from `good_measurements.csv` in the same
folder.  Every threshold is editable in the UI; the σ-multiplier
controls how many standard deviations from the good-sample mean
define the acceptable range.

Metrics
-------
From original measurement set:
    outer_radius, inner_radius, center_dist, ring_thickness,
    min_thickness, max_thickness, mean_radius, annular_area

New metrics added:
    thickness_range   — max − min wall thickness  (uniformity indicator)
    thickness_ratio   — max / min wall thickness   (should be ≈ 1.0)
    eccentricity_pct  — center_dist / mean_radius × 100
    circularity_outer — 4πA / P²  of outer contour (1.0 = perfect circle)

Usage:
    python rework/inspection_gui.py

Author: GitHub Copilot
Date: February 10, 2026
"""

import sys
import csv
import math
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

# Per-model CSV paths
MODEL_CSV = {
    "Model 1": SCRIPT_DIR / "model1good_measurements.csv",
    "Model 2": SCRIPT_DIR / "good_measurements.csv",
}
DEFAULT_MODEL = "Model 2"

# Metric definitions -------------------------------------------------------
# (key, display_name, unit, thresh_type, decimals, spin_step, spin_lo, spin_hi)
#   thresh_type:  'max'   → value must be ≤ hi   (lo spinbox disabled)
#                 'min'   → value must be ≥ lo   (hi spinbox disabled)
#                 'range' → lo ≤ value ≤ hi

METRIC_DEFS = [
    # ---- primary (most discriminative) ------------------------------------
    ("center_dist",      "Concentricity",        "px",   "max",   1, 1.0,    0, 500),
    ("thickness_range",  "Thickness Range",       "px",   "max",   1, 1.0,    0, 500),
    ("thickness_ratio",  "Thickness Ratio",       "",     "max",   2, 0.01,   1, 3),
    ("eccentricity_pct", "Eccentricity",          "%",    "max",   2, 0.1,    0, 50),
    ("circularity_outer","Outer Circularity",     "",     "min",   3, 0.005,  0, 1),
    # ---- secondary --------------------------------------------------------
    ("outer_radius",     "Outer Radius",          "px",   "range", 1, 1.0,  400, 1200),
    ("inner_radius",     "Inner Radius",          "px",   "range", 1, 1.0,  200, 800),
    ("ring_thickness",   "Ring Thickness",        "px",   "range", 1, 1.0,  100, 600),
    ("mean_radius",      "Mean Radius",           "px",   "range", 1, 1.0,  300, 900),
    ("min_thickness",    "Min Wall Thickness",    "px",   "min",   1, 1.0,    0, 500),
    ("annular_area_k",   "Annular Area (×1 000)", "",     "range", 1, 5.0,    0, 5000),
]

# Fallback thresholds when good_measurements.csv is missing ----------------
DEFAULT_THRESHOLDS = {
    "center_dist":       {"lo": 0.0,   "hi": 35.0},
    "thickness_range":   {"lo": 0.0,   "hi": 70.0},
    "thickness_ratio":   {"lo": 1.0,   "hi": 1.25},
    "eccentricity_pct":  {"lo": 0.0,   "hi": 6.0},
    "circularity_outer": {"lo": 0.85,  "hi": 1.0},
    "outer_radius":      {"lo": 650.0, "hi": 680.0},
    "inner_radius":      {"lo": 375.0, "hi": 400.0},
    "ring_thickness":    {"lo": 260.0, "hi": 295.0},
    "mean_radius":       {"lo": 515.0, "hi": 535.0},
    "min_thickness":     {"lo": 230.0, "hi": 500.0},
    "annular_area_k":    {"lo": 780.0, "hi": 920.0},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics / threshold helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_good_stats(csv_path: Path) -> Optional[Dict]:
    """Return per-metric ``{mean, std}`` computed from raw good-sample CSV."""
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

    # Columns that exist directly in the CSV
    for key in ("center_dist", "outer_radius", "inner_radius",
                "ring_thickness", "mean_radius", "min_thickness", "max_thickness"):
        vals = [float(r[key]) for r in rows]
        m, s = _ms(vals)
        stats[key] = {"mean": m, "std": s}

    # Derived: thickness_range  = max − min
    ranges = [float(r["max_thickness"]) - float(r["min_thickness"]) for r in rows]
    m, s = _ms(ranges)
    stats["thickness_range"] = {"mean": m, "std": s}

    # Derived: thickness_ratio  = max / min
    ratios = [float(r["max_thickness"]) / float(r["min_thickness"])
              for r in rows if float(r["min_thickness"]) > 0]
    m, s = _ms(ratios)
    stats["thickness_ratio"] = {"mean": m, "std": s}

    # Derived: annular_area_k  = area / 1000
    areas_k = [float(r["annular_area"]) / 1000.0 for r in rows]
    m, s = _ms(areas_k)
    stats["annular_area_k"] = {"mean": m, "std": s}

    # Derived: eccentricity_pct
    ecc = [float(r["center_dist"]) / float(r["mean_radius"]) * 100 for r in rows]
    m, s = _ms(ecc)
    stats["eccentricity_pct"] = {"mean": m, "std": s}

    return stats


def compute_thresholds(stats: Optional[Dict], sigma: float = 2.5) -> Dict[str, Dict]:
    """Compute ``{lo, hi}`` per metric from good-sample stats."""
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

    if np.mean(binary == 255) > 0.75:          # background became white
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


def contour_distances(outer, inner, sample: int = 300):
    """Min / max wall thickness measured between contour point-sets."""
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
    """Run full measurement pipeline on a BGR image."""
    mask = build_mask(image, bg_value, threshold)
    outer, inner = find_contours(mask)
    if outer is None or inner is None:
        return None

    (ox, oy), orad = cv2.minEnclosingCircle(outer)
    (ix, iy), irad = cv2.minEnclosingCircle(inner)
    cdx, cdy = float(ox - ix), float(oy - iy)
    cdist = math.hypot(cdx, cdy)
    rthick = float(orad) - float(irad)
    mrad = (float(orad) + float(irad)) / 2.0

    min_t, max_t = contour_distances(outer, inner)
    area_px = int(np.count_nonzero(mask))

    # Circularity of outer contour: 4πA / P²
    o_area = cv2.contourArea(outer)
    o_peri = cv2.arcLength(outer, True)
    circ = (4.0 * math.pi * o_area / (o_peri ** 2)) if o_peri > 0 else 0.0

    return {
        "center_dist":       cdist,
        "center_dx":         cdx,
        "center_dy":         cdy,
        "thickness_range":   max_t - min_t,
        "thickness_ratio":   (max_t / min_t) if min_t > 0 else 99.0,
        "ring_thickness":    rthick,
        "outer_radius":      float(orad),
        "inner_radius":      float(irad),
        "mean_radius":       mrad,
        "min_thickness":     min_t,
        "max_thickness":     max_t,
        "annular_area":      area_px,
        "annular_area_k":    area_px / 1000.0,
        "eccentricity_pct":  (cdist / mrad * 100) if mrad > 0 else 0,
        "circularity_outer": circ,
        # keep contours for overlay drawing
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

    # Fitted-circle centres
    cv2.circle(vis, (int(ox), int(oy)), 10, (0, 255, 0), -1)
    cv2.circle(vis, (int(ix), int(iy)), 10, (0, 0, 255), -1)
    cv2.line(vis, (int(ox), int(oy)), (int(ix), int(iy)), (0, 255, 255), 2)

    # Fitted circles (dashed look via thin lines)
    cv2.circle(vis, (int(ox), int(oy)), int(result["outer_radius"]),
               (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(vis, (int(ix), int(iy)), int(result["inner_radius"]),
               (0, 0, 255), 1, cv2.LINE_AA)

    # Legend
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
        self.setWindowTitle("O-Ring Inspection — Pass / Rework")
        self.setGeometry(40, 40, 1520, 920)

        self.image: Optional[np.ndarray] = None
        self.overlay_image: Optional[np.ndarray] = None
        self.result: Optional[Dict] = None

        # Statistics & thresholds
        self.current_model = DEFAULT_MODEL
        self.good_stats = load_good_stats(MODEL_CSV[self.current_model])
        self.sigma = 2.5
        self.thresholds = compute_thresholds(self.good_stats, self.sigma)

        self.lo_spins: Dict[str, QDoubleSpinBox] = {}
        self.hi_spins: Dict[str, QDoubleSpinBox] = {}

        self._init_ui()
        self._populate_table()

        if self.good_stats:
            csv_name = MODEL_CSV[self.current_model].name
            self.info_label.setText(
                f"✓ Loaded good-sample stats from {csv_name}  "
                f"({self.current_model}, σ = {self.sigma})")
        else:
            self.info_label.setText(
                f"⚠ CSV for {self.current_model} not found — using default thresholds")

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
        from PySide6.QtWidgets import QComboBox
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

        sg.setLayout(sl)
        right_lay.addWidget(sg)

        # --- Verdict banner -----------------------------------------------
        self.verdict_frame = QFrame()
        self.verdict_frame.setMinimumHeight(100)
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
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Metric", "Measured", "Min", "Max", "Status"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3, 4):
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
        for row, (key, name, unit, ttype, dec, step, s_lo, s_hi) in \
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

        # Show extra informational metrics
        dx = self.result["center_dx"]
        dy = self.result["center_dy"]
        self.info_label.setText(
            f"center_dx = {dx:+.1f} px   center_dy = {dy:+.1f} px   "
            f"max_thickness = {self.result['max_thickness']:.1f} px   "
            f"annular_area = {self.result['annular_area']:,} px²")

    def _evaluate(self):
        """Compare each metric against thresholds and update the table."""
        if self.result is None:
            return

        self._read_thresholds_from_table()
        failures = []

        for row, (key, name, _unit, ttype, dec, *_) in enumerate(METRIC_DEFS):
            val = self.result.get(key)
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
                failures.append(name)

        # Overall verdict
        if not failures:
            self.verdict_label.setText("✅  PASS")
            self.verdict_label.setStyleSheet("color: white;")
            self.verdict_frame.setStyleSheet(
                "background:#2E7D32; border-radius:12px; padding:8px;")
            self.verdict_detail.setText("All metrics within tolerance")
            self.verdict_detail.setStyleSheet("color:#C8E6C9;")
        else:
            self.verdict_label.setText("❌  REWORK")
            self.verdict_label.setStyleSheet("color: white;")
            self.verdict_frame.setStyleSheet(
                "background:#C62828; border-radius:12px; padding:8px;")
            self.verdict_detail.setText(
                f"{len(failures)} metric(s) out of tolerance: "
                + ", ".join(failures))
            self.verdict_detail.setStyleSheet("color:#FFCDD2;")

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

    def _on_threshold_edited(self, _=None):
        """Re-evaluate when user manually edits a threshold spinbox."""
        if self.result is not None:
            self._evaluate()

    def _recompute_thresholds(self, sigma):
        """Recompute thresholds with a new σ multiplier."""
        self.sigma = sigma
        self.thresholds = compute_thresholds(self.good_stats, sigma)
        self._sync_thresholds_to_table()
        if self.result is not None:
            self._evaluate()

    def _on_model_changed(self, model_name: str):
        """Switch to a different o-ring model's reference statistics."""
        self.current_model = model_name
        csv_path = MODEL_CSV[model_name]
        self.good_stats = load_good_stats(csv_path)
        self.thresholds = compute_thresholds(self.good_stats, self.sigma)
        self._sync_thresholds_to_table()
        if self.result is not None:
            self._evaluate()
        if self.good_stats:
            self.info_label.setText(
                f"✓ Switched to {model_name} — stats from {csv_path.name}  "
                f"(σ = {self.sigma})")
        else:
            self.info_label.setText(
                f"⚠ CSV for {model_name} not found — using default thresholds")

    def _reset_thresholds(self):
        self.sigma_spin.setValue(2.5)       # triggers _recompute_thresholds

    # ── Display helpers ──────────────────────────────────────────────────

    def _show_cv(self, cv_img: np.ndarray, label: QLabel):
        """Display a cv2 BGR image (or grayscale) in a QLabel."""
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
