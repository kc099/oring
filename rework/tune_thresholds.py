"""
Tune thresholds so that ALL known-good images pass every metric.

For each metric the script:
    1. Computes the σ=2.5 threshold from good-sample stats (mean ± 2.5σ).
    2. Measures every good image and finds the worst-case value.
    3. If any good image falls outside the σ-threshold, calculates the
       *exact* tolerance % that must be added to the threshold band so
       that good image just barely passes.
    4. Saves tuned {lo, hi} and the required tolerance_pct per metric.

Outputs per model:
    rework/<model>_tuned_thresholds.json
        { "<metric>": {"lo": ..., "hi": ..., "tolerance_pct": ...}, ... }

Usage:
    python rework/tune_thresholds.py
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Re-use measurement code from inspection_gui ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rework.inspection_gui import (
    METRIC_DEFS, measure_oring, load_good_stats, compute_thresholds,
    MODEL_CSV,
)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE  = SCRIPT_DIR.parent

# ── Per-model config ─────────────────────────────────────────────────────
MODEL_CONFIG = {
    "Model 2": {
        "image_dir": WORKSPACE / "Original Data" / "good",
        "csv_path":  SCRIPT_DIR / "good_measurements.csv",
        "output":    SCRIPT_DIR / "model2_tuned_thresholds.json",
        "selected_only": False,  # use all images
    },
    "Model 1": {
        "image_dir": WORKSPACE / "Original Data" / "model1good",
        "csv_path":  SCRIPT_DIR / "model1good_measurements.csv",
        "output":    SCRIPT_DIR / "model1_tuned_thresholds.json",
        "selected_only": True,  # use only images listed in CSV
    },
}

SIGMA = 2.5
BG_VALUE = 20
THRESHOLD = 30


def get_image_list(cfg: dict) -> List[Path]:
    """Return list of image paths to process."""
    img_dir = cfg["image_dir"]
    if cfg["selected_only"]:
        # Read image names from existing CSV
        csv_path = cfg["csv_path"]
        names = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["image"])
        paths = []
        for name in names:
            p = img_dir / name
            if p.exists():
                paths.append(p)
            else:
                # try without extension / with different extension
                for ext in [".bmp", ".png", ".jpg"]:
                    p2 = img_dir / (Path(name).stem + ext)
                    if p2.exists():
                        paths.append(p2)
                        break
        return paths
    else:
        exts = {".bmp", ".png", ".jpg", ".jpeg", ".tiff"}
        return sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in exts)


def tune_model(model_name: str):
    """Run all good images, compute per-metric min tolerance % needed."""
    cfg = MODEL_CONFIG[model_name]
    print(f"\n{'='*60}")
    print(f"  Tuning thresholds for {model_name}")
    print(f"{'='*60}")

    # 1. Load stats & compute σ-based thresholds
    stats = load_good_stats(cfg["csv_path"])
    if stats is None:
        print(f"  ERROR: Cannot load stats from {cfg['csv_path']}")
        return
    base_thresholds = compute_thresholds(stats, SIGMA)

    # 2. Measure all good images
    images = get_image_list(cfg)
    print(f"  Images to process: {len(images)}")

    # Collect per-metric measured values from all good images
    metric_values: Dict[str, List[float]] = {m[0]: [] for m in METRIC_DEFS}
    failed_images = []

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP (cannot read): {img_path.name}")
            continue
        result = measure_oring(img, BG_VALUE, THRESHOLD)
        if result is None:
            print(f"  SKIP (detection failed): {img_path.name}")
            failed_images.append(img_path.name)
            continue
        for key, *_ in METRIC_DEFS:
            val = result.get(key)
            if val is not None:
                metric_values[key].append(val)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)}...")

    print(f"  Successfully measured: {len(images) - len(failed_images)}")
    if failed_images:
        print(f"  Failed: {failed_images}")

    # 3. For each metric, find the required tolerance %
    tuned: Dict[str, Dict] = {}

    print(f"\n  {'Metric':<22} {'σ-lo':>8} {'σ-hi':>8} │ "
          f"{'obs_min':>8} {'obs_max':>8} │ "
          f"{'tuned_lo':>8} {'tuned_hi':>8} │ {'tol%':>6}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} │ "
          f"{'─'*8} {'─'*8} │ "
          f"{'─'*8} {'─'*8} │ {'─'*6}")

    for key, name, _unit, ttype, *_ in METRIC_DEFS:
        vals = metric_values[key]
        if not vals:
            tuned[key] = {"lo": base_thresholds[key]["lo"],
                          "hi": base_thresholds[key]["hi"],
                          "tolerance_pct": 0.0}
            continue

        obs_min = min(vals)
        obs_max = max(vals)

        base_lo = base_thresholds[key]["lo"]
        base_hi = base_thresholds[key]["hi"]

        # Start from σ-based thresholds and expand just enough
        # so ALL good images pass.
        new_lo = base_lo
        new_hi = base_hi

        # For "range" type: value must be ≥ lo  AND  ≤ hi
        # For "min" type:   value must be ≥ lo  (hi is unused / 9999)
        # For "max" type:   value must be ≤ hi  (lo is unused / 0)

        # Calculate how much each bound needs to move
        tol_pct = 0.0

        if ttype in ("range", "min"):
            # lo bound: if obs_min < base_lo, we need to lower lo
            if obs_min < base_lo and base_lo > 0:
                deficit = base_lo - obs_min
                tol_lo_pct = (deficit / base_lo) * 100.0
                # Set new_lo slightly below obs_min to ensure pass
                eps = max(abs(obs_min) * 1e-4, 0.01)
                new_lo = obs_min - eps
                tol_pct = max(tol_pct, tol_lo_pct)

        if ttype in ("range", "max"):
            # hi bound: if obs_max > base_hi, we need to raise hi
            if obs_max > base_hi and base_hi > 0:
                surplus = obs_max - base_hi
                tol_hi_pct = (surplus / base_hi) * 100.0
                # Set new_hi slightly above obs_max to ensure pass
                eps = max(abs(obs_max) * 1e-4, 0.01)
                new_hi = obs_max + eps
                tol_pct = max(tol_pct, tol_hi_pct)

        # Round: lo down, hi up to avoid rounding-induced failures
        new_lo = math.floor(new_lo * 10000) / 10000
        new_hi = math.ceil(new_hi * 10000) / 10000
        tol_pct = round(tol_pct, 2)

        tuned[key] = {
            "lo": new_lo,
            "hi": new_hi,
            "tolerance_pct": tol_pct,
        }

        marker = " ◄" if tol_pct > 0 else ""
        print(f"  {name:<22} {base_lo:>8.2f} {base_hi:>8.2f} │ "
              f"{obs_min:>8.2f} {obs_max:>8.2f} │ "
              f"{new_lo:>8.2f} {new_hi:>8.2f} │ {tol_pct:>5.1f}%{marker}")

    # 4. Save tuned thresholds
    out_path = cfg["output"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2)
    print(f"\n  ✓ Saved tuned thresholds → {out_path.name}")

    # 5. Verify: re-check all images pass with tuned thresholds
    n_pass = 0
    n_total = len(metric_values[METRIC_DEFS[0][0]])
    for idx in range(n_total):
        all_ok = True
        for key, _name, _unit, ttype, *_ in METRIC_DEFS:
            if idx >= len(metric_values[key]):
                continue
            val = metric_values[key][idx]
            lo = tuned[key]["lo"]
            hi = tuned[key]["hi"]
            if ttype in ("range", "min") and val < lo:
                all_ok = False
            if ttype in ("range", "max") and val > hi:
                all_ok = False
        if all_ok:
            n_pass += 1
    print(f"  Verification: {n_pass}/{n_total} good images pass all metrics"
          f" {'✓' if n_pass == n_total else '✗ PROBLEM!'}")


if __name__ == "__main__":
    for model in MODEL_CONFIG:
        tune_model(model)
    print("\nDone.")
