"""
Update Model 1 good-sample measurements by combining:
    1. Existing images from model1good_measurements.csv  (18 images)
    2. New samples from rework_measurements.csv           (141 images)

Then re-measures **all** unique images using the full 18-metric pipeline
(from inspection_gui.py) and overwrites model1good_measurements.csv.

After running this, execute  tune_thresholds.py  to recompute the
model1_tuned_thresholds.json.

Usage:
    conda activate dl
    python rework/update_model1_stats.py

Author: GitHub Copilot
Date:   February 18, 2026
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from workspace root
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE))

from rework.inspection_gui import METRIC_DEFS, measure_oring

# ── Paths ─────────────────────────────────────────────────────────────────
IMAGE_DIR = WORKSPACE / "Original Data" / "model1good"
OLD_CSV = SCRIPT_DIR / "model1good_measurements.csv"            # existing 18
NEW_CSV = WORKSPACE / "rework_measurements.csv"                  # new 141
OUTPUT_CSV = SCRIPT_DIR / "model1good_measurements.csv"          # overwrite
STATS_CSV = SCRIPT_DIR / "model1good_measurements_stats.csv"

BG_VALUE = 20
THRESHOLD = 30

METRIC_COLS = [m[0] for m in METRIC_DEFS]  # 18 metric keys


def collect_image_names() -> list:
    """Combine unique image names from old and new CSVs."""
    names = set()

    # From existing model1good_measurements.csv
    if OLD_CSV.exists():
        with open(OLD_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                names.add(r["image"])
        print(f"  Old CSV ({OLD_CSV.name}): {len(names)} images")

    # From rework_measurements.csv 
    n_before = len(names)
    if NEW_CSV.exists():
        with open(NEW_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                names.add(r["image"])
        print(f"  New CSV ({NEW_CSV.name}): {len(names) - n_before} new images added")
    else:
        print(f"  WARNING: {NEW_CSV} not found!")

    return sorted(names)


def main():
    print("=" * 70)
    print("  UPDATE MODEL 1 GOOD-SAMPLE MEASUREMENTS")
    print("=" * 70)
    print(f"\n  Image directory: {IMAGE_DIR}")
    print(f"  BG value: {BG_VALUE}, Threshold: {THRESHOLD}\n")

    # 1. Collect all image names
    image_names = collect_image_names()
    print(f"\n  Total unique images: {len(image_names)}\n")

    # 2. Measure each image with the full 18-metric pipeline
    rows = []
    failed = []

    for i, fname in enumerate(image_names):
        fpath = IMAGE_DIR / fname
        if not fpath.exists():
            print(f"  ✗ File not found: {fname}")
            failed.append(fname)
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  ✗ Cannot read: {fname}")
            failed.append(fname)
            continue

        result = measure_oring(img, BG_VALUE, THRESHOLD)
        if result is None:
            print(f"  ✗ Detection failed: {fname}")
            failed.append(fname)
            continue

        result["image"] = fname
        rows.append(result)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_names):
            print(f"  [{i+1}/{len(image_names)}] {fname}  "
                  f"outer_r={result.get('outer_radius', 0):.1f}  "
                  f"inner_r={result.get('inner_radius', 0):.1f}  "
                  f"thick={result.get('ring_thickness', 0):.1f}")

    print(f"\n  Processed: {len(rows)}/{len(image_names)}  (failed: {len(failed)})")
    if failed:
        print(f"  Failed images: {failed}")

    if not rows:
        print("  No valid measurements!")
        sys.exit(1)

    # 3. Save per-image CSV (overwrite old model1good_measurements.csv)
    all_cols = ["image"] + METRIC_COLS
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  ✓ Saved measurements ({len(rows)} images): {OUTPUT_CSV}")

    # 4. Compute and save statistics
    print(f"\n{'='*90}")
    print(f"  GOOD MODEL 1 O-RING STATISTICS  (n = {len(rows)})")
    print(f"{'='*90}")
    header = f"{'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P5':>10} {'P95':>10}"
    print(header)
    print("-" * len(header))

    stats_rows = []
    for col in METRIC_COLS:
        vals = np.array([r.get(col, 0) for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))

        print(f"  {col:<22} {mean:>10.2f} {std:>10.2f} {vmin:>10.2f} {vmax:>10.2f} {p5:>10.2f} {p95:>10.2f}")

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
    print(f"\n  ✓ Saved statistics: {STATS_CSV}")
    print(f"\n  Next step: run  python rework/tune_thresholds.py  to update thresholds")


if __name__ == "__main__":
    main()
