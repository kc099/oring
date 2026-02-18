# Rework Inspection — Tuning & Verdict Logic

## Overview

The **O-Ring Inspection GUI** (`inspection_gui.py`) analyses a full-resolution
o-ring image and classifies it into one of three verdicts:

| Verdict | Meaning |
|---------|---------|
| **PASS** | All 18 geometric metrics are within tolerance. |
| **REWORK** | Shape issues detected (fixable by trimming excess material). |
| **REJECT** | Structural issues detected (thickness, concentricity, area — unfixable). |

---

## Metrics & Categories

### REWORK metrics (6) — shape, fixable by trimming

| Key | Display Name | Type | Note |
|-----|-------------|------|------|
| `outer_radius` | Outer Radius | max only | Only flags if **too large** (excess material). Min threshold is ignored — a smaller outer radius is not a rework condition. |
| `inner_radius` | Inner Radius | min only | Only flags if **too small** (excess material inward). Max threshold is ignored — a larger inner radius is not a rework condition. |
| `circularity_outer` | Outer Circularity | min | Low circularity = irregular shape. |
| `circularity_inner` | Inner Circularity | min | Low circularity = irregular shape. |
| `outer_radial_std` | Outer Radial Std | max | High std = bumpy/uneven outer edge. |
| `inner_radial_std` | Inner Radial Std | max | High std = bumpy/uneven inner edge. |

### REJECT metrics (12) — structural, unfixable

| Key | Display Name | Type |
|-----|-------------|------|
| `ring_thickness` | Ring Thickness (fitted) | range |
| `mean_thickness` | Mean Wall Thickness | range |
| `min_thickness` | Min Wall Thickness | min |
| `max_thickness` | Max Wall Thickness | max |
| `thickness_range` | Thickness Range | max |
| `thickness_ratio` | Thickness Ratio | max |
| `thickness_std` | Thickness Std Dev | max |
| `thickness_cv` | Thickness CV | max |
| `center_dist` | Center Distance | max |
| `eccentricity_pct` | Eccentricity | max |
| `annular_area_k` | Annular Area (×1000) | range |
| `edge_clearance` | Edge Clearance | min |

All REJECT thresholds are automatically widened by **+10 %** at load time
(lo × 0.9, hi × 1.1) to reduce false rejects near boundary values.

---

## Threshold Computation

### 1. σ-based baseline (`compute_thresholds`)

For each metric, a reference population of known-good o-rings is measured.
Thresholds are derived from **mean ± 2.5σ**:

- **range** metrics → `[mean − 2.5σ,  mean + 2.5σ]`
- **max** metrics → `[0,  mean + 2.5σ]`
- **min** metrics → `[mean − 2.5σ,  ∞]`

Source data: `good_measurements.csv` (Model 2, 260 images) and
`model1good_measurements.csv` (Model 1, 142 images — expanded from original 18
by incorporating 124 additional samples from `rework_measurements.csv`).

### 2. Tuned thresholds (`tune_thresholds.py`)

The σ-based thresholds may still reject a small number of known-good images
near the tails. The tuning script:

1. Measures **every** known-good image.
2. For each metric, finds the worst-case value across all good images.
3. If that worst case falls outside the σ-threshold, computes the minimum
   tolerance % needed to widen the band so the image just barely passes.
4. Applies floor/ceiling rounding + epsilon to avoid boundary issues.
5. Saves the result as a JSON file:

```
rework/model2_tuned_thresholds.json
rework/model1_tuned_thresholds.json
```

Each JSON entry looks like:

```json
{
  "outer_radius": {
    "lo": 642.0,
    "hi": 692.0,
    "tolerance_pct": 1.2
  }
}
```

### 3. Runtime loading (`_load_best_thresholds`)

When the GUI starts:

1. **Tuned JSON** is loaded if available (highest priority).
2. Any metrics missing from the JSON are filled from σ-based computation.
3. All **REJECT** thresholds are then widened by **10 %**
   (lo decreased by 10 %, hi increased by 10 %) to provide extra margin
   against false rejects on borderline samples.
4. If no tuned JSON exists and no CSV is found, hard-coded defaults are used.

Thresholds are shown in the table and are **editable** via spinboxes in the UI.

---

## Verdict Logic

The verdict is determined in `_evaluate()` after every analysis:

```
1. Read current threshold values from the UI spinboxes.
2. For each of the 18 metrics:
     a. Compare measured value vs [lo, hi] per its type.
     b. Special rules:
        • outer_radius — only fail if value > hi  (too large → rework by trimming)
        • inner_radius — only fail if value < lo   (too small → excess material inward)
     c. If fail:  bucket into "rework_fails" or "reject_fails" by category.
3. Determine overall verdict (priority order):
     • If ANY rework metric fails  →  REWORK  (shape fixable by trimming)
     • Else if ANY reject metric fails  →  REJECT  (structural, unfixable)
     • Else  →  PASS
```

REWORK is checked **first** because it is the more actionable outcome —
the part can be salvaged. If both rework and reject metrics fail, the verdict
is still REWORK (the reject failures are shown as "also failing" in the
detail banner).

---

## Resolution Normalization

All thresholds are calibrated at the **reference resolution** of **2448 × 2048**
(original camera images). When a different-resolution image is loaded (e.g. a
2×2 binned 1224 × 1024 image), measurements are automatically normalized so the
same thresholds still apply.

### How it works

1. A **scale factor** is computed:  `scale = max(img_w, img_h) / max(2448, 2048)`.
2. Each metric has a declared scale type:

   | Scale type | Metrics | Normalization |
   |------------|---------|---------------|
   | **linear** | radii, thicknesses, distances, std devs, clearance | `value / scale` |
   | **area** | `annular_area_k` | `value / scale²` |
   | **none** | circularity, thickness_ratio, thickness_cv, eccentricity_pct | unchanged |

3. The normalized values are compared against thresholds. The table in the UI
   shows normalized values so they're directly comparable to the reference
   thresholds.

### Examples

| Input image | scale | outer_radius raw | normalized |
|-------------|-------|-----------------|-----------|
| 2448 × 2048 (original) | 1.00 | 665 px | 665 px |
| 1224 × 1024 (2×2 binned) | 0.50 | 332 px | 665 px |
| 720 × 720 (binned+crop) | 0.29 | 195 px | 665 px |

### Limitations

- **Cropped images** (original resolution, but FOV trimmed to o-ring region):
  contour-based measurements (radii, thickness, circularity) are unchanged
  since they depend on contour geometry, not image size.  `edge_clearance` will
  be incorrect because it measures distance to the image border, which is now
  the crop boundary instead of the sensor edge.  All other metrics work fine.
- The auto-detection assumes the aspect ratio stays roughly the same.  An
  image that is both cropped and downsampled will get an approximate scale
  based on the larger dimension.

---

## File Reference

| File | Purpose |
|------|---------|
| `inspection_gui.py` | Main PySide6 inspection GUI with 3-way verdict |
| `tune_thresholds.py` | Computes tuned thresholds from all known-good images |
| `compute_good_model2_stats.py` | Measures 260 good Model 2 images → CSV |
| `compute_good_model1_stats.py` | Measures 18 selected good Model 1 images → CSV |
| `good_measurements.csv` | Model 2 per-image measurements |
| `model1good_measurements.csv` | Model 1 per-image measurements |
| `model2_tuned_thresholds.json` | Tuned thresholds for Model 2 |
| `model1_tuned_thresholds.json` | Tuned thresholds for Model 1 |
| `update_model1_stats.py` | Combines old + new CSVs, re-measures all 142 Model 1 images → `model1good_measurements.csv` |
| `rework_review_gui.py` | Older review GUI (superseded by inspection_gui) |
| `identify_rework_samples.py` | Script to flag rework candidates |
| `rework_measurements.csv` | 142 Model 1 good-sample measurements from production (input to update_model1_stats) |
---

## Threshold Tuning Workflow

When new good samples are collected and existing thresholds are too tight:

```bash
# Step 1 — Place new measurement data
#   Add raw measurements CSV to workspace root (rework_measurements.csv)
#   Images must exist in  Original Data/model1good/

# Step 2 — Re-measure all good images with full 18-metric pipeline
conda activate dl
python rework/update_model1_stats.py
#   → rewrites  rework/model1good_measurements.csv      (per-image)
#   → rewrites  rework/model1good_measurements_stats.csv (statistics)

# Step 3 — Recompute tuned thresholds
python rework/tune_thresholds.py
#   → rewrites  rework/model1_tuned_thresholds.json
#   → rewrites  rework/model2_tuned_thresholds.json
#   → Verifies all good images pass (should print "142/142 ✓")

# Step 4 — Launch GUI and test
python rework/inspection_gui.py
#   The GUI loads the updated JSON automatically.
```

---

## Threshold Tuning History

### 2026-02-18 — Model 1 expanded to 142 good samples

**Problem**: Original Model 1 thresholds were calibrated on only 18 images.
Many legitimate good samples were being flagged as REWORK or REJECT due to
overly tight thresholds.

**Solution**: Collected 124 additional production-verified good samples
(`rework_measurements.csv`), combined with the original 18, and re-tuned.

| Metric | Old lo | Old hi | New lo | New hi | Note |
|--------|--------|--------|--------|--------|------|
| outer_radius | 605.83 | 622.17 | 602.08 | 636.35 | Range widened both sides |
| inner_radius | 321.97 | 333.17 | 321.32 | 335.62 | Range widened both sides |
| circularity_outer | 0.84 | — | 0.46 | — | Much more variation observed |
| circularity_inner | 0.77 | — | 0.74 | — | Slightly widened |
| outer_radial_std | — | 25.36 | — | 143.11 | Large tol% (236%) — some images very irregular |
| inner_radial_std | — | 21.56 | — | 21.95 | Minimal change |
| ring_thickness | 281.26 / 291.23 | — | 276.80 / 304.79 | — | Wider range |
| mean_thickness | 278.62 / 288.01 | — | 276.84 / 289.77 | — | Minimal change |
| min_thickness | 270.74 | — | 147.91 | — | Large tol% (40%) — some images have thin spots |
| max_thickness | — | 300.80 | — | 313.66 | Accommodates thicker rings |
| thickness_range | — | 22.58 | — | 143.63 | Large tol% (184%) |
| thickness_ratio | — | 1.08 | — | 1.98 | Large tol% (55%) |
| thickness_std | — | 4.19 | — | 26.66 | Large tol% (173%) |
| thickness_cv | — | 1.48 | — | 9.62 | Large tol% (176%) |
| center_dist | — | 7.95 | — | 29.20 | Large tol% (61%) |
| eccentricity_pct | — | 1.68 | — | 6.08 | Large tol% (60%) |
| annular_area_k | 773.07 / 861.21 | — | 760.12 / 950.00 | — | Wider range |
| edge_clearance | 0 | — | 0 | — | No change |

**Result**: 142/142 good images now pass all metrics. The GUI's +10% REJECT
widening provides additional margin.