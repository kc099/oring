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
`model1good_measurements.csv` (Model 1, 18 selected images).

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
| `rework_review_gui.py` | Older review GUI (superseded by inspection_gui) |
| `identify_rework_samples.py` | Script to flag rework candidates |
| `rework_measurements.csv` | Measurements from rework candidate review |
