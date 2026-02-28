"""
Quick FP16 ONNX verification — run both models on all test images,
print score distributions, AUROC, and diagnose threshold issues.

Usage:
    cd "F:\\standard elastomers"
    conda activate dl
    python onnx_export/verify_fp16.py
"""

import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

import torch
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort
from sklearn.metrics import roc_auc_score

WORKSPACE = Path(__file__).resolve().parent.parent
ONNX_DIR = Path(__file__).resolve().parent
BINNED = WORKSPACE / "binned"

MODELS = {
    "model1_resnet50": {
        "good":    (BINNED / "model1good", 0),
        "defect":  (BINNED / "model1defect", 1),
        "defect2": (BINNED / "model1defect2", 1),
        "defect3": (BINNED / "model1defect3", 1),
    },
    "model2_resnet50": {
        "good":   (BINNED / "good", 0),
        "notok":  (BINNED / "notok", 1),
        "notok2": (BINNED / "notok2", 1),
        "notok3": (BINNED / "notok3", 1),
    },
}


def preprocess(image_path: Path, resize=660, crop=640) -> np.ndarray:
    """Match Python training pipeline: resize → center-crop → RGB [0,1] → NCHW."""
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    margin = (resize - crop) // 2
    img = img[margin:margin + crop, margin:margin + crop]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis]


def run_model(model_name: str, precision: str = "fp16"):
    suffix = "_fp16" if precision == "fp16" else ""
    onnx_path = ONNX_DIR / f"patchcore_{model_name}{suffix}.onnx"
    json_path = ONNX_DIR / f"patchcore_{model_name}.json"

    if not onnx_path.exists():
        print(f"  ✗ Not found: {onnx_path}")
        return

    # Load metadata
    meta = {}
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
    threshold = meta.get("recommended_threshold", 15.0)
    resize = meta.get("resize", 660)
    crop = meta.get("center_crop", 640)

    print(f"\n{'='*70}")
    print(f"  {model_name} — {precision.upper()} ONNX")
    print(f"  File: {onnx_path.name}  ({onnx_path.stat().st_size/1e6:.0f} MB)")
    print(f"  Threshold from JSON: {threshold}")
    print(f"{'='*70}")

    # Create GPU session
    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
    active = sess.get_providers()
    print(f"  Provider: {active[0]}")

    test_dirs = MODELS.get(model_name, {})
    all_scores = []
    all_labels = []

    print(f"\n  {'Set':<10} {'N':>4} {'Mean':>8} {'Std':>7} {'Min':>8} {'Max':>8} "
          f"{'<Thr':>5} {'>Thr':>5}")
    print(f"  {'-'*60}")

    for set_name, (set_dir, label) in test_dirs.items():
        images = sorted(set_dir.glob("*.bmp"))
        if not images:
            continue

        scores = []
        for img_path in images:
            inp = preprocess(img_path, resize, crop)
            out = sess.run(None, {"image": inp})
            scores.append(out[0][0])

        scores = np.array(scores)
        all_scores.extend(scores.tolist())
        all_labels.extend([label] * len(scores))

        below = (scores <= threshold).sum()
        above = (scores > threshold).sum()
        print(f"  {set_name:<10} {len(scores):>4} {scores.mean():>8.2f} "
              f"{scores.std():>7.2f} {scores.min():>8.2f} {scores.max():>8.2f} "
              f"{below:>5} {above:>5}")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    good_mask = all_labels == 0
    defect_mask = all_labels == 1
    auroc = roc_auc_score(all_labels, all_scores)
    good_max = all_scores[good_mask].max()
    defect_min = all_scores[defect_mask].min()
    gap = defect_min - good_max

    # Compute correct classifications at recommended threshold
    pred = (all_scores > threshold).astype(int)
    correct = (pred == all_labels).sum()
    accuracy = correct / len(all_labels) * 100

    print(f"\n  AUROC:          {auroc:.4f}")
    print(f"  Good max:       {good_max:.4f}")
    print(f"  Defect min:     {defect_min:.4f}")
    print(f"  Gap:            {gap:.4f}")
    print(f"  Threshold:      {threshold}")
    print(f"  Accuracy @thr:  {correct}/{len(all_labels)} ({accuracy:.1f}%)")

    # Check if threshold is correct
    if good_max < threshold < defect_min:
        print(f"  ✓ Threshold {threshold} is in the gap — perfect separation")
    else:
        print(f"  ⚠ THRESHOLD ISSUE:")
        if threshold <= good_max:
            n_fp = (all_scores[good_mask] > threshold).sum()
            print(f"    Threshold too LOW — {n_fp} good samples above threshold (false rejects)")
        if threshold >= defect_min:
            n_fn = (all_scores[defect_mask] <= threshold).sum()
            print(f"    Threshold too HIGH — {n_fn} defect samples below threshold (missed)")
        optimal = (good_max + defect_min) / 2
        print(f"    Optimal threshold: {optimal:.2f}")

    del sess
    return {"auroc": auroc, "good_max": good_max, "defect_min": defect_min,
            "gap": gap, "threshold": threshold, "accuracy": accuracy}


def main():
    print("PatchCore FP16 ONNX Verification")
    print("=" * 70)

    for model_name in ["model1_resnet50", "model2_resnet50"]:
        # Run FP16
        run_model(model_name, "fp16")

    print(f"\n{'='*70}")
    print("Done.")
    os._exit(0)


if __name__ == "__main__":
    main()
