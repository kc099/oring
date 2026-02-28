"""
FP32 vs FP16 ONNX Ablation Study for PatchCore.

Converts the exported FP32 ONNX model to FP16 and compares:
  - Score accuracy (per-image scores on all test sets)
  - Anomaly map fidelity (max/mean difference)
  - AUROC preservation (still 1.0?)
  - GPU inference latency
  - ONNX file size

Usage:
    cd "F:\\standard elastomers"
    conda activate dl
    python onnx_export/ablation_fp16.py --model model1
    python onnx_export/ablation_fp16.py --model model2
    python onnx_export/ablation_fp16.py --all

Author: GitHub Copilot
Date:   February 28, 2026
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure cuDNN DLLs are on PATH for onnxruntime-gpu
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import onnx
from onnxconverter_common import float16
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

WORKSPACE = Path(__file__).resolve().parent.parent
ONNX_DIR = Path(__file__).resolve().parent
BINNED = WORKSPACE / "binned"

# ImageNet normalization is inside the model, so we just need [0,1] RGB.

# ─── Test image sets ─────────────────────────────────────────────────────

MODEL_TEST_DIRS = {
    "model1": {
        "good":    (BINNED / "model1good", 0),
        "defect":  (BINNED / "model1defect", 1),
        "defect2": (BINNED / "model1defect2", 1),
        "defect3": (BINNED / "model1defect3", 1),
    },
    "model2": {
        "good":   (BINNED / "good", 0),
        "notok":  (BINNED / "notok", 1),
        "notok2": (BINNED / "notok2", 1),
        "notok3": (BINNED / "notok3", 1),
    },
}


def load_and_preprocess(image_path: Path, resize: int = 660,
                        crop: int = 640) -> np.ndarray:
    """Load BMP → resize → center-crop → RGB [0,1] → NCHW float32."""
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    margin = (resize - crop) // 2
    img = img[margin:margin + crop, margin:margin + crop]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)


def convert_to_fp16(fp32_path: Path, fp16_path: Path) -> Path:
    """Convert FP32 ONNX model to FP16."""
    print(f"  Converting FP32 → FP16...")
    t0 = time.perf_counter()
    model_fp32 = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,  # Keep inputs/outputs as float32 for compatibility
    )
    onnx.save(model_fp16, str(fp16_path))
    elapsed = time.perf_counter() - t0

    size_fp32 = fp32_path.stat().st_size / (1024 * 1024)
    size_fp16 = fp16_path.stat().st_size / (1024 * 1024)
    print(f"    FP32: {size_fp32:.1f} MB")
    print(f"    FP16: {size_fp16:.1f} MB  ({size_fp16/size_fp32:.1%} of FP32)")
    print(f"    Conversion: {elapsed:.1f}s")
    return fp16_path


def create_session(onnx_path: Path) -> ort.InferenceSession:
    """Create ORT session on GPU only. Fails if CUDA is not available."""
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDA GPU not available for ONNX Runtime. "
            "Install onnxruntime-gpu and ensure CUDA 12 + cuDNN 9 are on PATH."
        )
    return sess


def run_ablation(model_name: str, backbone: str = "resnet50"):
    """Run FP32 vs FP16 comparison on all test images."""
    fp32_path = ONNX_DIR / f"patchcore_{model_name}_{backbone}.onnx"
    fp16_path = ONNX_DIR / f"patchcore_{model_name}_{backbone}_fp16.onnx"

    if not fp32_path.exists():
        print(f"  ✗ FP32 model not found: {fp32_path}")
        return

    print(f"\n{'='*70}")
    print(f"  FP32 vs FP16 Ablation: {model_name}_{backbone}")
    print(f"{'='*70}")

    # 1. Convert to FP16
    convert_to_fp16(fp32_path, fp16_path)

    # 2. Load both sessions (GPU)
    print(f"\n  Loading ORT sessions (GPU)...")
    sess_fp32 = create_session(fp32_path)
    sess_fp16 = create_session(fp16_path)

    active_fp32 = sess_fp32.get_providers()
    active_fp16 = sess_fp16.get_providers()
    print(f"    FP32 providers: {active_fp32}")
    print(f"    FP16 providers: {active_fp16}")

    # 3. Load metadata for preprocessing params
    json_path = ONNX_DIR / f"patchcore_{model_name}_{backbone}.json"
    resize, crop = 660, 640
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        resize = meta.get("resize", 660)
        crop = meta.get("center_crop", 640)

    # 4. Run on all test sets
    test_dirs = MODEL_TEST_DIRS.get(model_name, {})
    all_scores_fp32 = []
    all_scores_fp16 = []
    all_labels = []
    score_diffs = []
    map_diffs = []

    print(f"\n  Running inference on all test sets...")
    print(f"  {'Set':<10} {'N':>4}  {'FP32 mean':>10} {'FP16 mean':>10} "
          f"{'Score Δ max':>12} {'Map Δ max':>10}")
    print(f"  {'-'*60}")

    for set_name, (set_dir, label) in test_dirs.items():
        images = sorted(set_dir.glob("*.bmp"))
        if not images:
            continue

        set_scores_fp32 = []
        set_scores_fp16 = []
        set_score_diffs = []
        set_map_diffs = []

        for img_path in images:
            img_np = load_and_preprocess(img_path, resize, crop)

            out_fp32 = sess_fp32.run(None, {"image": img_np})
            out_fp16 = sess_fp16.run(None, {"image": img_np})

            s32 = out_fp32[0][0]
            s16 = out_fp16[0][0]
            m32 = out_fp32[1]
            m16 = out_fp16[1]

            set_scores_fp32.append(s32)
            set_scores_fp16.append(s16)
            set_score_diffs.append(abs(s32 - s16))
            set_map_diffs.append(np.abs(m32 - m16).max())

            all_scores_fp32.append(s32)
            all_scores_fp16.append(s16)
            all_labels.append(label)

        score_diffs.extend(set_score_diffs)
        map_diffs.extend(set_map_diffs)

        print(f"  {set_name:<10} {len(images):>4}  "
              f"{np.mean(set_scores_fp32):>10.4f} {np.mean(set_scores_fp16):>10.4f} "
              f"{np.max(set_score_diffs):>12.4f} {np.max(set_map_diffs):>10.4f}")

    # 5. Aggregate metrics
    all_scores_fp32 = np.array(all_scores_fp32)
    all_scores_fp16 = np.array(all_scores_fp16)
    all_labels = np.array(all_labels)
    score_diffs = np.array(score_diffs)
    map_diffs = np.array(map_diffs)

    auroc_fp32 = roc_auc_score(all_labels, all_scores_fp32)
    auroc_fp16 = roc_auc_score(all_labels, all_scores_fp16)

    # Separation gap
    good_mask = all_labels == 0
    defect_mask = all_labels == 1
    gap_fp32 = all_scores_fp32[defect_mask].min() - all_scores_fp32[good_mask].max()
    gap_fp16 = all_scores_fp16[defect_mask].min() - all_scores_fp16[good_mask].max()

    print(f"\n  {'='*60}")
    print(f"  AGGREGATE RESULTS")
    print(f"  {'='*60}")
    print(f"  {'Metric':<30} {'FP32':>12} {'FP16':>12} {'Δ':>10}")
    print(f"  {'-'*64}")
    print(f"  {'AUROC':<30} {auroc_fp32:>12.4f} {auroc_fp16:>12.4f} "
          f"{auroc_fp16 - auroc_fp32:>+10.4f}")
    print(f"  {'Good max score':<30} {all_scores_fp32[good_mask].max():>12.4f} "
          f"{all_scores_fp16[good_mask].max():>12.4f}")
    print(f"  {'Defect min score':<30} {all_scores_fp32[defect_mask].min():>12.4f} "
          f"{all_scores_fp16[defect_mask].min():>12.4f}")
    print(f"  {'Separation gap':<30} {gap_fp32:>12.4f} {gap_fp16:>12.4f} "
          f"{gap_fp16 - gap_fp32:>+10.4f}")
    print(f"  {'Score diff (mean)':<30} {score_diffs.mean():>12.4f}")
    print(f"  {'Score diff (max)':<30} {score_diffs.max():>12.4f}")
    print(f"  {'Map diff (mean)':<30} {np.mean(map_diffs):>12.4f}")
    print(f"  {'Map diff (max)':<30} {np.max(map_diffs):>12.4f}")
    print(f"  {'ONNX size (MB)':<30} "
          f"{fp32_path.stat().st_size/1e6:>12.1f} "
          f"{fp16_path.stat().st_size/1e6:>12.1f}")

    # 6. Benchmark GPU latency
    print(f"\n  GPU Latency Benchmark (5 runs each)...")
    test_img = load_and_preprocess(
        list(list(test_dirs.values())[0][0].glob("*.bmp"))[0], resize, crop)

    # Warmup
    sess_fp32.run(None, {"image": test_img})
    sess_fp16.run(None, {"image": test_img})

    for label, sess in [("FP32", sess_fp32), ("FP16", sess_fp16)]:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            sess.run(None, {"image": test_img})
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"    {label}: mean={arr.mean():.0f}ms  min={arr.min():.0f}ms  "
              f"max={arr.max():.0f}ms")

    # 7. Verdict
    print(f"\n  {'='*60}")
    size_fp32 = fp32_path.stat().st_size / 1e6
    size_fp16 = fp16_path.stat().st_size / 1e6
    size_pct = (1 - size_fp16 / size_fp32) * 100
    if auroc_fp16 >= 1.0 and gap_fp16 > 0:
        print(f"  ✓ FP16 is SAFE — AUROC={auroc_fp16:.4f}, gap={gap_fp16:.2f}")
        print(f"    Recommendation: Use FP16 for {size_pct:.0f}% size reduction.")
    elif auroc_fp16 >= 0.999 and gap_fp16 > 0:
        print(f"  ✓ FP16 is ACCEPTABLE — AUROC={auroc_fp16:.4f}, gap={gap_fp16:.2f}")
        print(f"    Minor score drift but still perfect separation.")
    elif gap_fp16 > 0:
        print(f"  ⚠ FP16 CAUTION — AUROC={auroc_fp16:.4f}, gap={gap_fp16:.2f}")
        print(f"    Scores shifted but gap preserved. Adjust threshold.")
    else:
        print(f"  ✗ FP16 NOT RECOMMENDED — gap={gap_fp16:.4f} (overlap!)")
        print(f"    Good and defect score ranges overlap. Stick with FP32.")
    print(f"  {'='*60}")

    # Cleanup sessions explicitly
    del sess_fp32, sess_fp16

    # Save results
    results = {
        "model": f"{model_name}_{backbone}",
        "auroc_fp32": float(auroc_fp32),
        "auroc_fp16": float(auroc_fp16),
        "gap_fp32": float(gap_fp32),
        "gap_fp16": float(gap_fp16),
        "score_diff_mean": float(score_diffs.mean()),
        "score_diff_max": float(score_diffs.max()),
        "map_diff_mean": float(np.mean(map_diffs)),
        "map_diff_max": float(np.max(map_diffs)),
        "onnx_size_fp32_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "onnx_size_fp16_mb": round(fp16_path.stat().st_size / 1e6, 1),
        "good_max_fp32": float(all_scores_fp32[good_mask].max()),
        "good_max_fp16": float(all_scores_fp16[good_mask].max()),
        "defect_min_fp32": float(all_scores_fp32[defect_mask].min()),
        "defect_min_fp16": float(all_scores_fp16[defect_mask].min()),
    }
    out_json = ONNX_DIR / f"patchcore_{model_name}_{backbone}_fp16_ablation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="FP32 vs FP16 ablation for PatchCore ONNX")
    parser.add_argument("--model", type=str, default=None,
                        choices=["model1", "model2"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("PatchCore FP32 vs FP16 Ablation")
    print("=" * 70)

    if args.all or args.model is None:
        for name in ["model1", "model2"]:
            fp32 = ONNX_DIR / f"patchcore_{name}_{args.backbone}.onnx"
            if fp32.exists():
                run_ablation(name, args.backbone)
    else:
        run_ablation(args.model, args.backbone)

    print("\nDone.")
    os._exit(0)  # Avoid onnxruntime-gpu CUDA cleanup hang


if __name__ == "__main__":
    main()
