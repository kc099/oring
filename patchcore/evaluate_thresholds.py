"""
Evaluate PatchCore models on their training crops to determine thresholds.

Runs each trained PatchCore model on all its cropped images, records per-image
anomaly scores, and computes summary statistics (mean, std, min, max).

Since PatchCore is trained on "good" images only, scores on good images
represent the normal distribution. A threshold can be set as:
    threshold = mean + k * std   (e.g. k=2 or k=3)

Outputs:
    results_cropped/thresholds_model1.csv   — per-image scores
    results_cropped/thresholds_model2.csv
    results_cropped/threshold_summary.csv   — aggregate stats

Usage:
    python evaluate_thresholds.py
"""

import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from patchcore.config import ModelConfig, RESIZE_SIZE, CENTER_CROP_SIZE
from patchcore.patchcore_model import PatchCore, aggregate_features
from patchcore.dataset import get_transform

RESULTS_DIR = SCRIPT_DIR / "results_cropped"

MODELS = {
    "model1": {
        "pkl": RESULTS_DIR / "model1_cropped_resnet50" / "model1_cropped_resnet50_patchcore.pkl",
        "crops_dir": RESULTS_DIR / "patchcore-model1-crops",
    },
    "model2": {
        "pkl": RESULTS_DIR / "model2_cropped_resnet50" / "model2_cropped_resnet50_patchcore.pkl",
        "crops_dir": RESULTS_DIR / "patchcore-model2-crops",
    },
}

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_patchcore(pkl_path: Path) -> PatchCore:
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)
    backbone = state.get("backbone", "resnet50")
    cfg = ModelConfig(
        name=pkl_path.stem,
        backbone=backbone,
        resize=state.get("resize", RESIZE_SIZE),
        center_crop=state.get("center_crop", CENTER_CROP_SIZE),
        n_neighbors=state.get("n_neighbors", 9),
        output_dir=pkl_path.parent,
        batch_size=1,
    )
    model = PatchCore(cfg)
    model.load(pkl_path)
    return model


def infer_single(model: PatchCore, img_bgr: np.ndarray):
    """Run PatchCore on a single BGR image. Returns (score, anomaly_map)."""
    tfm = get_transform(model.cfg.resize, model.cfg.center_crop)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = tfm(img_rgb).unsqueeze(0).to(model.device)

    with torch.no_grad():
        feat_maps = model.extractor(tensor)
        patches = aggregate_features(feat_maps)

        model._prepare_bank()
        bank = model._bank_gpu
        bank_sq = model._bank_sq_norms
        k = model.cfg.n_neighbors

        q = patches.to(model.device, dtype=torch.float16)
        q_sq = (q ** 2).sum(dim=1, keepdim=True)

        chunk_size = 2048
        min_dists = []
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            q_chunk = q[start:end]
            q_sq_chunk = q_sq[start:end]
            dist_sq = q_sq_chunk + bank_sq.unsqueeze(0) - 2.0 * (q_chunk @ bank.t())
            dist_sq.clamp_(min=0.0)
            topk_sq, _ = dist_sq.topk(k, dim=1, largest=False)
            min_dists.append(topk_sq.sqrt().mean(dim=1))

        min_dists = torch.cat(min_dists, dim=0).float().cpu().numpy()

    H, W = model.spatial_shape
    amap = min_dists.reshape(H, W)
    amap = gaussian_filter(amap, sigma=4)
    return float(amap.max()), amap


def evaluate_model(model_key: str, info: dict) -> list:
    """Run inference on all crops, return list of (filename, score) tuples."""
    pkl_path = info["pkl"]
    crops_dir = info["crops_dir"]

    if not pkl_path.exists():
        print(f"  WARNING: model not found: {pkl_path}")
        return []
    if not crops_dir.exists():
        print(f"  WARNING: crops dir not found: {crops_dir}")
        return []

    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_key}")
    print(f"  Model : {pkl_path.name}")
    print(f"  Crops : {crops_dir}")
    print(f"{'='*70}")

    model = load_patchcore(pkl_path)

    image_files = sorted(
        p for p in crops_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"  Images: {len(image_files)}")

    results = []
    t0 = time.time()
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ERROR reading: {img_path.name}")
            continue

        score, _ = infer_single(model, img)
        results.append((img_path.name, score))

        if (i + 1) % 20 == 0 or i == len(image_files) - 1:
            print(f"    [{i+1}/{len(image_files)}] last score: {score:.4f}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(image_files):.2f}s per image)")

    # Release GPU memory
    model.release_bank()
    del model
    torch.cuda.empty_cache()

    return results


def main():
    summary_rows = []

    for model_key, info in MODELS.items():
        results = evaluate_model(model_key, info)
        if not results:
            continue

        scores = [s for _, s in results]
        scores_arr = np.array(scores)

        mean_s = float(scores_arr.mean())
        std_s = float(scores_arr.std())
        min_s = float(scores_arr.min())
        max_s = float(scores_arr.max())
        median_s = float(np.median(scores_arr))
        p95 = float(np.percentile(scores_arr, 95))
        p99 = float(np.percentile(scores_arr, 99))

        # Suggested thresholds
        thresh_2sigma = mean_s + 2 * std_s
        thresh_3sigma = mean_s + 3 * std_s

        # Save per-image CSV
        csv_path = RESULTS_DIR / f"thresholds_{model_key}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "anomaly_score"])
            for fname, score in results:
                writer.writerow([fname, f"{score:.6f}"])
        print(f"  Per-image scores saved: {csv_path}")

        # Print summary
        print(f"\n  --- {model_key} Summary ---")
        print(f"  Images    : {len(results)}")
        print(f"  Mean      : {mean_s:.4f}")
        print(f"  Std       : {std_s:.4f}")
        print(f"  Min       : {min_s:.4f}")
        print(f"  Max       : {max_s:.4f}")
        print(f"  Median    : {median_s:.4f}")
        print(f"  P95       : {p95:.4f}")
        print(f"  P99       : {p99:.4f}")
        print(f"  Threshold (mean+2σ) : {thresh_2sigma:.4f}")
        print(f"  Threshold (mean+3σ) : {thresh_3sigma:.4f}")

        summary_rows.append({
            "model": model_key,
            "num_images": len(results),
            "mean": mean_s,
            "std": std_s,
            "min": min_s,
            "max": max_s,
            "median": median_s,
            "p95": p95,
            "p99": p99,
            "threshold_2sigma": thresh_2sigma,
            "threshold_3sigma": thresh_3sigma,
        })

    # Save summary CSV
    summary_path = RESULTS_DIR / "threshold_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fields = ["model", "num_images", "mean", "std", "min", "max",
                  "median", "p95", "p99", "threshold_2sigma", "threshold_3sigma"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            # Round floats for readability
            for k in row:
                if isinstance(row[k], float):
                    row[k] = round(row[k], 6)
            writer.writerow(row)
    print(f"\n  Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("THRESHOLD EVALUATION COMPLETE")
    print("=" * 70)
    for row in summary_rows:
        print(f"  {row['model']}:  mean={row['mean']:.4f}  "
              f"min={row['min']:.4f}  max={row['max']:.4f}  "
              f"threshold(2σ)={row['threshold_2sigma']:.4f}  "
              f"threshold(3σ)={row['threshold_3sigma']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
