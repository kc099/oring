"""
PatchCore Training Script
=========================
Train PatchCore anomaly detection models for O-ring inspection.

Trains all 4 combinations by default:
    - Model 1 × ResNet-50
    - Model 1 × ResNet-101
    - Model 2 × ResNet-50
    - Model 2 × ResNet-101

Usage:
    # Train all 4 models
    python -m patchcore.train

    # Train specific model + backbone
    python -m patchcore.train --model model1 --backbone resnet50
    python -m patchcore.train --model model2 --backbone resnet101

    # Train both backbones for one model
    python -m patchcore.train --model model1

    # Custom coreset ratio
    python -m patchcore.train --coreset 0.10

Author: GitHub Copilot
Date:   February 27, 2026
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import (
    get_model1_config,
    get_model2_config,
    get_all_configs,
    ModelConfig,
)
from .dataset import get_train_loader, get_test_loaders
from .patchcore_model import PatchCore


def train_and_evaluate(cfg: ModelConfig) -> dict:
    """Train PatchCore on good images and evaluate on all test sets."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Build model & fit ──
    model = PatchCore(cfg)
    train_loader = get_train_loader(cfg)
    model.fit(train_loader)
    model.save()

    # ── Evaluate on each test set ──
    print(f"\n  Evaluating {cfg.name} ...")
    test_loaders = get_test_loaders(cfg)
    all_results = {}
    all_scores = []
    all_labels = []

    for label_name, loader in test_loaders:
        res = model.evaluate(loader, label_name=label_name)
        all_results[label_name] = {
            "n_samples": res["n_samples"],
            "auroc": res.get("auroc"),
            "score_mean": float(res["scores"].mean()),
            "score_std": float(res["scores"].std()),
            "score_min": float(res["scores"].min()),
            "score_max": float(res["scores"].max()),
        }
        all_scores.append(res["scores"])
        all_labels.append(res["labels"])

    # Combined AUROC across all test sets
    combined_scores = np.concatenate(all_scores)
    combined_labels = np.concatenate(all_labels)
    if len(np.unique(combined_labels)) > 1:
        from sklearn.metrics import roc_auc_score
        combined_auroc = roc_auc_score(combined_labels, combined_scores)
        print(f"\n  ** Combined AUROC ({cfg.name}): {combined_auroc:.4f} **")
        all_results["combined_auroc"] = combined_auroc
    else:
        all_results["combined_auroc"] = None

    elapsed = time.time() - t0
    all_results["training_time_s"] = round(elapsed, 1)
    all_results["backbone"] = cfg.backbone
    all_results["coreset_ratio"] = cfg.coreset_ratio
    all_results["memory_bank_size"] = int(model.memory_bank.shape[0]) if model.memory_bank is not None else 0

    # Save evaluation results
    results_path = cfg.output_dir / f"{cfg.name}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {results_path}")
    print(f"  Total time: {elapsed:.1f}s\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detection for O-ring inspection"
    )
    parser.add_argument(
        "--model",
        choices=["model1", "model2", "all"],
        default="all",
        help="Which O-ring model to train (default: all)",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101", "both"],
        default="both",
        help="Backbone architecture (default: both)",
    )
    parser.add_argument(
        "--coreset",
        type=float,
        default=0.25,
        help="Coreset sampling ratio (default: 0.25)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size after center crop (default: 640)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    args = parser.parse_args()

    # Build list of configs to train
    configs = []
    backbones = ["resnet50", "resnet101"] if args.backbone == "both" else [args.backbone]

    for bb in backbones:
        if args.model in ("model1", "all"):
            configs.append(get_model1_config(bb))
        if args.model in ("model2", "all"):
            configs.append(get_model2_config(bb))

    # Override coreset / batch / workers / image size
    for cfg in configs:
        cfg.coreset_ratio = args.coreset
        cfg.batch_size = args.batch_size
        cfg.num_workers = args.workers
        cfg.center_crop = args.image_size
        cfg.resize = args.image_size + 20  # small margin for center crop

    print(f"\n{'#'*60}")
    print(f"  PatchCore Training Pipeline")
    print(f"  Models to train: {len(configs)}")
    for cfg in configs:
        print(f"    - {cfg.name}  (backbone={cfg.backbone}, coreset={cfg.coreset_ratio}, "
              f"image={cfg.center_crop}×{cfg.center_crop}, batch={cfg.batch_size})")
    print(f"{'#'*60}\n")

    summary = {}
    for cfg in configs:
        results = train_and_evaluate(cfg)
        summary[cfg.name] = results

    # Final summary
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, res in summary.items():
        auroc_str = f"{res['combined_auroc']:.4f}" if res.get("combined_auroc") else "N/A"
        print(f"  {name:30s}  AUROC={auroc_str}  "
              f"bank={res['memory_bank_size']:,}  "
              f"time={res['training_time_s']}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
