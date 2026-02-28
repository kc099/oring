"""
Coreset Ablation Study
======================
Sub-sample existing PatchCore memory banks to smaller sizes and
re-evaluate AUROC.  No re-training needed — just random subsampling
of the already-built coreset.

This answers: "How small can the memory bank be while keeping AUROC=1.0?"

Usage:
    python -m patchcore.ablation_coreset

Author: GitHub Copilot
Date:   February 28, 2026
"""

import pickle
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import get_model1_config, get_model2_config, ModelConfig, WORKSPACE
from .dataset import get_test_loaders
from .patchcore_model import PatchCore

# Ratios to test (fraction of full coreset)
# Full coreset is already 25% of all patches — these are fractions of THAT
SUBSAMPLE_FRACTIONS = [1.0, 0.50, 0.25, 0.125, 0.10, 0.05, 0.025, 0.01, 0.005]


def run_ablation(cfg: ModelConfig, pkl_path: Path):
    """Load full memory bank, subsample to different sizes, evaluate each."""
    print(f"\n{'='*70}")
    print(f"  Ablation: {cfg.name}")
    print(f"  Model: {pkl_path}")
    print(f"{'='*70}")

    # Load full model
    model = PatchCore(cfg)
    model.load(pkl_path)
    full_bank = model.memory_bank.copy()
    N_full = full_bank.shape[0]
    print(f"  Full memory bank: {N_full:,} × {full_bank.shape[1]}")

    # Get test loaders
    test_loaders = get_test_loaders(cfg)

    results = []
    rng = np.random.RandomState(42)

    for frac in SUBSAMPLE_FRACTIONS:
        M = max(1, int(N_full * frac))
        print(f"\n  ── Bank size: {M:,} ({frac:.1%} of {N_full:,}) ──")

        # Subsample and invalidate GPU cache
        model.release_bank()
        if frac < 1.0:
            indices = rng.choice(N_full, size=M, replace=False)
            model.memory_bank = full_bank[indices]
        else:
            model.memory_bank = full_bank

        # Evaluate
        all_scores = []
        all_labels = []
        score_stats = {}
        t0 = time.time()

        for label_name, loader in test_loaders:
            res = model.evaluate(loader, label_name=label_name)
            all_scores.append(res["scores"])
            all_labels.append(res["labels"])
            score_stats[label_name] = {
                "n": res["n_samples"],
                "mean": float(res["scores"].mean()),
                "std": float(res["scores"].std()),
                "min": float(res["scores"].min()),
                "max": float(res["scores"].max()),
            }

        elapsed = time.time() - t0
        combined_scores = np.concatenate(all_scores)
        combined_labels = np.concatenate(all_labels)

        if len(np.unique(combined_labels)) > 1:
            auroc = roc_auc_score(combined_labels, combined_scores)
        else:
            auroc = None

        # Gap between good max and defect min
        good_scores = combined_scores[combined_labels == 0]
        defect_scores = combined_scores[combined_labels == 1]
        gap = defect_scores.min() - good_scores.max() if len(defect_scores) > 0 else 0

        entry = {
            "fraction": frac,
            "bank_size": M,
            "auroc": auroc,
            "good_max": float(good_scores.max()),
            "defect_min": float(defect_scores.min()) if len(defect_scores) > 0 else None,
            "gap": float(gap),
            "eval_time_s": round(elapsed, 1),
            "bank_mb": round(M * full_bank.shape[1] * 4 / 1e6, 1),
            "per_label": score_stats,
        }
        results.append(entry)

        auroc_str = f"{auroc:.4f}" if auroc else "N/A"
        print(f"    AUROC={auroc_str}  gap={gap:.2f}  "
              f"bank={M:,} ({entry['bank_mb']}MB)  time={elapsed:.1f}s")

    # Restore full bank
    model.memory_bank = full_bank

    # Summary table
    print(f"\n  {'─'*70}")
    print(f"  {'Fraction':>10}  {'Bank':>8}  {'MB':>6}  {'AUROC':>8}  "
          f"{'Good max':>10}  {'Def min':>10}  {'Gap':>8}  {'Time':>6}")
    print(f"  {'─'*70}")
    for r in results:
        auroc_str = f"{r['auroc']:.4f}" if r['auroc'] else "N/A"
        def_min = f"{r['defect_min']:.4f}" if r['defect_min'] else "N/A"
        print(f"  {r['fraction']:>10.1%}  {r['bank_size']:>8,}  {r['bank_mb']:>6.1f}  "
              f"{auroc_str:>8}  {r['good_max']:>10.4f}  {def_min:>10}  "
              f"{r['gap']:>8.2f}  {r['eval_time_s']:>5.1f}s")
    print(f"  {'─'*70}")

    # Save
    out_path = cfg.output_dir / f"{cfg.name}_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")

    return results


def main():
    configs_and_pkls = []

    # Model 1
    cfg1 = get_model1_config("resnet50")
    cfg1.batch_size = 8
    cfg1.num_workers = 4
    pkl1 = cfg1.output_dir / f"{cfg1.name}_patchcore.pkl"
    if pkl1.exists():
        configs_and_pkls.append((cfg1, pkl1))

    # Model 2
    cfg2 = get_model2_config("resnet50")
    cfg2.batch_size = 8
    cfg2.num_workers = 4
    pkl2 = cfg2.output_dir / f"{cfg2.name}_patchcore.pkl"
    if pkl2.exists():
        configs_and_pkls.append((cfg2, pkl2))

    if not configs_and_pkls:
        print("No trained models found! Train first with: python -m patchcore.train")
        return

    print(f"\n{'#'*70}")
    print(f"  Coreset Ablation Study")
    print(f"  Testing fractions: {SUBSAMPLE_FRACTIONS}")
    print(f"  Models: {[c.name for c, _ in configs_and_pkls]}")
    print(f"{'#'*70}")

    for cfg, pkl in configs_and_pkls:
        run_ablation(cfg, pkl)


if __name__ == "__main__":
    main()
