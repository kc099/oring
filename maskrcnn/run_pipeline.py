"""
End-to-End Mask R-CNN Pipeline for O-Ring Defect Detection.

Runs the full pipeline for a given o-ring model:
  1. Preprocessing  — balance dataset, COCO conversion, train/val/test split
  2. Training       — Mask R-CNN with early stopping, checkpointing
  3. Evaluation     — test set metrics (P/R/F1)
  4. Inference      — save visualized results for test images

Usage:
    python run_pipeline.py model1          # full pipeline for model1
    python run_pipeline.py model2          # full pipeline for model2
    python run_pipeline.py combined        # train on model1 + model2 merged
    python run_pipeline.py model1 --skip-preprocess   # skip step 1
    python run_pipeline.py model1 --skip-training     # skip step 2 (use existing checkpoint)
    python run_pipeline.py model1 --epochs 80 --batch-size 2

Author: GitHub Copilot
Date: February 8, 2026
"""

import argparse
import json
import sys
import time
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Ensure maskrcnn folder is on path ───────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import OUTPUT_ROOT, ORING_MODELS, TRAINING_CONFIG


def step_banner(step_num: int, title: str):
    print(f"\n{'#'*80}")
    print(f"#  STEP {step_num}: {title}")
    print(f"{'#'*80}\n")


def run_preprocessing(model_name: str):
    """Step 1: Preprocess — balance, convert to COCO, split."""
    step_banner(1, "PREPROCESSING")

    from preprocess_to_coco import preprocess_model
    from config import ORING_MODELS, TRAINING_CONFIG

    model_cfg = ORING_MODELS[model_name]

    # Clean previous output for this model
    model_output = OUTPUT_ROOT / model_name
    if model_output.exists():
        print(f"  Cleaning previous output: {model_output}")
        shutil.rmtree(model_output)

    preprocess_model(model_cfg, TRAINING_CONFIG)

    # Verify output
    ann_dir = model_output / "annotations"
    for split in ["train", "val", "test"]:
        ann_file = ann_dir / f"{split}.json"
        if ann_file.exists():
            with open(ann_file) as f:
                data = json.load(f)
            n_imgs = len(data["images"])
            n_anns = len(data["annotations"])
            print(f"  ✓ {split}: {n_imgs} images, {n_anns} annotations")
        else:
            print(f"  ✗ {split}: annotation file not found!")
            return False

    return True


def run_training(model_name: str, cfg=TRAINING_CONFIG, resume_path: str = None):
    """Step 2: Train Mask R-CNN."""
    step_banner(2, "TRAINING")

    from train import train_model
    train_model(model_name, cfg, resume_path)

    # Verify checkpoint exists
    ckpt_path = OUTPUT_ROOT / model_name / "checkpoints" / "best_model.pth"
    if ckpt_path.exists():
        print(f"\n  ✓ Best model saved: {ckpt_path}")
        return True
    else:
        print(f"\n  ✗ No best model checkpoint found!")
        return False


def run_evaluation(model_name: str, cfg=TRAINING_CONFIG):
    """Step 3: Evaluate on test set."""
    step_banner(3, "EVALUATION ON TEST SET")

    from inference import OringDefectDetector

    model_dir = OUTPUT_ROOT / model_name
    test_img_dir = model_dir / "images" / "test"
    test_ann_path = model_dir / "annotations" / "test.json"

    if not test_ann_path.exists():
        print(f"  ERROR: Test annotations not found: {test_ann_path}")
        return False

    detector = OringDefectDetector(
        model_name=model_name,
        score_threshold=cfg.score_threshold
    )

    with open(test_ann_path, 'r') as f:
        coco = json.load(f)

    # Group annotations by image_id
    img_anns = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        img_anns.setdefault(img_id, []).append(ann)

    tp = fp = fn = tn = 0
    results_list = []

    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_path = test_img_dir / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = detector.predict(image)
        has_gt = img_id in img_anns and len(img_anns[img_id]) > 0

        if has_gt and result["has_defect"]:
            tp += 1
            verdict = "TP"
        elif has_gt and not result["has_defect"]:
            fn += 1
            verdict = "FN"
        elif not has_gt and result["has_defect"]:
            fp += 1
            verdict = "FP"
        else:
            tn += 1
            verdict = "TN"

        results_list.append({
            "filename": img_info["file_name"],
            "has_gt_defect": has_gt,
            "predicted_defect": result["has_defect"],
            "num_detections": result["num_detections"],
            "max_score": float(result["scores"].max()) if result["num_detections"] > 0 else 0.0,
            "verdict": verdict
        })

    total = tp + fp + fn + tn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(total, 1)

    metrics = {
        "model": model_name,
        "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "score_threshold": cfg.score_threshold,
        "results": results_list
    }

    metrics_path = model_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {'='*50}")
    print(f"  TEST RESULTS: {model_name}")
    print(f"  {'='*50}")
    print(f"  Total: {total}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  {'='*50}")
    print(f"  Saved: {metrics_path}")

    return True


def run_inference_visualization(model_name: str, cfg=TRAINING_CONFIG):
    """Step 4: Save visualized inference results with GT comparison for test images.
    
    Organizes results into TP/FP/FN/TN folders so you can review model accuracy
    and identify images that need relabeling.
    Each image shows:
      - Green polygons = ground truth labels
      - Red/colored masks + boxes = model predictions
      - Text overlay with verdict and scores
    """
    step_banner(4, "SAVING INFERENCE VISUALIZATIONS (with GT comparison)")

    from inference import OringDefectDetector
    from utils import draw_predictions, polygon_to_mask

    model_dir = OUTPUT_ROOT / model_name
    test_img_dir = model_dir / "images" / "test"
    test_ann_path = model_dir / "annotations" / "test.json"

    if not test_ann_path.exists():
        print(f"  ERROR: Test annotations not found")
        return False

    detector = OringDefectDetector(
        model_name=model_name,
        score_threshold=cfg.score_threshold
    )

    with open(test_ann_path, 'r') as f:
        coco = json.load(f)

    # Group annotations by image_id
    img_anns = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        img_anns.setdefault(img_id, []).append(ann)

    # Create output dirs: TP, FP, FN, TN
    vis_dir = model_dir / "inference_results" / "test_visualizations"
    verdict_dirs = {}
    for verdict in ["TP", "FP", "FN", "TN"]:
        d = vis_dir / verdict
        d.mkdir(parents=True, exist_ok=True)
        verdict_dirs[verdict] = d

    total = len(coco["images"])
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for i, img_info in enumerate(coco["images"]):
        img_id = img_info["id"]
        img_path = test_img_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # Get ground truth
        gt_anns = img_anns.get(img_id, [])
        has_gt = len(gt_anns) > 0

        # Get prediction
        result = detector.predict(image)
        has_pred = result["has_defect"]

        # Determine verdict
        if has_gt and has_pred:
            verdict = "TP"
        elif not has_gt and has_pred:
            verdict = "FP"
        elif has_gt and not has_pred:
            verdict = "FN"
        else:
            verdict = "TN"
        counts[verdict] += 1

        # --- Build comparison visualization ---
        vis = image.copy()

        # Draw ground truth polygons (green outline, light green fill)
        for ann in gt_anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and len(seg) > 0:
                flat_pts = seg[0]
                pts = np.array(
                    [(int(flat_pts[j]), int(flat_pts[j+1])) for j in range(0, len(flat_pts), 2)],
                    dtype=np.int32
                )
                # Semi-transparent green fill for GT
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 0))
                vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
                # Green outline
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        # Draw predictions (red masks + boxes)
        for j in range(result["num_detections"]):
            score = result["scores"][j]
            # Red mask overlay
            if result["masks"] is not None and j < len(result["masks"]):
                mask = result["masks"][j]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h))
                overlay = vis.copy()
                overlay[mask > 0] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

            # Red bounding box
            if j < len(result["boxes"]):
                x1, y1, x2, y2 = result["boxes"][j].astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"pred {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Verdict banner at top
        banner_colors = {"TP": (0, 180, 0), "FP": (0, 0, 255), "FN": (0, 165, 255), "TN": (180, 180, 180)}
        banner_color = banner_colors[verdict]
        cv2.rectangle(vis, (0, 0), (w, 30), banner_color, -1)
        label_text = f"{verdict} | GT: {'DEFECT' if has_gt else 'GOOD'} | Pred: {'DEFECT' if has_pred else 'GOOD'}"
        if has_pred:
            label_text += f" | max_score={result['scores'].max():.2f}"
        cv2.putText(vis, label_text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Legend at bottom
        cv2.rectangle(vis, (0, h - 25), (w, h), (40, 40, 40), -1)
        cv2.putText(vis, "Green = Ground Truth    Red = Prediction", (5, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save to verdict folder
        out_name = f"{img_info['file_name'].rsplit('.', 1)[0]}_result.png"
        cv2.imwrite(str(verdict_dirs[verdict] / out_name), vis)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Processed {i+1}/{total} images...")

    print(f"\n  Results breakdown:")
    print(f"    TP (correct defect):  {counts['TP']}")
    print(f"    TN (correct good):    {counts['TN']}")
    print(f"    FP (false alarm):     {counts['FP']}  ← review for over-detection")
    print(f"    FN (missed defect):   {counts['FN']}  ← review for relabeling")
    print(f"\n  Visualizations saved to: {vis_dir}")
    for v in ["TP", "FP", "FN", "TN"]:
        print(f"    {v}/  ({counts[v]} images)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Mask R-CNN pipeline for o-ring defect detection"
    )
    parser.add_argument("model", type=str, choices=["model1", "model2", "combined"],
                        help="O-ring model to process (combined = both merged)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use existing COCO data)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing checkpoint)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override score threshold for inference")

    args = parser.parse_args()

    # Apply overrides
    cfg = TRAINING_CONFIG
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.threshold:
        cfg.score_threshold = args.threshold

    print("=" * 80)
    print(f"  MASK R-CNN PIPELINE — {ORING_MODELS[args.model].description}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    pipeline_start = time.time()

    # Step 1: Preprocessing
    if not args.skip_preprocess:
        ok = run_preprocessing(args.model)
        if not ok:
            print("\n✗ Pipeline failed at preprocessing. Aborting.")
            sys.exit(1)
    else:
        print("\n  [Skipping preprocessing — using existing COCO data]")

    # Step 2: Training
    if not args.skip_training:
        ok = run_training(args.model, cfg, args.resume)
        if not ok:
            print("\n✗ Pipeline failed at training. Aborting.")
            sys.exit(1)
    else:
        print("\n  [Skipping training — using existing checkpoint]")

    # Step 3: Evaluation
    run_evaluation(args.model, cfg)

    # Step 4: Inference visualizations
    run_inference_visualization(args.model, cfg)

    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*80}")
    print(f"  PIPELINE COMPLETE — {args.model}")
    print(f"  Total time: {minutes}m {seconds}s")
    print(f"  Output:     {OUTPUT_ROOT / args.model}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
