"""
YOLO Segmentation Model Training Script.
Trains a YOLO segmentation model on the defect detection dataset.

After training completes, runs full evaluation on the validation set
and produces TP / FP / TN / FN classification plus saved visualisations.

Usage:
    python train_yolo.py                                   # defaults
    python train_yolo.py --model yolo11m-seg --epochs 200  # medium model
"""

import argparse
import os
import json
import shutil
import torch
import gc
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOSegmentationTrainer:
    """YOLO v8 Segmentation model trainer."""
    
    def __init__(self, data_yaml: str, output_dir: str = 'runs/detect'):
        """
        Initialize the trainer.
        
        Args:
            data_yaml: Path to data.yaml file
            output_dir: Output directory for results
        """
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.device = self._get_device()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Data config: {data_yaml}")
    
    def _get_device(self) -> str:
        """Get the device to use for training."""
        if torch.cuda.is_available():
            device = 0  # GPU
            logger.info(f"GPU available: {torch.cuda.get_device_name(device)}")
        else:
            device = 'cpu'
            logger.warning("GPU not available, using CPU (slow training)")
        return device
    
    def train(
        self,
        model_name: str = 'yolo11n-seg',
        epochs: int = 100,
        imgsz: int = 720,
        batch_size: int = 8,
        lr0: float = 0.01,
        patience: int = 20,
        save: bool = True,
        save_period: int = -1,
        resume: bool = False,
        device: str = None,
        workers: int = 4,
        **kwargs,
    ) -> dict:
        """
        Train the YOLO segmentation model.

        Args:
            model_name: Model variant ('yolo11n-seg' or 'yolo11m-seg')
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size
            lr0: Initial learning rate
            patience: Early stopping patience
            save: Whether to save checkpoints
            save_period: Save checkpoint every X epochs (-1 = best only)
            resume: Resume training from last checkpoint
            device: Device to use ('cpu' or GPU index)
            workers: Number of data loader workers

        Returns:
            Training results
        """
        if device is None:
            device = self.device

        logger.info("=" * 80)
        logger.info("YOLO Segmentation Model Training")
        logger.info("=" * 80)
        logger.info(f"Model       : {model_name}")
        logger.info(f"Epochs      : {epochs}")
        logger.info(f"Image size  : {imgsz}")
        logger.info(f"Batch size  : {batch_size}")
        logger.info(f"Learning rate: {lr0}")
        logger.info(f"Device      : {device}")
        logger.info(f"Resume      : {resume}")
        logger.info("=" * 80)

        # Load model
        if resume:
            logger.info("Resuming training from last checkpoint...")
            model = YOLO(f'{model_name}.pt')
        else:
            logger.info(f"Loading model: {model_name}")
            model = YOLO(f'{model_name}.pt')

        # Training parameters
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': patience,
            'device': device,
            'workers': workers,
            'save': save,
            'save_period': save_period,
            'resume': resume,
            'project': self.output_dir,
            'name': f'{model_name}_training',
            'exist_ok': True if resume else False,
            'verbose': True,
            'plots': True,
            'amp': True,
            'lr0': lr0,
            # Augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 180.0,
            'translate': 0.1,
            'scale': 0.5,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.0,
        }
        train_args.update(kwargs)

        # Train (no per-epoch prediction callback — evaluation happens after)
        try:
            results = model.train(**train_args)
            logger.info("=" * 80)
            logger.info("Training Completed Successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # ── Post-training evaluation on full validation set ──────────────
        # Discover the actual run directory that ultralytics created
        # (may be _training, _training2, _training3, …)
        actual_run_dir = None
        if hasattr(results, 'save_dir'):
            actual_run_dir = str(results.save_dir)
        if not actual_run_dir or not os.path.isdir(actual_run_dir):
            # fallback: find newest matching dir
            actual_run_dir = self._find_latest_run_dir(model_name)

        if actual_run_dir:
            best_pt = os.path.join(actual_run_dir, 'weights', 'best.pt')
        else:
            best_pt = os.path.join(
                self.output_dir, f'{model_name}_training', 'weights', 'best.pt')

        if os.path.exists(best_pt):
            logger.info("\n")
            logger.info("=" * 80)
            logger.info("POST-TRAINING EVALUATION  (full val set)")
            logger.info("=" * 80)
            self._evaluate_val_set(best_pt)
        else:
            logger.warning(f"best.pt not found at {best_pt} — skipping evaluation")

        return results

    # ── Post-training evaluation ─────────────────────────────────────────

    def _find_latest_run_dir(self, model_name: str) -> str | None:
        """Find the latest run directory matching `model_name_training*`."""
        prefix = f'{model_name}_training'
        if not os.path.isdir(self.output_dir):
            return None
        candidates = [
            os.path.join(self.output_dir, d)
            for d in os.listdir(self.output_dir)
            if d.startswith(prefix) and os.path.isdir(os.path.join(self.output_dir, d))
        ]
        if not candidates:
            return None
        # Sort by modification time descending → newest first
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _evaluate_val_set(self, best_pt: str):
        """Run the best model on every val image and classify TP/FP/TN/FN.

        The evaluation output directory is derived from ``best_pt``
        (sibling of the ``weights/`` folder that contains it).

        Ground truth:
            - label .txt is non-empty  →  has defect (positive)
            - label .txt is empty / missing  →  no defect (negative)

        Model prediction:
            - ≥1 detection  →  predicted positive
            - 0 detections  →  predicted negative

        Saves:
            <run_dir>/val_evaluation/
                summary.json          — counts & per-image classification
                TP/  FP/  TN/  FN/    — annotated images in each bucket
        """
        import yaml as _yaml

        # Resolve val images dir from data.yaml
        with open(self.data_yaml, 'r') as f:
            data_cfg = _yaml.safe_load(f)
        data_root = data_cfg.get('path', os.path.dirname(self.data_yaml))
        val_rel = data_cfg.get('val', 'images/val')
        val_images_dir = os.path.normpath(
            os.path.join(data_root, val_rel) if not os.path.isabs(val_rel) else val_rel)
        val_labels_dir = val_images_dir.replace(
            os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)

        if not os.path.isdir(val_images_dir):
            logger.warning(f"Val images dir not found: {val_images_dir}")
            return

        # Derive run dir from best_pt path:  …/<run_dir>/weights/best.pt
        run_dir = os.path.dirname(os.path.dirname(best_pt))
        eval_dir = os.path.join(run_dir, 'val_evaluation')
        for bucket in ('TP', 'FP', 'TN', 'FN'):
            os.makedirs(os.path.join(eval_dir, bucket), exist_ok=True)

        # Collect val image list
        exts = {'.bmp', '.jpg', '.jpeg', '.png'}
        image_files = sorted([
            f for f in os.listdir(val_images_dir)
            if Path(f).suffix.lower() in exts
        ])
        if not image_files:
            logger.warning("No images found in val set.")
            return

        logger.info(f"Evaluating {len(image_files)} validation images …")

        # Load model once
        eval_model = YOLO(best_pt)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        per_image = []

        for img_name in image_files:
            img_path = os.path.join(val_images_dir, img_name)
            stem = Path(img_name).stem
            label_path = os.path.join(val_labels_dir, stem + '.txt')

            # Ground truth
            has_gt = False
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    content = lf.read().strip()
                has_gt = len(content) > 0

            # Prediction
            results = eval_model.predict(
                img_path, conf=0.3, iou=0.45, verbose=False, device=device)
            result = results[0]
            n_det = len(result.boxes) if result.boxes is not None else 0
            has_pred = n_det > 0
            top_score = float(result.boxes.conf.max()) if has_pred else 0.0

            # Classify
            if has_gt and has_pred:
                bucket = 'TP'
            elif (not has_gt) and has_pred:
                bucket = 'FP'
            elif (not has_gt) and (not has_pred):
                bucket = 'TN'
            else:  # has_gt and not has_pred
                bucket = 'FN'

            counts[bucket] += 1
            per_image.append({
                'image': img_name,
                'ground_truth': has_gt,
                'predicted': has_pred,
                'num_detections': n_det,
                'top_score': round(top_score, 4),
                'classification': bucket,
            })

            # Save annotated image into bucket folder
            annotated = result.plot()
            # If ground truth exists, also draw GT polygons in green
            if has_gt:
                annotated = self._draw_gt_polygons(
                    annotated, label_path, annotated.shape[1], annotated.shape[0])
            out_path = os.path.join(eval_dir, bucket, f'{stem}.png')
            cv2.imwrite(out_path, annotated)

        # Clean up
        del eval_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Summary
        total = len(image_files)
        precision = counts['TP'] / max(counts['TP'] + counts['FP'], 1)
        recall = counts['TP'] / max(counts['TP'] + counts['FN'], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (counts['TP'] + counts['TN']) / max(total, 1)

        summary = {
            'total_images': total,
            'counts': counts,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'per_image': per_image,
        }
        summary_path = os.path.join(eval_dir, 'summary.json')
        with open(summary_path, 'w') as sf:
            json.dump(summary, sf, indent=2)

        logger.info("")
        logger.info("╔" + "═" * 50 + "╗")
        logger.info("║  VALIDATION EVALUATION RESULTS".ljust(51) + "║")
        logger.info("╠" + "═" * 50 + "╣")
        logger.info(f"║  Total images : {total:<33}║")
        logger.info(f"║  TP (correct defect)    : {counts['TP']:<23}║")
        logger.info(f"║  FP (false alarm)       : {counts['FP']:<23}║")
        logger.info(f"║  TN (correct no-defect) : {counts['TN']:<23}║")
        logger.info(f"║  FN (missed defect)     : {counts['FN']:<23}║")
        logger.info("╠" + "═" * 50 + "╣")
        logger.info(f"║  Precision : {precision:.4f}".ljust(51) + "║")
        logger.info(f"║  Recall    : {recall:.4f}".ljust(51) + "║")
        logger.info(f"║  F1 Score  : {f1:.4f}".ljust(51) + "║")
        logger.info(f"║  Accuracy  : {accuracy:.4f}".ljust(51) + "║")
        logger.info("╚" + "═" * 50 + "╝")
        logger.info(f"\nDetailed results saved to: {eval_dir}")

    @staticmethod
    def _draw_gt_polygons(image: np.ndarray, label_path: str, w: int, h: int) -> np.ndarray:
        """Draw ground-truth YOLO polygons on an image in green."""
        if not os.path.exists(label_path):
            return image
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            coords = parts[1:]
            if len(coords) % 2 != 0:
                continue
            points = []
            for i in range(0, len(coords), 2):
                px = int(round(float(coords[i]) * w))
                py = int(round(float(coords[i + 1]) * h))
                points.append([px, py])
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Legend
        cv2.putText(image, "Green=GT", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def validate(self, model_path: str) -> dict:
        """Validate a trained model."""
        logger.info("Running validation...")
        model = YOLO(model_path)
        results = model.val(data=self.data_yaml, device=self.device)
        return results


def parse_args():
    """Parse CLI arguments for standalone execution."""
    p = argparse.ArgumentParser(
        description="Train YOLO segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, default="yolo11n-seg",
        choices=["yolo11n-seg", "yolo11m-seg"],
        help="YOLO model variant",
    )
    p.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--imgsz", type=int, default=720, help="Input image size")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt")
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training — only run TP/FP/TN/FN evaluation on the val set using existing best.pt",
    )
    return p.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    dataset_split_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset_split')
    data_yaml = os.path.join(dataset_split_dir, 'data.yaml')
    output_dir = os.path.join(os.path.dirname(__file__), 'runs')

    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml not found at {data_yaml}")
        logger.info("Please run 'python prepare_dataset.py' first to prepare the dataset.")
        return

    trainer = YOLOSegmentationTrainer(data_yaml=data_yaml, output_dir=output_dir)

    if args.eval_only:
        # Standalone evaluation — find the latest matching run dir
        latest_dir = trainer._find_latest_run_dir(args.model)
        if latest_dir:
            best_pt = os.path.join(latest_dir, 'weights', 'best.pt')
        else:
            best_pt = os.path.join(output_dir, f'{args.model}_training', 'weights', 'best.pt')
        if not os.path.exists(best_pt):
            logger.error(f"best.pt not found at {best_pt}")
            logger.info("Train a model first, or check --model matches the trained variant.")
            return
        logger.info("\n")
        logger.info("=" * 80)
        logger.info(f"STANDALONE EVALUATION  (full val set) — {os.path.dirname(os.path.dirname(best_pt))}")
        logger.info("=" * 80)
        trainer._evaluate_val_set(best_pt)
    else:
        trainer.train(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            lr0=args.lr,
            patience=args.patience,
            resume=args.resume,
            save=True,
            workers=4,
        )
        logger.info(f"\nTraining complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
