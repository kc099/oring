"""
YOLO11 Segmentation Training Script for Patchcore Defect Detection.

Trains yolo11m-seg on the prepared 640x480 dataset.
Uses early stopping (patience-based) with val = train since the dataset is small.

Usage:
    python train.py                          # defaults
    python train.py --model yolo11n-seg      # lighter model
    python train.py --epochs 300 --batch 4   # custom settings
"""

import argparse
import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = SCRIPT_DIR / "dataset" / "data.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "runs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLO11-seg on patchcore defect dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default="yolo11m-seg",
                    help="YOLO model variant")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML),
                    help="Path to data.yaml")
    p.add_argument("--epochs", type=int, default=200,
                    help="Max training epochs")
    p.add_argument("--batch", type=int, default=8,
                    help="Batch size")
    p.add_argument("--imgsz", type=int, default=512,
                    help="Input image size")
    p.add_argument("--lr", type=float, default=0.01,
                    help="Initial learning rate")
    p.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience (epochs without improvement)")
    p.add_argument("--workers", type=int, default=4,
                    help="Dataloader workers")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Project output directory")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        logger.error(f"data.yaml not found: {args.data}")
        logger.error("Run prepare_dataset.py first.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected — training will be slow.")

    logger.info("=" * 70)
    logger.info("YOLO11 SEGMENTATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Model     : {args.model}")
    logger.info(f"  Data      : {args.data}")
    logger.info(f"  Epochs    : {args.epochs}")
    logger.info(f"  Batch     : {args.batch}")
    logger.info(f"  Img size  : {args.imgsz}")
    logger.info(f"  LR        : {args.lr}")
    logger.info(f"  Patience  : {args.patience}")
    logger.info(f"  Device    : {device}")
    logger.info("=" * 70)

    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=args.output,
        name=f"{args.model}_training",
        exist_ok=False,
        verbose=True,
        plots=True,
        amp=True,
        lr0=args.lr,
        save=True,
        save_period=-1,
        # Augmentation — aggressive for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # Print best model path
    if hasattr(results, "save_dir"):
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        logger.info(f"\nBest model saved to: {best_pt}")
    else:
        logger.info(f"\nResults saved to: {args.output}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
