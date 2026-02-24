"""
Quick start script - Runs the entire training pipeline in sequence.
Execute this to train YOLO model from start to finish.

Usage examples:
    # Full pipeline with defaults (yolo11n-seg, 100 epochs)
    python run_training_pipeline.py

    # Use medium model, custom LR and epochs
    python run_training_pipeline.py --model yolo11m-seg --lr 0.005 --epochs 200 --batch-size 4

    # Skip preprocessing, just retrain
    python run_training_pipeline.py --skip-preprocess --epochs 50

    # Available models: yolo11n-seg, yolo11m-seg
"""

import argparse
import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Full command list [python, script, args...]
        description: Description of what the script does

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(description)
    logger.info("=" * 80)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error code {e.returncode}\n")
        return False
    except Exception as e:
        logger.error(f"✗ Error: {e}\n")
        return False


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="YOLO Segmentation Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, default="yolo11n-seg",
        choices=["yolo11n-seg", "yolo11m-seg"],
        help="YOLO model variant to train",
    )
    p.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    p.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip mask conversion and dataset splitting (use existing yolo_dataset_split)",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training — only run TP/FP/TN/FN evaluation on existing best.pt",
    )
    return p.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "YOLO SEGMENTATION MODEL - TRAINING PIPELINE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  LR         : {args.lr}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Batch size : {args.batch_size}")
    logger.info(f"  Skip preprocess: {args.skip_preprocess}")
    logger.info(f"  Eval only  : {args.eval_only}")
    logger.info("\n")

    here = os.path.dirname(__file__)
    failed_steps = []

    # ── STEP 1 & 2: preprocessing (can be skipped) ───────────────────────
    if not args.skip_preprocess:
        convert_script = os.path.join(here, '..', 'yolo_preprocessing', 'convert_masks_to_yolo.py')
        ok = run_command(
            [sys.executable, convert_script],
            "STEP 1: Converting Masks to YOLO Format",
        )
        if not ok:
            failed_steps.append("STEP 1: Converting Masks")

        prepare_script = os.path.join(here, 'prepare_dataset.py')
        ok = run_command(
            [sys.executable, prepare_script],
            "STEP 2: Preparing Dataset (Train/Val/Test Split)",
        )
        if not ok:
            failed_steps.append("STEP 2: Preparing Dataset")
    else:
        logger.info("⏭  Skipping STEP 1 & 2 (--skip-preprocess)\n")

    # ── STEP 3: training (or eval-only) ───────────────────────────────────
    train_script = os.path.join(here, 'train_yolo.py')
    train_cmd = [
        sys.executable, train_script,
        "--model", args.model,
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.eval_only:
        train_cmd.append("--eval-only")
    step_desc = "STEP 3: Evaluating existing model" if args.eval_only else "STEP 3: Training YOLO Segmentation Model"
    ok = run_command(train_cmd, step_desc)
    if not ok:
        failed_steps.append(step_desc)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "PIPELINE EXECUTION SUMMARY".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")

    if not failed_steps:
        logger.info("✓ All steps completed successfully!")
        logger.info("\nTraining pipeline finished. Check results in:")
        logger.info("  - Models: yolo_dataset_split/")
        logger.info("  - Training: runs/")
        logger.info("  - Evaluation: runs/<model>_training/val_evaluation/")
        return 0
    else:
        logger.error("✗ Some steps failed:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        logger.info("\nPlease check the errors above and fix them.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
