"""
Run the full pipeline: prepare dataset → train YOLO11-seg.

Usage:
    python run_pipeline.py                          # defaults
    python run_pipeline.py --skip-prepare           # retrain only
    python run_pipeline.py --model yolo11n-seg      # lighter model
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list, desc: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {desc}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Full YOLO11-seg pipeline")
    p.add_argument("--skip-prepare", action="store_true",
                    help="Skip dataset preparation (reuse existing)")
    p.add_argument("--model", type=str, default="yolo11m-seg")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()

    py = sys.executable

    # Step 1: Prepare dataset
    if not args.skip_prepare:
        ok = run(
            [py, str(SCRIPT_DIR / "prepare_dataset.py")],
            "Step 1: Prepare dataset (resize + convert to YOLO format)",
        )
        if not ok:
            return

    # Step 2: Train
    train_cmd = [
        py, str(SCRIPT_DIR / "train.py"),
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
    ]
    ok = run(train_cmd, "Step 2: Train YOLO11-seg model")
    if not ok:
        return

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("  To run inference GUI:")
    print(f"    python {SCRIPT_DIR / 'inference_gui.py'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
