"""
Full Pipeline: YOLO Crop → PatchCore Train.

Usage:
    python run_full_pipeline.py                         # all defaults
    python run_full_pipeline.py --skip-crop             # retrain only
    python run_full_pipeline.py --model model1          # one model
    python run_full_pipeline.py --backbone resnet101    # different backbone
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list, desc: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED: {desc}")
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Full PatchCore pipeline")
    p.add_argument("--skip-crop", action="store_true",
                    help="Skip YOLO cropping (reuse existing crops)")
    p.add_argument("--model", choices=["model1", "model2", "all"], default="all")
    p.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    p.add_argument("--coreset", type=float, default=0.03)
    p.add_argument("--yolo", type=str, default="",
                    help="Path to YOLO best.pt")
    args = p.parse_args()

    py = sys.executable

    # Step 1: YOLO crop
    if not args.skip_crop:
        crop_cmd = [py, str(SCRIPT_DIR / "crop_with_yolo.py")]
        if args.yolo:
            crop_cmd += ["--model", args.yolo]
        ok = run(crop_cmd, "Step 1: Crop images with YOLO segmentation")
        if not ok:
            return

    # Step 2: Train PatchCore
    train_cmd = [
        py, str(SCRIPT_DIR / "train_patchcore.py"),
        "--model", args.model,
        "--backbone", args.backbone,
        "--coreset", str(args.coreset),
    ]
    ok = run(train_cmd, "Step 2: Train PatchCore on cropped images")
    if not ok:
        return

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("")
    print("  To run inference GUI:")
    print(f"    python {SCRIPT_DIR / 'patchcore_inference_gui.py'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
