"""
Stage 2 Pipeline: 640x640 patches → dataset → YOLO format

This script reuses existing tools:
- prepare_defect_dataset.py: include non-empty masks from all folders
    (including nested <folder>_masks) and include empty masks only from
    good/model1good
- yolo_preprocessing/convert_masks_to_yolo.py: convert JSON masks to YOLO labels

Usage:
  python pipeline_stage2_yolo_dataset.py

Outputs:
- dataset/         (filtered samples by label rules)
- yolo_dataset/    (YOLO-formatted images + labels)

Author: GitHub Copilot
Date: February 6, 2026
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path, env: dict) -> None:
    """Run a Python script in the current environment."""
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n▶ Running: {script_path.name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_path.name}")


def main() -> None:
    root = Path(__file__).parent

    dataset_dir = root / "dataset"
    yolo_dir = root / "yolo_dataset"

    patches_root = os.environ.get("ORING_PATCHES_ROOT", str(root / "oring_patches_640"))
    if not Path(patches_root).exists():
        raise FileNotFoundError(
            f"Patches folder not found: {patches_root}\n"
            "Set ORING_PATCHES_ROOT env var or rename your patches folder to 'oring_patches_640'."
        )

    print("=" * 90)
    print("PIPELINE STAGE 2: PATCHES → DATASET → YOLO FORMAT")
    print("=" * 90)

    # Clean previous outputs (optional but recommended)
    if dataset_dir.exists():
        print(f"Cleaning: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    if yolo_dir.exists():
        print(f"Cleaning: {yolo_dir}")
        shutil.rmtree(yolo_dir)

    # Step 1: build dataset with strict label rules
    env = os.environ.copy()
    env["ORING_PATCHES_ROOT"] = patches_root

    run_step(root / "prepare_defect_dataset.py", env)

    # Step 2: convert to YOLO format
    run_step(root / "yolo_preprocessing" / "convert_masks_to_yolo.py", env)

    print("\n✓ Stage 2 complete: yolo_dataset is ready")


if __name__ == "__main__":
    main()
