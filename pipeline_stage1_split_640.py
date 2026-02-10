"""
Stage 1 Pipeline: Original images → 640x640 patches (9 per image)

This script reuses existing tools:
- extract_oring_crops.py: Otsu-based background subtraction + crop bbox info
- split_oring_crops.py: 640x640 patch splitting + mask coordinate transform

Usage:
  python pipeline_stage1_split_640.py

Outputs:
- oring_crops/              (cropped images + bbox info)
- oring_patches_640/         (images + labels in patch form)

Author: GitHub Copilot
Date: February 6, 2026
"""

import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    """Run a Python script in the current environment."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n▶ Running: {script_name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name}")


def main() -> None:
    print("=" * 90)
    print("PIPELINE STAGE 1: ORIGINAL → 640x640 PATCHES")
    print("=" * 90)

    # Step 1: background subtraction + crop + bbox info
    run_step("extract_oring_crops.py")

    # Step 2: split crops into 640x640 patches + transform masks
    run_step("split_oring_crops.py")

    print("\n✓ Stage 1 complete: oring_patches_640 is ready")


if __name__ == "__main__":
    main()
