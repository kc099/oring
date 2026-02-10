"""
Randomly select 200 images each from good and model1good patches and add to dataset.
Creates empty mask JSON files for each selected image.

Author: GitHub Copilot
Date: February 6, 2026
"""

import json
import os
import random
import shutil
from pathlib import Path


def add_good_samples(patches_root: str, dataset_root: str, sample_size: int = 200, seed: int = 42) -> None:
    patches_root = Path(patches_root)
    dataset_root = Path(dataset_root)

    folders = ['good', 'model1good']
    rng = random.Random(seed)

    for folder in folders:
        src_img_dir = patches_root / 'images' / folder
        if not src_img_dir.exists():
            print(f"{folder}: source images folder not found")
            continue

        image_files = sorted([
            *src_img_dir.glob('*.bmp'),
            *src_img_dir.glob('*.jpg'),
            *src_img_dir.glob('*.jpeg'),
            *src_img_dir.glob('*.png')
        ])

        if len(image_files) < sample_size:
            print(f"{folder}: only {len(image_files)} images available, using all")
            chosen = image_files
        else:
            chosen = rng.sample(image_files, sample_size)

        out_img_dir = dataset_root / 'images' / folder
        out_lbl_dir = dataset_root / 'labels' / folder
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            # Copy image
            shutil.copy2(img_path, out_img_dir / img_path.name)

            # Create empty mask JSON
            stem = img_path.stem
            mask_path = out_lbl_dir / f"{stem}_mask.json"
            if not mask_path.exists():
                with open(mask_path, 'w', encoding='utf-8') as f:
                    json.dump({"polygons": []}, f, indent=2)

        print(f"{folder}: added {len(chosen)} images with empty labels")


def main() -> None:
    # Allow overriding the patches root via environment variable
    PATCHES_ROOT = os.environ.get(
        "ORING_PATCHES_ROOT",
        r"F:\standard elastomers\oring_patches_640"
    )
    DATASET_ROOT = r"F:\standard elastomers\dataset"

    add_good_samples(PATCHES_ROOT, DATASET_ROOT, sample_size=200, seed=42)


if __name__ == "__main__":
    main()
