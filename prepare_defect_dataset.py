"""
Prepare a training dataset from 640x640 patches with strict label rules.

Rules:
- Include ALL non-empty masks from: good, model1good, notok, model1defect
- Include EMPTY masks ONLY from: good, model1good

This avoids wrongly-labeled samples in defect folders while still keeping
true good samples from good/model1good.

Outputs:
    dataset/
        images/<folder>/
        labels/<folder>/

Author: GitHub Copilot
Date: February 6, 2026
"""

import json
import os
import shutil
from pathlib import Path


def has_non_empty_polygons(mask_path: Path) -> bool:
    """Return True if mask JSON has at least one polygon with >= 3 points."""
    try:
        with open(mask_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False

    polygons = data.get('polygons', [])
    for poly in polygons:
        points = poly.get('points', [])
        if len(points) >= 3:
            return True
    return False


def prepare_defect_dataset(patches_root: str, output_root: str) -> None:
    patches_root = Path(patches_root)
    output_root = Path(output_root)

    folders = ['good', 'model1good', 'notok', 'model1defect']
    empty_allowed = {'good', 'model1good'}

    total_masks = 0
    kept = 0
    kept_empty = 0
    kept_non_empty = 0
    total_images = 0

    for folder in folders:
        label_dir = patches_root / 'labels' / folder
        image_dir = patches_root / 'images' / folder
        nested_label_dir = label_dir / f"{folder}_masks"

        if not label_dir.exists() or not image_dir.exists():
            print(f"{folder}: missing images/labels folder")
            continue

        out_img_dir = output_root / 'images' / folder
        out_lbl_dir = output_root / 'labels' / folder
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        non_empty_masks = {}
        any_masks = set()
        if nested_label_dir.exists():
            mask_files = sorted(nested_label_dir.glob('*_mask.json'))
            for mask_path in mask_files:
                stem = mask_path.stem.replace('_mask', '')
                any_masks.add(stem)
                if has_non_empty_polygons(mask_path):
                    non_empty_masks[stem] = mask_path
            total_masks += len(mask_files)
        else:
            print(f"{folder}: nested masks folder not found: {nested_label_dir}")

        # Process images:
        # - Defect folders: keep ONLY images with non-empty masks
        # - Good folders: keep ONLY images with NO mask file at all
        image_files = []
        for ext in ('.bmp', '.jpg', '.jpeg', '.png'):
            image_files.extend(image_dir.glob(f'*{ext}'))

        folder_kept = 0
        for image_path in sorted(image_files):
            total_images += 1
            stem = image_path.stem
            non_empty_mask_path = non_empty_masks.get(stem)

            if folder in empty_allowed:
                # Good folders: keep only images with no mask file at all
                if stem in any_masks:
                    continue
                shutil.copy2(image_path, out_img_dir / image_path.name)
                empty_mask_path = out_lbl_dir / f"{stem}_mask.json"
                with open(empty_mask_path, 'w', encoding='utf-8') as f:
                    json.dump({"polygons": []}, f, indent=2)
                folder_kept += 1
                kept += 1
                kept_empty += 1
            else:
                # Defect folders: keep only non-empty masks
                if non_empty_mask_path is None:
                    continue
                shutil.copy2(image_path, out_img_dir / image_path.name)
                shutil.copy2(non_empty_mask_path, out_lbl_dir / non_empty_mask_path.name)
                folder_kept += 1
                kept += 1
                kept_non_empty += 1

        print(f"{folder}: kept {folder_kept} / {len(image_files)}")

    print("=" * 80)
    print(f"Total masks scanned: {total_masks}")
    print(f"Total images scanned: {total_images}")
    print(f"Total kept (non-empty): {kept_non_empty}")
    print(f"Total kept (empty): {kept_empty}")
    print(f"Total kept (all): {kept}")
    print(f"Dataset output: {output_root}")
    print("=" * 80)


def main() -> None:
    # Allow overriding the patches root via environment variable
    PATCHES_ROOT = os.environ.get(
        "ORING_PATCHES_ROOT",
        r"F:\standard elastomers\oring_patches_640"
    )
    OUTPUT_ROOT = r"F:\standard elastomers\dataset"

    prepare_defect_dataset(PATCHES_ROOT, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
