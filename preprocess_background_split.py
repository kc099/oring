"""
Preprocess: background subtraction (Otsu) + split into 640x640 patches.

Pipeline:
1) Load original 2448x2048 image
2) Build foreground mask using Otsu thresholding (largest component)
3) Optionally fill holes in the mask
4) Subtract background (set background to black)
5) Split into 640x640 patches with overlap

Outputs:
- Patched images saved to OUTPUT_ROOT/images/<folder>
- Optional debug masks saved to OUTPUT_ROOT/masks/<folder>

Author: GitHub Copilot
Date: February 6, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


class BackgroundSubtractionSplitter:
    def __init__(self,
                 source_root: str,
                 output_root: str,
                 patch_size: int = 640,
                 overlap: int = 100,
                 padding: int = 0,
                 save_debug_masks: bool = False,
                 fill_holes: bool = True):
        """
        Args:
            source_root: Root folder containing image folders (good/notok/model1good/model1defect)
            output_root: Output root for patches
            patch_size: Patch size (default 640 for YOLO)
            overlap: Overlap between patches (helps preserve edge defects)
            padding: Optional padding (unused for subtraction; kept for future extension)
            save_debug_masks: If True, saves binary mask patches
            fill_holes: If True, fills holes inside the o-ring mask
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding
        self.save_debug_masks = save_debug_masks
        self.fill_holes = fill_holes

        self.folders = ['good', 'notok', 'model1good', 'model1defect']

    def _calculate_patches(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Calculate patch coordinates to cover entire image."""
        stride = self.patch_size - self.overlap
        num_x = max(1, int(np.ceil((width - self.overlap) / stride)))
        num_y = max(1, int(np.ceil((height - self.overlap) / stride)))

        patches = []
        for iy in range(num_y):
            for ix in range(num_x):
                x = ix * stride
                y = iy * stride

                # Snap last patch to border
                if x + self.patch_size > width:
                    x = max(0, width - self.patch_size)
                if y + self.patch_size > height:
                    y = max(0, height - self.patch_size)

                w = min(self.patch_size, width - x)
                h = min(self.patch_size, height - y)
                patches.append((x, y, w, h))

        # Remove duplicates if any
        patches = list(set(patches))
        patches.sort()
        return patches

    def _otsu_mask(self, image: np.ndarray) -> np.ndarray:
        """Create foreground mask using Otsu thresholding and largest component."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Heuristic: if white region is too large, invert (background might be white)
        white_ratio = np.mean(binary == 255)
        if white_ratio > 0.75:
            binary = cv2.bitwise_not(binary)

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return binary

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.zeros_like(binary)
        mask[labels == largest_label] = 255

        if self.fill_holes:
            mask = self._fill_holes(mask)

        return mask

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in a binary mask (useful for ring interiors)."""
        h, w = mask.shape[:2]
        flood = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = mask | flood_inv
        return filled

    def _subtract_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image and set background to black."""
        result = image.copy()
        result[mask == 0] = 0
        return result

    def process_image(self, image_path: Path, folder: str) -> int:
        """Process one image and save patches. Returns patch count."""
        image = cv2.imread(str(image_path))
        if image is None:
            return 0

        mask = self._otsu_mask(image)
        fg_image = self._subtract_background(image, mask)

        h, w = fg_image.shape[:2]
        patches = self._calculate_patches(w, h)

        out_img_folder = self.output_root / 'images' / folder
        out_img_folder.mkdir(parents=True, exist_ok=True)

        if self.save_debug_masks:
            out_mask_folder = self.output_root / 'masks' / folder
            out_mask_folder.mkdir(parents=True, exist_ok=True)
        else:
            out_mask_folder = None

        patch_count = 0
        for idx, (x, y, pw, ph) in enumerate(patches):
            patch = fg_image[y:y+ph, x:x+pw]
            patch_name = f"{image_path.stem}_patch_{idx:03d}.bmp"
            cv2.imwrite(str(out_img_folder / patch_name), patch)

            if out_mask_folder is not None:
                mask_patch = mask[y:y+ph, x:x+pw]
                mask_name = f"{image_path.stem}_patch_{idx:03d}_mask.png"
                cv2.imwrite(str(out_mask_folder / mask_name), mask_patch)

            patch_count += 1

        return patch_count

    def process_all(self) -> None:
        """Process all images in all folders."""
        print("=" * 100)
        print("BACKGROUND SUBTRACTION + PATCH SPLIT")
        print(f"Source: {self.source_root}")
        print(f"Output: {self.output_root}")
        print(f"Patch:  {self.patch_size}x{self.patch_size} (overlap={self.overlap}px)")
        print(f"Fill holes: {self.fill_holes}")
        print(f"Save debug masks: {self.save_debug_masks}")
        print("=" * 100)

        total_images = 0
        total_patches = 0

        for folder in self.folders:
            folder_path = self.source_root / folder
            if not folder_path.exists():
                print(f"\n{folder}: NOT FOUND")
                continue

            image_files = (
                list(folder_path.glob('*.bmp')) +
                list(folder_path.glob('*.jpg')) +
                list(folder_path.glob('*.png'))
            )

            print(f"\n{folder}: {len(image_files)} images")
            folder_patches = 0

            for img_path in sorted(image_files):
                folder_patches += self.process_image(img_path, folder)
                total_images += 1

            total_patches += folder_patches
            print(f"  → {folder_patches} patches")

        print("\n" + "=" * 100)
        print(f"Total images processed: {total_images}")
        print(f"Total patches created:  {total_patches}")
        print("=" * 100)


def main() -> None:
    SOURCE_ROOT = r"F:\standard elastomers\Original Data"
    OUTPUT_ROOT = r"F:\standard elastomers\preprocessed_patches_640"

    splitter = BackgroundSubtractionSplitter(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        patch_size=640,
        overlap=100,
        save_debug_masks=False,
        fill_holes=True
    )

    splitter.process_all()


if __name__ == "__main__":
    main()
