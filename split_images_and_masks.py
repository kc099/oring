"""
Split large images and their corresponding segmentation masks into smaller patches for YOLOv8 training.
This script handles:
- Splitting 2kx2k images into smaller patches (default 640x640)
- Transforming corresponding JSON mask polygon coordinates
- Creating a new directory structure with split images and masks
- Handling good images (without masks)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import shutil


class ImageMaskSplitter:
    def __init__(self, 
                 source_root: str,
                 output_root: str,
                 patch_size: int = 640,
                 overlap: int = 0,
                 min_brightness_threshold: int = 10,
                 min_nonblack_ratio: float = 0.05):
        """
        Initialize the splitter.
        
        Args:
            source_root: Root directory containing image folders and masks folder
            output_root: Output directory for split images and masks
            patch_size: Size of each patch (default 640x640 for YOLOv8)
            overlap: Overlap between patches in pixels (default 0)
            min_brightness_threshold: Minimum average brightness (0-255) to consider patch valid
            min_nonblack_ratio: Minimum ratio of non-black pixels (0-1) to consider patch valid
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_brightness_threshold = min_brightness_threshold
        self.min_nonblack_ratio = min_nonblack_ratio
        self.skipped_patches = 0
        
        # Define image folders and their corresponding mask folders
        self.folder_mapping = {
            'model1defect': 'model1defect_masks',
            'model1rework': 'model1rework_masks',
            'notok': 'notok_masks',
            'Rework': 'Rework_masks',
            'model1good': None,  # No masks for good images
            'good': None  # No masks for good images
        }
        
    def calculate_patches(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate patch coordinates for splitting an image.
        
        Returns:
            List of (x_start, y_start, x_end, y_end) tuples
        """
        patches = []
        stride = self.patch_size - self.overlap
        
        # Calculate number of patches needed
        num_patches_x = (image_width - self.overlap) // stride
        if (image_width - self.overlap) % stride != 0:
            num_patches_x += 1
            
        num_patches_y = (image_height - self.overlap) // stride
        if (image_height - self.overlap) % stride != 0:
            num_patches_y += 1
        
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                x_start = j * stride
                y_start = i * stride
                
                # Adjust last patches to fit within image bounds
                x_end = min(x_start + self.patch_size, image_width)
                y_end = min(y_start + self.patch_size, image_height)
                
                # Adjust start if patch would be smaller than patch_size
                if x_end - x_start < self.patch_size:
                    x_start = max(0, x_end - self.patch_size)
                if y_end - y_start < self.patch_size:
                    y_start = max(0, y_end - self.patch_size)
                
                patches.append((x_start, y_start, x_end, y_end))
        
        return patches
    
    def point_in_patch(self, x: float, y: float, patch: Tuple[int, int, int, int]) -> bool:
        """Check if a point is within a patch."""
        x_start, y_start, x_end, y_end = patch
        return x_start <= x < x_end and y_start <= y < y_end
    
    def transform_polygon(self, 
                         polygon: Dict, 
                         patch: Tuple[int, int, int, int]) -> Dict:
        """
        Transform polygon coordinates to patch-local coordinates.
        
        Args:
            polygon: Original polygon dict with points
            patch: (x_start, y_start, x_end, y_end)
            
        Returns:
            Transformed polygon or None if polygon doesn't intersect patch
        """
        x_start, y_start, x_end, y_end = patch
        
        # Transform all points
        transformed_points = []
        for point in polygon['points']:
            x = point['x'] - x_start
            y = point['y'] - y_start
            
            # Clamp coordinates to patch boundaries
            x = max(0, min(x, self.patch_size))
            y = max(0, min(y, self.patch_size))
            
            transformed_points.append({'x': x, 'y': y})
        
        # Check if any point is within the patch (with some tolerance)
        any_point_in_patch = False
        for point in polygon['points']:
            if self.point_in_patch(point['x'], point['y'], patch):
                any_point_in_patch = True
                break
        
        # Also check if polygon intersects the patch boundaries
        # (simplified check - just see if transformed points are not all at edges)
        if not any_point_in_patch:
            # Check if polygon spans the patch
            x_coords = [p['x'] for p in polygon['points']]
            y_coords = [p['y'] for p in polygon['points']]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Check if polygon bounds intersect patch
            if not (max_x < x_start or min_x >= x_end or max_y < y_start or min_y >= y_end):
                any_point_in_patch = True
        
        if not any_point_in_patch:
            return None
        
        # Create transformed polygon
        transformed_polygon = {
            'id': polygon['id'],
            'num_points': len(transformed_points),
            'points': transformed_points
        }
        
        return transformed_polygon
    
    def is_patch_valid(self, patch_img: np.ndarray, debug: bool = False) -> Tuple[bool, str]:
        """
        Check if a patch contains an o-ring (with or without defects) using edge and texture analysis.
        This detects o-ring shapes regardless of whether they have defect annotations.
        
        Args:
            patch_img: Image patch array
            debug: If True, return debug information
            
        Returns:
            Tuple of (is_valid, debug_info)
        """
        # Convert to grayscale
        if len(patch_img.shape) == 3:
            gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch_img
        
        # 1. Check variation - uniform images are empty
        std_dev = np.std(gray)
        if std_dev < 1.5:
            return False, f"Uniform (std={std_dev:.2f})"
        
        # 2. Apply CLAHE to enhance contrast (makes dark o-rings visible)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Edge detection with adaptive thresholds
        median_val = np.median(enhanced)
        lower = int(max(0, 0.5 * median_val))
        upper = int(min(255, 1.5 * median_val))
        edges = cv2.Canny(enhanced, lower, upper)
        
        edge_pixel_count = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_ratio = edge_pixel_count / total_pixels
        
        # 4. Texture analysis - o-rings have texture, empty patches don't
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # 5. Contour detection for shape presence
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) >= 50]
        
        debug_info = (f"Std:{std_dev:.1f} Edge:{edge_ratio:.2%} "
                     f"Texture:{texture_var:.1f} Contours:{len(significant_contours)}")
        
        # Decision: Patch is VALID (has o-ring) if it meets these criteria
        # Based on analysis: o-rings have edge_ratio>6%, texture>100, OR std>15 with multiple contours
        is_valid = (
            (edge_ratio > 0.06 and texture_var > 100) or  # Clear o-ring structure
            (edge_ratio > 0.04 and texture_var > 50 and len(significant_contours) >= 5) or  # O-ring with detail
            (std_dev > 15 and edge_ratio > 0.03 and len(significant_contours) >= 3)  # Darker o-ring with variation
        )
        
        return is_valid, debug_info if not is_valid else ""
    
    def split_image_and_mask(self, 
                            image_folder: str, 
                            image_filename: str,
                            mask_folder: str = None):
        """
        Split a single image and its corresponding mask.
        
        Args:
            image_folder: Name of the image folder
            image_filename: Name of the image file
            mask_folder: Name of the mask folder (None for good images)
        """
        # Read image
        image_path = self.source_root / image_folder / image_filename
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            return
        
        height, width = image.shape[:2]
        
        # Read mask if exists
        mask_data = None
        if mask_folder is not None:
            mask_filename = image_filename.replace('.bmp', '_mask.json').replace('.jpg', '_mask.json').replace('.png', '_mask.json')
            mask_path = self.source_root / 'masks' / mask_folder / mask_filename
            
            if mask_path.exists():
                with open(mask_path, 'r') as f:
                    mask_data = json.load(f)
        
        # Calculate patches
        patches = self.calculate_patches(width, height)
        
        # Create output directories
        output_image_dir = self.output_root / image_folder
        output_image_dir.mkdir(parents=True, exist_ok=True)
        
        if mask_folder is not None:
            output_mask_dir = self.output_root / 'masks' / mask_folder
            output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each patch
        base_name = Path(image_filename).stem
        extension = Path(image_filename).suffix
        
        patches_saved = 0
        patches_skipped = 0
        
        for idx, patch in enumerate(patches):
            x_start, y_start, x_end, y_end = patch
            
            # Extract image patch
            patch_img = image[y_start:y_end, x_start:x_end]
            
            # Validate patch - check if it contains an o-ring
            is_valid, debug_info = self.is_patch_valid(patch_img)
            if not is_valid:
                patches_skipped += 1
                self.skipped_patches += 1
                # Print first few skipped patches for debugging
                if patches_skipped <= 2:
                    print(f"  Skipping patch {idx}: {debug_info}")
                continue
            
            # Generate patch filename
            patch_filename = f"{base_name}_patch_{idx:03d}{extension}"
            patch_path = output_image_dir / patch_filename
            
            # Save image patch
            cv2.imwrite(str(patch_path), patch_img)
            patches_saved += 1
            
            # Process mask if exists
            if mask_data is not None:
                # Transform polygons for this patch
                patch_polygons = []
                for polygon in mask_data.get('polygons', []):
                    transformed = self.transform_polygon(polygon, patch)
                    if transformed is not None:
                        patch_polygons.append(transformed)
                
                # Create mask data for patch
                patch_mask = {
                    'image_filename': patch_filename,
                    'image_folder': image_folder,
                    'image_path': str(patch_path),
                    'image_width': x_end - x_start,
                    'image_height': y_end - y_start,
                    'patch_info': {
                        'original_image': image_filename,
                        'patch_index': idx,
                        'x_start': x_start,
                        'y_start': y_start,
                        'x_end': x_end,
                        'y_end': y_end
                    },
                    'num_polygons': len(patch_polygons),
                    'polygons': patch_polygons
                }
                
                # Save mask
                mask_output_path = output_mask_dir / f"{base_name}_patch_{idx:03d}_mask.json"
                with open(mask_output_path, 'w') as f:
                    json.dump(patch_mask, f, indent=2)
        
        if patches_skipped > 0:
            print(f"Processed: {image_filename} -> {patches_saved} patches saved, {patches_skipped} skipped (empty)")
        else:
            print(f"Processed: {image_filename} -> {patches_saved} patches")
    
    def process_all(self):
        """Process all images in all folders."""
        print(f"Starting image and mask splitting...")
        print(f"Source: {self.source_root}")
        print(f"Output: {self.output_root}")
        print(f"Patch size: {self.patch_size}x{self.patch_size}")
        print(f"Overlap: {self.overlap}px\n")
        
        total_images = 0
        total_patches = 0
        
        for image_folder, mask_folder in self.folder_mapping.items():
            image_dir = self.source_root / image_folder
            
            if not image_dir.exists():
                print(f"Skipping {image_folder} (not found)")
                continue
            
            print(f"\nProcessing folder: {image_folder}")
            
            # Get all image files
            image_files = []
            for ext in ['*.bmp', '*.jpg', '*.jpeg', '*.png']:
                image_files.extend(image_dir.glob(ext))
            
            print(f"Found {len(image_files)} images")
            
            for img_file in sorted(image_files):
                patches_before = total_patches
                self.split_image_and_mask(image_folder, img_file.name, mask_folder)
                total_images += 1
                
                # Estimate patches created (rough calculation)
                total_patches = len(list((self.output_root / image_folder).glob('*')))
            
            print(f"Completed {image_folder}: {len(image_files)} images")
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total original images: {total_images}")
        print(f"Total patches created: {total_patches}")
        print(f"Total patches skipped: {self.skipped_patches} (empty/black)")
        print(f"{'='*60}")
        
        # Print summary of output structure
        print(f"\nOutput structure:")
        for folder in self.folder_mapping.keys():
            output_folder = self.output_root / folder
            if output_folder.exists():
                count = len(list(output_folder.glob('*')))
                print(f"  {folder}: {count} patches")


def main():
    # Configuration
    SOURCE_ROOT = r"F:\standard elastomers"
    OUTPUT_ROOT = r"F:\standard elastomers\split_dataset"
    PATCH_SIZE = 640  # Standard YOLOv8 input size
    OVERLAP = 0  # No overlap, can increase for better detection at boundaries
    
    # Filtering parameters for empty patches
    MIN_BRIGHTNESS = 10  # Minimum average brightness (0-255)
    MIN_NONBLACK_RATIO = 0.05  # Minimum 5% of pixels should be non-black
    
    # Create splitter
    splitter = ImageMaskSplitter(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        min_brightness_threshold=MIN_BRIGHTNESS,
        min_nonblack_ratio=MIN_NONBLACK_RATIO
    )
    
    # Process all images
    splitter.process_all()
    
    print(f"\nDataset ready for YOLOv8 training!")
    print(f"Split images are in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
