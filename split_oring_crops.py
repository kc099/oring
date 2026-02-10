"""
Split o-ring crops into patches and transform polygon masks accordingly.
Uses the extracted o-ring crops (after background removal) and splits them for YOLOv8 training.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import shutil


class CropSplitter:
    def __init__(self,
                 crops_root: str,
                 masks_root: str,
                 output_root: str,
                 patch_size: int = 800,
                 overlap: int = 50):
        """
        Initialize crop splitter.
        
        Args:
            crops_root: Root with extracted o-ring crops
            masks_root: Root with original mask JSON files
            output_root: Output directory for split patches
            patch_size: Size of each patch (800 recommended for 2x2, 640 for 3x3)
            overlap: Overlap between patches (pixels)
        """
        self.crops_root = Path(crops_root)
        self.masks_root = Path(masks_root)
        self.output_root = Path(output_root)
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Map crop folders to mask folders
        # Note: good and model1good have no masks (these are defect-free images)
        self.folder_mapping = {
            'model1defect': 'model1defect_masks',
            'notok': 'notok_masks',
        }
        
        # Path to bbox info
        self.bbox_root = self.crops_root / 'bbox_info'
        
        self.stats = {
            'total_images': 0,
            'total_patches': 0,
            'patches_with_labels': 0,
            'patches_without_labels': 0
        }
    
    def calculate_patches(self, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate patch positions to cover entire image.
        
        Returns: List of (x, y, width, height) tuples
        """
        patches = []
        stride = self.patch_size - self.overlap
        
        # Calculate number of patches needed
        num_patches_x = max(1, int(np.ceil((img_width - self.overlap) / stride)))
        num_patches_y = max(1, int(np.ceil((img_height - self.overlap) / stride)))
        
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                x = j * stride
                y = i * stride
                
                # Adjust last patch to fit within image
                if x + self.patch_size > img_width:
                    x = img_width - self.patch_size
                if y + self.patch_size > img_height:
                    y = img_height - self.patch_size
                
                # Ensure non-negative
                x = max(0, x)
                y = max(0, y)
                
                w = min(self.patch_size, img_width - x)
                h = min(self.patch_size, img_height - y)
                
                patches.append((x, y, w, h))
        
        # Remove duplicates (can happen with overlap adjustments)
        patches = list(set(patches))
        patches.sort()
        
        return patches
    
    def transform_polygon(self, polygon: Dict, bbox_x: int, bbox_y: int,
                         patch_x: int, patch_y: int, 
                         patch_w: int, patch_h: int) -> Dict:
        """
        Transform polygon coordinates from original image to patch coordinates.
        This is a two-step transformation:
        1. Original image coords → Crop coords (subtract bbox offset)
        2. Crop coords → Patch coords (subtract patch offset)
        
        Returns: Transformed polygon or None if outside patch
        """
        points = polygon['points']
        transformed_points = []
        
        for point in points:
            # Step 1: Transform from original image to crop coordinates
            crop_x = point['x'] - bbox_x
            crop_y = point['y'] - bbox_y
            
            # Step 2: Transform from crop to patch-local coordinates
            patch_local_x = crop_x - patch_x
            patch_local_y = crop_y - patch_y
            
            # Keep point if within patch bounds (with small tolerance for boundary points)
            if -1 <= patch_local_x <= patch_w + 1 and -1 <= patch_local_y <= patch_h + 1:
                # Clamp to patch boundaries
                clamped_x = max(0, min(patch_w, patch_local_x))
                clamped_y = max(0, min(patch_h, patch_local_y))
                transformed_points.append({'x': clamped_x, 'y': clamped_y})
        
        # Only keep polygon if at least 3 points are inside
        if len(transformed_points) >= 3:
            return {
                'id': polygon['id'],
                'points': transformed_points
            }
        
        return None
    
    def split_crop_and_mask(self, crop_path: Path, folder: str) -> int:
        """
        Split a single crop and its mask.
        
        Returns: Number of patches created
        """
        # Load crop
        img = cv2.imread(str(crop_path))
        if img is None:
            return 0
        
        img_height, img_width = img.shape[:2]
        
        # Load bounding box info
        bbox_info_path = self.bbox_root / folder / f"{crop_path.stem}_bbox.json"
        bbox_x, bbox_y = 0, 0
        
        if bbox_info_path.exists():
            with open(bbox_info_path, 'r') as f:
                bbox_data = json.load(f)
                bbox_x = bbox_data['bbox_x']
                bbox_y = bbox_data['bbox_y']
        else:
            print(f"  ⚠ Warning: No bbox info found for {crop_path.name}, assuming no offset")
        
        # Load mask if exists
        mask_data = None
        mask_folder = self.folder_mapping.get(folder)
        if mask_folder:
            mask_path = self.masks_root / mask_folder / f"{crop_path.stem}_mask.json"
            if mask_path.exists():
                with open(mask_path, 'r') as f:
                    mask_data = json.load(f)
        
        # Calculate patches
        patches = self.calculate_patches(img_width, img_height)
        
        # Create output folders
        output_img_folder = self.output_root / 'images' / folder
        output_img_folder.mkdir(parents=True, exist_ok=True)
        
        output_label_folder = self.output_root / 'labels' / folder
        output_label_folder.mkdir(parents=True, exist_ok=True)
        
        patch_count = 0
        
        for idx, (patch_x, patch_y, patch_w, patch_h) in enumerate(patches):
            # Extract patch
            patch = img[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            
            # Generate patch filename
            patch_filename = f"{crop_path.stem}_patch_{idx:03d}"
            
            # Save patch image
            patch_img_path = output_img_folder / f"{patch_filename}.bmp"
            cv2.imwrite(str(patch_img_path), patch)
            
            # Transform and save mask
            patch_mask = {'polygons': []}
            
            if mask_data and 'polygons' in mask_data:
                for polygon in mask_data['polygons']:
                    transformed = self.transform_polygon(polygon, bbox_x, bbox_y, 
                                                        patch_x, patch_y, patch_w, patch_h)
                    if transformed:
                        patch_mask['polygons'].append(transformed)
            
            # Save mask JSON
            patch_mask_path = output_label_folder / f"{patch_filename}_mask.json"
            with open(patch_mask_path, 'w') as f:
                json.dump(patch_mask, f, indent=2)
            
            # Update stats
            if patch_mask['polygons']:
                self.stats['patches_with_labels'] += 1
            else:
                self.stats['patches_without_labels'] += 1
            
            patch_count += 1
        
        return patch_count
    
    def process_all(self):
        """Process all crops in all folders."""
        print("="*100)
        print(f"SPLITTING O-RING CROPS")
        print(f"Crops:   {self.crops_root}")
        print(f"Masks:   {self.masks_root}")
        print(f"Output:  {self.output_root}")
        print(f"Patch:   {self.patch_size}x{self.patch_size} (overlap={self.overlap}px)")
        print("="*100)
        
        folders = ['good', 'notok', 'model1defect', 'model1good']
        
        for folder in folders:
            folder_path = self.crops_root / folder
            
            if not folder_path.exists():
                print(f"\n{folder}: NOT FOUND")
                continue
            
            print(f"\n{folder}:")
            print("-"*100)
            
            # Get all images
            image_files = list(folder_path.glob('*.bmp')) + \
                         list(folder_path.glob('*.jpg')) + \
                         list(folder_path.glob('*.png'))
            
            print(f"Found {len(image_files)} images")
            
            folder_patches = 0
            for img_path in sorted(image_files):
                patches = self.split_crop_and_mask(img_path, folder)
                folder_patches += patches
                self.stats['total_images'] += 1
            
            self.stats['total_patches'] += folder_patches
            print(f"Created {folder_patches} patches ({folder_patches/len(image_files):.1f} per image)")
        
        # Print summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"Total images processed:      {self.stats['total_images']}")
        print(f"Total patches created:       {self.stats['total_patches']}")
        print(f"  Patches with labels:       {self.stats['patches_with_labels']}")
        print(f"  Patches without labels:    {self.stats['patches_without_labels']}")
        print(f"Average patches per image:   {self.stats['total_patches']/max(1, self.stats['total_images']):.1f}")
        print("="*100)


def main():
    CROPS_ROOT = r"F:\standard elastomers\oring_crops"
    MASKS_ROOT = r"F:\standard elastomers\Original Data\masks"
    OUTPUT_ROOT = r"F:\standard elastomers\oring_patches_640"
    
    # Clean output if exists
    output_path = Path(OUTPUT_ROOT)
    if output_path.exists():
        print(f"Cleaning existing output: {OUTPUT_ROOT}\n")
        shutil.rmtree(output_path)
    
    # Split using 640x640 patches (standard YOLOv8 size with overlap)
    splitter = CropSplitter(
        crops_root=CROPS_ROOT,
        masks_root=MASKS_ROOT,
        output_root=OUTPUT_ROOT,
        patch_size=640,
        overlap=100  # Overlap to ensure no data loss at edges
    )
    
    splitter.process_all()
    
    print(f"\n✓ Patches saved to: {OUTPUT_ROOT}")
    print(f"  - Images: {OUTPUT_ROOT}\\images\\<folder>")
    print(f"  - Labels: {OUTPUT_ROOT}\\labels\\<folder>")


if __name__ == "__main__":
    main()
