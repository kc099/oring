"""
Analyze extracted o-ring crop sizes to determine optimal patch size for splitting.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List


def analyze_crop_sizes(crops_root: str) -> Dict:
    """
    Analyze all crop sizes to find max dimensions.
    
    Returns dict with statistics per folder and overall max.
    """
    crops_path = Path(crops_root)
    folders = ['good', 'notok', 'model1defect', 'model1good']
    
    all_widths = []
    all_heights = []
    
    stats = {}
    
    print("="*100)
    print("ANALYZING CROP SIZES")
    print("="*100)
    
    for folder in folders:
        folder_path = crops_path / folder
        
        if not folder_path.exists():
            print(f"\n{folder}: NOT FOUND")
            continue
        
        # Get all images
        image_files = list(folder_path.glob('*.bmp')) + \
                     list(folder_path.glob('*.jpg')) + \
                     list(folder_path.glob('*.png'))
        
        widths = []
        heights = []
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                all_widths.append(w)
                all_heights.append(h)
        
        if widths:
            stats[folder] = {
                'count': len(widths),
                'width_min': min(widths),
                'width_max': max(widths),
                'width_avg': int(np.mean(widths)),
                'height_min': min(heights),
                'height_max': max(heights),
                'height_avg': int(np.mean(heights))
            }
            
            print(f"\n{folder}: ({len(widths)} images)")
            print(f"  Width:  min={min(widths):4d}  max={max(widths):4d}  avg={int(np.mean(widths)):4d}")
            print(f"  Height: min={min(heights):4d}  max={max(heights):4d}  avg={int(np.mean(heights)):4d}")
    
    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    
    if all_widths:
        max_width = max(all_widths)
        max_height = max(all_heights)
        
        print(f"\nTotal images: {len(all_widths)}")
        print(f"Maximum width:  {max_width}")
        print(f"Maximum height: {max_height}")
        print(f"Average width:  {int(np.mean(all_widths))}")
        print(f"Average height: {int(np.mean(all_heights))}")
        
        # Suggest patch sizes
        print("\n" + "="*100)
        print("PATCH SIZE RECOMMENDATIONS")
        print("="*100)
        
        # Find optimal patch size that minimizes waste
        patch_sizes = [512, 640, 768, 800, 896, 1024]
        
        print(f"\nFor max size ({max_width}x{max_height}):\n")
        
        for patch_size in patch_sizes:
            patches_x = int(np.ceil(max_width / patch_size))
            patches_y = int(np.ceil(max_height / patch_size))
            total_patches = patches_x * patches_y
            
            # Calculate overlap needed to avoid losing data at edges
            if patches_x > 1:
                required_width = patch_size * patches_x
                overlap_x = (required_width - max_width) / (patches_x - 1)
            else:
                overlap_x = 0
            
            if patches_y > 1:
                required_height = patch_size * patches_y
                overlap_y = (required_height - max_height) / (patches_y - 1)
            else:
                overlap_y = 0
            
            print(f"  Patch {patch_size}x{patch_size}: {patches_x}x{patches_y} grid = {total_patches} patches/image")
            if overlap_x > 0 or overlap_y > 0:
                print(f"    → Overlap needed: {int(overlap_x)}px horizontal, {int(overlap_y)}px vertical")
        
        # Recommendation
        print("\n" + "-"*100)
        print("RECOMMENDATION:")
        print("-"*100)
        
        # Best options based on max dimensions
        if max_width <= 1600 and max_height <= 1600:
            print(f"\n  Option 1 (No overlap): Use 800x800 patches")
            print(f"    → 2x2 grid = 4 patches per image")
            print(f"    → Clean split, no overlap needed")
            
            print(f"\n  Option 2 (Standard YOLOv8): Use 640x640 patches")
            print(f"    → 3x3 grid = 9 patches per image (with ~100px overlap)")
            print(f"    → Standard input size for YOLOv8")
        else:
            patches_x_640 = int(np.ceil(max_width / 640))
            patches_y_640 = int(np.ceil(max_height / 640))
            print(f"\n  Use 640x640 patches with overlap")
            print(f"    → {patches_x_640}x{patches_y_640} grid = {patches_x_640*patches_y_640} patches per image")
        
        print("\n" + "="*100)
        
        return {
            'max_width': max_width,
            'max_height': max_height,
            'avg_width': int(np.mean(all_widths)),
            'avg_height': int(np.mean(all_heights)),
            'total_images': len(all_widths),
            'folder_stats': stats
        }
    
    return {}


def main():
    CROPS_ROOT = r"F:\standard elastomers\oring_crops"
    
    stats = analyze_crop_sizes(CROPS_ROOT)


if __name__ == "__main__":
    main()
