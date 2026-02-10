"""
Extract o-ring regions using Otsu thresholding with padding.
Analyzes crop sizes to determine optimal split strategy.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict
import shutil


class OringExtractor:
    def __init__(self,
                 source_root: str,
                 output_root: str,
                 padding: int = 100,
                 use_otsu: bool = True):
        """
        Initialize o-ring extractor.
        
        Args:
            source_root: Root with Original Data folder
            output_root: Output directory for crops
            padding: Padding to add around o-ring region (pixels)
            use_otsu: If True use Otsu, else use HSV
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.padding = padding
        self.use_otsu = use_otsu
        
        self.folders = ['good', 'notok', 'model1defect', 'model1good']
        
        self.stats = {
            folder: {
                'count': 0,
                'widths': [],
                'heights': [],
                'areas': []
            } for folder in self.folders
        }
    
    def detect_oring_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect o-ring region using Otsu or HSV thresholding.
        
        Returns: Binary mask (255=o-ring, 0=background)
        """
        if self.use_otsu:
            # Otsu's method
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # HSV Value channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            _, binary = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)
        
        # Find largest connected component (the o-ring)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return binary
        
        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask with only largest component
        mask = np.zeros_like(binary)
        mask[labels == largest_label] = 255
        
        return mask
    
    def get_oring_bbox_with_padding(self, mask: np.ndarray, image_shape: Tuple) -> Tuple[int, int, int, int]:
        """
        Get bounding box of o-ring with padding.
        
        Returns: (x, y, width, height)
        """
        coords = cv2.findNonZero(mask)
        if coords is None:
            return 0, 0, image_shape[1], image_shape[0]
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(image_shape[1] - x, w + 2 * self.padding)
        h = min(image_shape[0] - y, h + 2 * self.padding)
        
        return x, y, w, h
    
    def process_image(self, image_path: Path, folder: str) -> Dict:
        """Process a single image and extract o-ring region."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Detect o-ring mask
        mask = self.detect_oring_mask(img)
        
        # Get bounding box with padding
        x, y, w, h = self.get_oring_bbox_with_padding(mask, img.shape)
        
        # Extract crop
        crop = img[y:y+h, x:x+w]
        
        # Save crop
        output_folder = self.output_root / folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        output_path = output_folder / image_path.name
        cv2.imwrite(str(output_path), crop)
        
        # Save bounding box info as JSON for later use
        bbox_info_folder = self.output_root / 'bbox_info' / folder
        bbox_info_folder.mkdir(parents=True, exist_ok=True)
        bbox_info_path = bbox_info_folder / f"{image_path.stem}_bbox.json"
        
        bbox_data = {
            'filename': image_path.name,
            'original_width': img.shape[1],
            'original_height': img.shape[0],
            'bbox_x': int(x),
            'bbox_y': int(y),
            'bbox_width': int(w),
            'bbox_height': int(h),
            'crop_width': int(w),
            'crop_height': int(h)
        }
        
        with open(bbox_info_path, 'w') as f:
            json.dump(bbox_data, f, indent=2)
        
        # Also save mask visualization for verification
        mask_vis_folder = self.output_root / 'mask_visualization' / folder
        mask_vis_folder.mkdir(parents=True, exist_ok=True)
        
        # Draw bounding box on original image
        vis_img = img.copy()
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        mask_vis_path = mask_vis_folder / image_path.name
        cv2.imwrite(str(mask_vis_path), vis_img)
        
        # Update statistics
        self.stats[folder]['count'] += 1
        self.stats[folder]['widths'].append(w)
        self.stats[folder]['heights'].append(h)
        self.stats[folder]['areas'].append(w * h)
        
        return {
            'filename': image_path.name,
            'original_size': (img.shape[1], img.shape[0]),
            'crop_bbox': (x, y, w, h),
            'crop_size': (w, h)
        }
    
    def process_all(self):
        """Process all images in all folders."""
        method_name = "Otsu" if self.use_otsu else "HSV"
        print(f"O-ring Extraction using {method_name} thresholding")
        print(f"Source: {self.source_root}")
        print(f"Output: {self.output_root}")
        print(f"Padding: {self.padding}px")
        print("="*100)
        
        for folder in self.folders:
            folder_path = self.source_root / folder
            
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
            
            # Process all images
            for img_path in sorted(image_files):
                result = self.process_image(img_path, folder)
                if result:
                    if self.stats[folder]['count'] <= 3:  # Print first 3
                        print(f"  {result['filename']}: {result['original_size']} → {result['crop_size']}")
            
            print(f"Processed {self.stats[folder]['count']} images")
        
        # Print summary statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistics about extracted regions."""
        print("\n" + "="*100)
        print("EXTRACTION STATISTICS")
        print("="*100)
        
        for folder in self.folders:
            if self.stats[folder]['count'] == 0:
                continue
            
            widths = self.stats[folder]['widths']
            heights = self.stats[folder]['heights']
            areas = self.stats[folder]['areas']
            
            print(f"\n{folder}: ({self.stats[folder]['count']} images)")
            print(f"  Width:  min={min(widths):4d}  max={max(widths):4d}  avg={int(np.mean(widths)):4d}  std={int(np.std(widths)):4d}")
            print(f"  Height: min={min(heights):4d}  max={max(heights):4d}  avg={int(np.mean(heights)):4d}  std={int(np.std(heights)):4d}")
            print(f"  Area:   min={min(areas):7d}  max={max(areas):7d}  avg={int(np.mean(areas)):7d}")
            
            # Suggest split strategy
            avg_w = int(np.mean(widths))
            avg_h = int(np.mean(heights))
            
            print(f"\n  Suggested split strategy for avg size ({avg_w}x{avg_h}):")
            if avg_w <= 640 and avg_h <= 640:
                print(f"    → No split needed (resize to 640x640)")
            elif avg_w <= 1280 and avg_h <= 1280:
                patches_x = int(np.ceil(avg_w / 640))
                patches_y = int(np.ceil(avg_h / 640))
                print(f"    → Split into {patches_x}x{patches_y} grid (each ~640x640)")
            else:
                patches_x = int(np.ceil(avg_w / 640))
                patches_y = int(np.ceil(avg_h / 640))
                print(f"    → Split into {patches_x}x{patches_y} grid")
        
        print("\n" + "="*100)
        print(f"\nCrops saved to: {self.output_root}")
        print(f"Mask visualizations saved to: {self.output_root / 'mask_visualization'}")
        print("="*100)


def main():
    SOURCE_ROOT = r"F:\standard elastomers\Original Data"
    OUTPUT_ROOT = r"F:\standard elastomers\oring_crops"
    
    # Clean output directory if exists
    output_path = Path(OUTPUT_ROOT)
    if output_path.exists():
        print(f"Cleaning existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(output_path)
    
    # Extract using Otsu (best method)
    extractor = OringExtractor(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        padding=100,  # 100px padding on all sides
        use_otsu=True
    )
    
    extractor.process_all()


if __name__ == "__main__":
    main()
