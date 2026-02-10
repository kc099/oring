"""
Script to convert JSON masks to YOLO segmentation format.
Processes notok and model1defect folders from split_dataset.
Creates YOLO dataset with images and normalized polygon coordinates.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import random


def normalize_polygon_points(points: List[Dict], image_width: int, image_height: int) -> List[Tuple[float, float]]:
    """
    Normalize polygon points to YOLO format (0-1 range).
    
    Args:
        points: List of points with 'x' and 'y' keys
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of normalized (x, y) tuples in range [0, 1]
    """
    normalized_points = []
    for point in points:
        x_norm = point['x'] / image_width
        y_norm = point['y'] / image_height
        # Clamp values to [0, 1] range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized_points.append((x_norm, y_norm))
    return normalized_points


def write_yolo_label(output_path: str, polygons: List[Dict], image_width: int, image_height: int) -> bool:
    """
    Write YOLO format label file with normalized polygon coordinates.
    
    Args:
        output_path: Path to write the label file
        polygons: List of polygon objects from JSON
        image_width: Image width
        image_height: Image height
    
    Returns:
        True if polygons were written, False if empty (no polygons)
    """
    if not polygons or len(polygons) == 0:
        return False
    
    with open(output_path, 'w') as f:
        for polygon in polygons:
            points = polygon.get('points', [])
            if len(points) == 0:
                continue
            
            # Normalize points
            normalized_points = normalize_polygon_points(points, image_width, image_height)
            
            # YOLO format: class_id x1 y1 x2 y2 ... xn yn
            # Using class 0 for defects
            line = "0"  # class_id
            for x, y in normalized_points:
                line += f" {x:.6f} {y:.6f}"
            
            f.write(line + "\n")
    
    return True


def process_dataset(
    source_image_folders: List[str],
    source_mask_folders: List[str],
    output_dir: str
) -> Dict[str, int]:
    """
    Process images and masks, converting to YOLO format.
    Keeps all defect images with non-empty masks and samples good images
    to be half of the defect count.
    
    Args:
        source_image_folders: List of source image folder paths
        source_mask_folders: List of source mask folder paths
        output_dir: Output directory for YOLO dataset
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_images': 0,
        'images_with_labels': 0,
        'images_without_labels': 0,
        'empty_labels_created': 0,
        'errors': 0
    }
    
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Build a mapping of image names to mask files
    mask_mapping = {}
    for mask_folder in source_mask_folders:
        if not os.path.exists(mask_folder):
            print(f"Warning: Mask folder not found: {mask_folder}")
            continue
        
        for mask_file in os.listdir(mask_folder):
            if mask_file.endswith('_mask.json'):
                mask_path = os.path.join(mask_folder, mask_file)
                # Extract image name from mask file
                # e.g., Image_20260130144046195_patch_004_mask.json -> Image_20260130144046195_patch_004
                image_name = mask_file.replace('_mask.json', '')
                mask_mapping[image_name] = mask_path
    
    print(f"Found {len(mask_mapping)} mask files")

    # Build image lists and select samples
    defect_folder_names = {'notok', 'model1defect'}
    good_folder_names = {'good', 'model1good'}

    def list_image_files(folder_path: str) -> List[str]:
        return [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))
        ]

    def has_non_empty_polygons(mask_path: str) -> bool:
        try:
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)
            polygons = mask_data.get('polygons', [])
            return len(polygons) > 0
        except Exception:
            return False

    selected_images_by_folder: Dict[str, List[str]] = {}
    defect_total = 0

    # Select all defect images that have non-empty masks
    for image_folder in source_image_folders:
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue

        folder_name = os.path.basename(image_folder).lower()
        if folder_name not in defect_folder_names:
            continue

        image_files = list_image_files(image_folder)
        selected = []
        for image_file in image_files:
            image_base_name = os.path.splitext(image_file)[0]
            mask_path = mask_mapping.get(image_base_name)
            if not mask_path:
                continue
            if has_non_empty_polygons(mask_path):
                selected.append(image_file)

        selected_images_by_folder[image_folder] = selected
        defect_total += len(selected)

    # Select good images to be half of defect samples
    good_target = defect_total // 2
    good_pool: List[Tuple[str, str]] = []
    for image_folder in source_image_folders:
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue

        folder_name = os.path.basename(image_folder).lower()
        if folder_name not in good_folder_names:
            continue

        image_files = list_image_files(image_folder)
        for image_file in image_files:
            good_pool.append((image_folder, image_file))

    if good_target > 0 and len(good_pool) > 0:
        good_target = min(good_target, len(good_pool))
        selected_good = random.sample(good_pool, good_target)
    else:
        selected_good = []

    for image_folder, image_file in selected_good:
        selected_images_by_folder.setdefault(image_folder, []).append(image_file)

    print(f"Selected {defect_total} defect images with non-empty masks")
    print(f"Selected {sum(len(v) for v in selected_images_by_folder.values()) - defect_total} good images")

    # Process selected images
    for image_folder in source_image_folders:
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue
        
        print(f"\nProcessing images from: {image_folder}")

        image_files = selected_images_by_folder.get(image_folder, [])
        if not image_files:
            print("  (no selected images)")
            continue

        for image_file in image_files:
            if not image_file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                continue
            
            stats['total_images'] += 1
            image_path = os.path.join(image_folder, image_file)
            image_base_name = os.path.splitext(image_file)[0]
            
            # Copy image to output directory
            output_image_path = os.path.join(images_dir, image_file)
            try:
                shutil.copy2(image_path, output_image_path)
            except Exception as e:
                print(f"Error copying image {image_file}: {e}")
                stats['errors'] += 1
                continue
            
            # Check if mask exists for this image
            if image_base_name in mask_mapping:
                mask_path = mask_mapping[image_base_name]
                
                try:
                    with open(mask_path, 'r') as f:
                        mask_data = json.load(f)
                    
                    image_width = mask_data.get('image_width')
                    image_height = mask_data.get('image_height')
                    polygons = mask_data.get('polygons', [])

                    # Fallback to image size if missing in mask JSON
                    if image_width is None or image_height is None:
                        img = cv2.imread(image_path)
                        if img is not None:
                            image_height, image_width = img.shape[:2]
                    
                    # Write YOLO label
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    output_label_path = os.path.join(labels_dir, label_file)
                    
                    has_polygons = write_yolo_label(output_label_path, polygons, image_width, image_height)
                    
                    if has_polygons:
                        stats['images_with_labels'] += 1
                        print(f"✓ {image_file} -> {label_file} (with segments)")
                    else:
                        # Create empty label file
                        with open(output_label_path, 'w') as f:
                            f.write("")
                        stats['images_without_labels'] += 1
                        stats['empty_labels_created'] += 1
                        print(f"✓ {image_file} -> {label_file} (empty - no segments)")
                
                except Exception as e:
                    print(f"Error processing mask for {image_file}: {e}")
                    # Create empty label file as fallback
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    output_label_path = os.path.join(labels_dir, label_file)
                    with open(output_label_path, 'w') as f:
                        f.write("")
                    stats['images_without_labels'] += 1
                    stats['empty_labels_created'] += 1
                    stats['errors'] += 1
            else:
                # No mask for this image - create empty label
                label_file = os.path.splitext(image_file)[0] + '.txt'
                output_label_path = os.path.join(labels_dir, label_file)
                with open(output_label_path, 'w') as f:
                    f.write("")
                stats['images_without_labels'] += 1
                stats['empty_labels_created'] += 1
                print(f"✓ {image_file} -> {label_file} (empty - no mask found)")
    
    return stats


def main():
    """Main preprocessing function."""
    # Define paths
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')

    source_image_folders = [
        os.path.join(dataset_dir, 'images', 'model1defect'),
        os.path.join(dataset_dir, 'images', 'notok'),
        os.path.join(dataset_dir, 'images', 'good'),
        os.path.join(dataset_dir, 'images', 'model1good')
    ]

    source_mask_folders = [
        os.path.join(dataset_dir, 'labels', 'model1defect'),
        os.path.join(dataset_dir, 'labels', 'notok'),
        os.path.join(dataset_dir, 'labels', 'good'),
        os.path.join(dataset_dir, 'labels', 'model1good')
    ]

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset')

    print("=" * 80)
    print("YOLO Segmentation Dataset Preprocessing")
    print("=" * 80)
    print(f"\nSource image folders:")
    for folder in source_image_folders:
        print(f"  - {folder}")
    
    print(f"\nSource mask folders:")
    for folder in source_mask_folders:
        print(f"  - {folder}")
    
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "=" * 80)
    print("Processing...\n")
    
    # Process dataset
    stats = process_dataset(
        source_image_folders,
        source_mask_folders,
        output_dir
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Total images processed: {stats['total_images']}")
    print(f"  Images with segments: {stats['images_with_labels']}")
    print(f"  Images without segments: {stats['images_without_labels']}")
    print(f"  Empty label files created: {stats['empty_labels_created']}")
    print(f"  Errors encountered: {stats['errors']}")
    print(f"\nYOLO dataset saved to: {output_dir}")
    print(f"  - Images: {os.path.join(output_dir, 'images')}")
    print(f"  - Labels: {os.path.join(output_dir, 'labels')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
