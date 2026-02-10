"""
Advanced o-ring detection using various image processing techniques.
Tests different methods to see which works best for isolating the o-ring.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def method1_simple_threshold(image, threshold=15):
    """Simple thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def method2_adaptive_threshold(image):
    """Adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)
    return binary


def method3_otsu(image):
    """Otsu's thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method4_hsv_based(image):
    """HSV color space thresholding."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Focus on value channel (brightness)
    v_channel = hsv[:, :, 2]
    _, binary = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)
    return binary


def method5_lab_based(image):
    """LAB color space thresholding."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    _, binary = cv2.threshold(l_channel, 30, 255, cv2.THRESH_BINARY)
    return binary


def method6_morphology(image):
    """Morphological operations to find structure."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened


def method7_edge_based(image):
    """Edge detection to find o-ring boundary."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Edge detection
    edges = cv2.Canny(enhanced, 30, 100)
    
    # Dilate to connect edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Fill enclosed regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dilated)
    cv2.drawContours(filled, contours, -1, 255, -1)
    
    return filled


def find_largest_component(binary):
    """Find the largest connected component."""
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        return binary
    
    # Find largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only largest component
    mask = np.zeros_like(binary)
    mask[labels == largest_label] = 255
    
    return mask


def get_content_bbox(binary):
    """Get bounding box of white pixels."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def test_methods(image_path):
    """Test all methods on an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load: {image_path}")
        return
    
    print(f"\nTesting: {image_path.name}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    methods = [
        ("1. Simple Threshold (15)", method1_simple_threshold(img, 15)),
        ("2. Adaptive Threshold", method2_adaptive_threshold(img)),
        ("3. Otsu", method3_otsu(img)),
        ("4. HSV Value", method4_hsv_based(img)),
        ("5. LAB Lightness", method5_lab_based(img)),
        ("6. Morphology", method6_morphology(img)),
        ("7. Edge-based", method7_edge_based(img))
    ]
    
    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    for idx, (name, binary) in enumerate(methods):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        # Get largest component
        largest = find_largest_component(binary)
        
        # Get bounding box
        bbox = get_content_bbox(largest)
        
        # Display
        axes[row, col].imshow(largest, cmap='gray')
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
        
        if bbox:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[row, col].add_patch(rect)
            
            # Calculate percentage
            white_pixels = np.count_nonzero(largest)
            total_pixels = largest.size
            percentage = white_pixels / total_pixels * 100
            
            print(f"{name}: BBox=({x},{y},{w},{h}), Content={percentage:.1f}%")
    
    # Hide unused subplot
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'oring_detection_test_{image_path.stem}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: oring_detection_test_{image_path.stem}.png")
    plt.show()


def main():
    source_root = Path(r"F:\standard elastomers\Original Data")
    
    # Test on one image from each folder
    test_images = [
        source_root / 'good' / 'Image_20260130155645509.bmp',
        source_root / 'notok' / 'Image_20260130122214876.bmp',
        source_root / 'model1defect' / 'Image_20260130144046195.bmp',
        source_root / 'model1good' / 'Image_20260130150708473.bmp'
    ]
    
    for img_path in test_images:
        if img_path.exists():
            test_methods(img_path)
        else:
            print(f"Not found: {img_path}")


if __name__ == "__main__":
    main()
