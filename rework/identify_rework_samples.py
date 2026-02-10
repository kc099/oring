"""
Identify rework samples based on concentricity and diameter tolerance.

Workflow:
1) Load image
2) Remove background using Otsu thresholding (foreground = o-ring)
3) Extract outer contour and inner contour
4) Fit circles to both contours
5) Compute metrics:
   - center distance (eccentricity) between inner/outer circles
   - average ring thickness (outer_radius - inner_radius)
   - mean radius vs reference
6) Save per-image measurements to CSV for manual calibration

Outputs:
- rework/measurements.csv

Author: GitHub Copilot
Date: February 6, 2026
"""

import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


def otsu_mask(image: np.ndarray) -> np.ndarray:
    """Create binary mask using Otsu and keep largest component."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background becomes white
    if np.mean(binary == 255) > 0.75:
        binary = cv2.bitwise_not(binary)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(binary)
    mask[labels == largest_label] = 255
    return mask


def find_inner_outer_contours(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find outer and inner contours from a ring mask."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, None

    # Find outer contour: largest area
    areas = [cv2.contourArea(c) for c in contours]
    outer_idx = int(np.argmax(areas))
    outer = contours[outer_idx]

    # Find inner contour: largest child of outer (hole)
    inner = None
    h = hierarchy[0]
    for i, hinfo in enumerate(h):
        parent = hinfo[3]
        if parent == outer_idx:
            # candidate inner
            if inner is None or cv2.contourArea(contours[i]) > cv2.contourArea(inner):
                inner = contours[i]

    return outer, inner


def fit_circle(contour: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Fit a circle to a contour using min enclosing circle."""
    (x, y), r = cv2.minEnclosingCircle(contour)
    return (float(x), float(y)), float(r)


def measure_ring(image_path: Path) -> Optional[dict]:
    """Measure ring metrics for a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    mask = otsu_mask(image)
    outer, inner = find_inner_outer_contours(mask)

    if outer is None or inner is None:
        return {
            'image': image_path.name,
            'outer_radius': None,
            'inner_radius': None,
            'center_dx': None,
            'center_dy': None,
            'center_dist': None,
            'ring_thickness': None,
            'mean_radius': None,
            'valid': False
        }

    (ox, oy), orad = fit_circle(outer)
    (ix, iy), irad = fit_circle(inner)

    center_dx = ox - ix
    center_dy = oy - iy
    center_dist = (center_dx ** 2 + center_dy ** 2) ** 0.5
    ring_thickness = orad - irad
    mean_radius = (orad + irad) / 2.0

    return {
        'image': image_path.name,
        'outer_radius': orad,
        'inner_radius': irad,
        'center_dx': center_dx,
        'center_dy': center_dy,
        'center_dist': center_dist,
        'ring_thickness': ring_thickness,
        'mean_radius': mean_radius,
        'valid': True
    }


def process_folder(input_dir: Path, output_csv: Path) -> None:
    """Process all images in a folder and write measurements CSV."""
    image_files = []
    for ext in ('.bmp', '.jpg', '.jpeg', '.png'):
        image_files.extend(input_dir.glob(f'*{ext}'))

    rows: List[dict] = []
    for img_path in sorted(image_files):
        result = measure_ring(img_path)
        if result is not None:
            rows.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            'image', 'outer_radius', 'inner_radius', 'center_dx', 'center_dy',
            'center_dist', 'ring_thickness', 'mean_radius', 'valid'
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Processed {len(rows)} images")
    print(f"Saved measurements: {output_csv}")


def main() -> None:
    # Change this to the folder you want to analyze
    INPUT_DIR = Path(r"F:\standard elastomers\Original Data\good")
    OUTPUT_CSV = Path(r"F:\standard elastomers\rework\measurements.csv")

    process_folder(INPUT_DIR, OUTPUT_CSV)


if __name__ == '__main__':
    main()
