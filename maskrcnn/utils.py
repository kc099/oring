"""
Utility functions for Mask R-CNN O-Ring Defect Detection.

- JSON polygon label I/O
- COCO format conversion
- Visualization helpers
- Metric computation

Author: GitHub Copilot
Date: February 8, 2026
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pycocotools import mask as mask_utils


# ─── Label I/O ───────────────────────────────────────────────────────────────

def load_json_label(label_path: Path) -> dict:
    """Load a JSON polygon label file."""
    with open(label_path, 'r') as f:
        return json.load(f)


def polygons_from_label(label: dict) -> List[List[Tuple[int, int]]]:
    """
    Extract polygon point lists from a JSON label.
    Returns list of polygons, each polygon is a list of (x, y) tuples.
    """
    polygons = []
    for poly in label.get("polygons", []):
        points = [(p["x"], p["y"]) for p in poly["points"]]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def polygon_to_mask(polygon: List[Tuple[int, int]], height: int, width: int) -> np.ndarray:
    """Convert a single polygon to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_to_coco_segmentation(polygon: List[Tuple[int, int]]) -> List[float]:
    """Convert polygon points to COCO segmentation format [x1,y1,x2,y2,...,xn,yn]."""
    flat = []
    for x, y in polygon:
        flat.extend([float(x), float(y)])
    return flat


def polygon_to_bbox(polygon: List[Tuple[int, int]]) -> List[float]:
    """Compute bounding box [x, y, width, height] from polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def polygon_area(polygon: List[Tuple[int, int]]) -> float:
    """Compute area using the Shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


# ─── COCO Format ─────────────────────────────────────────────────────────────

def build_coco_dataset(
    image_dir: Path,
    label_dir: Path,
    image_files: List[Path],
    category_name: str = "defect"
) -> dict:
    """
    Build a COCO-format annotation dict from image files and JSON labels.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing JSON label files
        image_files: List of image file paths
        category_name: Name of the defect category

    Returns:
        COCO-format dict with images, annotations, categories
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": category_name, "supercategory": "defect"}
        ]
    }

    ann_id = 1
    for img_id, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        # Look for matching label
        label_path = label_dir / (img_path.stem + ".json")
        if not label_path.exists():
            continue

        label = load_json_label(label_path)
        polygons = polygons_from_label(label)

        for poly in polygons:
            seg = polygon_to_coco_segmentation(poly)
            bbox = polygon_to_bbox(poly)
            area = polygon_area(poly)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [seg],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    return coco


# ─── Visualization ───────────────────────────────────────────────────────────

COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def draw_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_threshold: float = 0.5,
    mask_alpha: float = 0.4
) -> np.ndarray:
    """
    Draw predicted bounding boxes and masks on an image.

    Args:
        image: BGR image (H, W, 3)
        boxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        labels: (N,) class labels
        score_threshold: minimum score to display
        mask_alpha: transparency of mask overlay

    Returns:
        Annotated image
    """
    vis = image.copy()

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        color = COLORS[i % len(COLORS)]
        score = scores[i]

        # Draw mask
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
            overlay = vis.copy()
            overlay[mask > 0.5] = color
            vis = cv2.addWeighted(vis, 1 - mask_alpha, overlay, mask_alpha, 0)

        # Draw box
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_text = f"defect {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_ground_truth(
    image: np.ndarray,
    polygons: List[List[Tuple[int, int]]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw ground truth polygons on an image."""
    vis = image.copy()
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, thickness)
    return vis


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_iou_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    if total == 0:
        return 0.0
    return float(2 * intersection) / float(total)
