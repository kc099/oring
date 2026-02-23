"""
PyTorch Dataset for COCO-format O-Ring Defect Detection.

Loads images + polygon annotations in COCO format and returns
tensors compatible with torchvision's Mask R-CNN.

Author: GitHub Copilot
Date: February 8, 2026
"""

import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors
from pycocotools.coco import COCO

from utils import polygon_to_mask


class OringDefectDataset(Dataset):
    """
    Dataset for o-ring defect detection with Mask R-CNN.

    Each sample returns:
        image:  (3, H, W) float tensor, normalized to [0, 1]
        target: dict with keys:
            - boxes:    (N, 4) float tensor [x1, y1, x2, y2]
            - labels:   (N,) int64 tensor (class ids)
            - masks:    (N, H, W) uint8 tensor
            - image_id: int tensor
            - area:     (N,) float tensor
            - iscrowd:  (N,) int64 tensor
    """

    def __init__(
        self,
        image_dir: Path,
        annotation_file: Path,
        transforms=None,
        binary_mode: bool = True
    ):
        """
        Args:
            image_dir: Directory containing images for this split
            annotation_file: Path to COCO JSON annotation file
            transforms: torchvision v2 transforms (applied to image + target)
            binary_mode: If True, collapse all categories to 1 (defect)
        """
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.binary_mode = binary_mode

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build lookup structures
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = list(self.images.keys())

        # Group annotations by image_id
        self.img_to_anns: Dict[int, list] = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self.img_to_anns:
                self.img_to_anns[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.image_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Get annotations for this image
        anns = self.img_to_anns[img_id]

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Parse segmentation → mask
            seg = ann["segmentation"]
            if isinstance(seg, list) and len(seg) > 0:
                # Polygon format: [[x1,y1,x2,y2,...]]
                flat_pts = seg[0]
                polygon = [(flat_pts[i], flat_pts[i+1])
                           for i in range(0, len(flat_pts), 2)]
                mask = polygon_to_mask(polygon, h, w)
            else:
                continue

            # Compute tight bbox from mask
            pos = np.where(mask > 0)
            if len(pos[0]) == 0:
                continue
            y_min, y_max = pos[0].min(), pos[0].max()
            x_min, x_max = pos[1].min(), pos[1].max()

            # Skip degenerate boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            cat_id = ann.get("category_id", 1)
            labels.append(1 if self.binary_mode else cat_id)
            masks.append(mask)
            areas.append(ann.get("area", float((x_max - x_min) * (y_max - y_min))))
            iscrowd.append(ann.get("iscrowd", 0))

        # Handle images with no annotations (good images)
        if len(boxes) == 0:
            boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            )
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8))
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            )
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = tv_tensors.Mask(torch.as_tensor(np.array(masks), dtype=torch.uint8))
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # Convert image to tv_tensors.Image so v2 transforms
        # can jointly transform image + boxes + masks together
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image = tv_tensors.Image(image)

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def get_train_transforms(cfg) -> T.Compose:
    """Training augmentations using torchvision v2 transforms.

    Includes random rotation so the model learns to detect defects
    regardless of part orientation (addresses field observation that
    predictions vary when the same o-ring is rotated in place).

    Requires that image is wrapped in ``tv_tensors.Image``, boxes in
    ``tv_tensors.BoundingBoxes``, and masks in ``tv_tensors.Mask``
    so that v2 transforms apply the *same* geometric operation to
    image, boxes, and masks jointly.
    """
    transforms = [
        T.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
        T.RandomVerticalFlip(p=cfg.vertical_flip_prob),
    ]
    # Random rotation — full 360° to handle any placement orientation
    rot_deg = getattr(cfg, "rotation_degrees", 15)
    if rot_deg > 0:
        transforms.append(
            T.RandomRotation(
                degrees=rot_deg,
                interpolation=T.InterpolationMode.BILINEAR,
                expand=False,          # keep original canvas size
            )
        )
    if cfg.brightness_jitter > 0 or cfg.contrast_jitter > 0:
        transforms.append(
            T.ColorJitter(
                brightness=cfg.brightness_jitter,
                contrast=cfg.contrast_jitter
            )
        )
    # Clamp boxes inside canvas and remove degenerate ones
    # (rotation can push boxes partially outside the image)
    transforms.append(T.ClampBoundingBoxes())
    transforms.append(T.SanitizeBoundingBoxes(min_size=1))
    return T.Compose(transforms)


def get_val_transforms() -> None:
    """Validation: no augmentation."""
    return None


def collate_fn(batch):
    """Custom collate that handles variable-size targets."""
    return tuple(zip(*batch))
