"""
Mask R-CNN Inference for O-Ring Defect Detection.

Modes:
1) Single image inference with visualization
2) Batch inference on a folder
3) Full-resolution inference (split → predict → stitch)
4) Evaluation on test set with metrics

Usage:
    python inference.py --model model1 --image path/to/image.bmp
    python inference.py --model model1 --folder path/to/images/
    python inference.py --model model1 --evaluate
    python inference.py --model model1 --full-image path/to/2448x2048.bmp

Author: GitHub Copilot
Date: February 8, 2026
"""

import argparse
import json
import time
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import (
    OUTPUT_ROOT, ORING_MODELS, TRAINING_CONFIG, PROJECT_ROOT
)
from train import build_maskrcnn
from utils import draw_predictions, compute_iou_masks, compute_dice

# Reuse background subtraction from existing script
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess_background_split import BackgroundSubtractionSplitter


class OringDefectDetector:
    """
    Mask R-CNN inference wrapper for o-ring defect detection.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        mask_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold

        # Load model
        if checkpoint_path is None:
            checkpoint_path = OUTPUT_ROOT / model_name / "checkpoints" / "best_model.pth"
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        num_classes = checkpoint.get("config", {}).get("num_classes", 2)
        self.model = build_maskrcnn(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Device: {self.device}")
        print(f"  Epoch: {checkpoint.get('epoch', '?') + 1}")
        print(f"  Val loss: {checkpoint.get('best_val_loss', '?')}")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            dict with keys: boxes, masks, scores, labels
        """
        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Forward pass
        outputs = self.model([img_tensor])[0]

        # Filter by score
        keep = outputs["scores"] >= self.score_threshold
        boxes = outputs["boxes"][keep].cpu().numpy()
        scores = outputs["scores"][keep].cpu().numpy()
        labels = outputs["labels"][keep].cpu().numpy()

        # Process masks
        if "masks" in outputs and keep.sum() > 0:
            masks = outputs["masks"][keep]
            masks = (masks.squeeze(1) > self.mask_threshold).cpu().numpy().astype(np.uint8)
        else:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

        return {
            "boxes": boxes,
            "masks": masks,
            "scores": scores,
            "labels": labels,
            "num_detections": len(scores),
            "has_defect": len(scores) > 0
        }

    def predict_and_visualize(
        self,
        image: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Run inference and draw results on image."""
        result = self.predict(image)

        vis = draw_predictions(
            image,
            result["boxes"],
            result["masks"],
            result["scores"],
            result["labels"],
            score_threshold=self.score_threshold
        )

        if save_path:
            cv2.imwrite(str(save_path), vis)

        return vis, result

    def predict_full_image(
        self,
        image_path: Path,
        patch_size: int = 640,
        overlap: int = 100,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run inference on a full-resolution image (2448x2048) by:
        1) Background subtraction
        2) Split into 640x640 patches
        3) Predict on each patch
        4) Stitch results back to full image

        Returns:
            combined_mask: (H, W) full-resolution defect mask
            results: dict with per-patch and overall stats
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        h, w = image.shape[:2]

        # Background subtraction (reuse existing class)
        splitter = BackgroundSubtractionSplitter(
            source_root=".",  # not used for single image
            output_root=".",  # not used for single image
            patch_size=patch_size,
            overlap=overlap,
            fill_holes=True
        )
        mask = splitter._otsu_mask(image)
        fg_image = splitter._subtract_background(image, mask)

        # Calculate patches
        patches = splitter._calculate_patches(w, h)

        # Predict on each patch
        combined_mask = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        all_detections = []

        for idx, (px, py, pw, ph) in enumerate(patches):
            patch = fg_image[py:py+ph, px:px+pw]

            # Pad to patch_size if needed
            if pw < patch_size or ph < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            result = self.predict(patch)

            # Map masks back to full image
            for i in range(result["num_detections"]):
                patch_mask = result["masks"][i][:ph, :pw]
                combined_mask[py:py+ph, px:px+pw] = np.maximum(
                    combined_mask[py:py+ph, px:px+pw],
                    patch_mask.astype(np.float32) * result["scores"][i]
                )

                # Adjust boxes to full image coordinates
                box = result["boxes"][i].copy()
                box[0] += px
                box[1] += py
                box[2] += px
                box[3] += py

                all_detections.append({
                    "patch_idx": idx,
                    "box": box.tolist(),
                    "score": float(result["scores"][i]),
                    "mask_area": int(patch_mask.sum())
                })

            count_map[py:py+ph, px:px+pw] += 1.0

        # Normalize overlapping regions
        count_map = np.maximum(count_map, 1.0)

        # Binarize combined mask
        binary_mask = (combined_mask > self.mask_threshold).astype(np.uint8)

        # Create visualization
        vis = image.copy()
        overlay = vis.copy()
        overlay[binary_mask > 0] = (0, 0, 255)  # Red for defects
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Draw bounding boxes
        for det in all_detections:
            box = np.array(det["box"]).astype(int)
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(vis, f'{det["score"]:.2f}', (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        results = {
            "num_patches": len(patches),
            "total_detections": len(all_detections),
            "has_defect": len(all_detections) > 0,
            "defect_area_pixels": int(binary_mask.sum()),
            "defect_area_percent": float(binary_mask.sum()) / (h * w) * 100,
            "detections": all_detections
        }

        return vis, binary_mask, results


# ─── CLI Commands ────────────────────────────────────────────────────────────

def infer_single(args):
    """Inference on a single image."""
    detector = OringDefectDetector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        score_threshold=args.threshold
    )

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: Cannot read {args.image}")
        return

    vis, result = detector.predict_and_visualize(image)

    # Save result
    out_dir = OUTPUT_ROOT / args.model / "inference_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_result.png"
    cv2.imwrite(str(out_path), vis)

    print(f"\nResult: {'DEFECT DETECTED' if result['has_defect'] else 'OK (no defect)'}")
    print(f"  Detections: {result['num_detections']}")
    if result['num_detections'] > 0:
        print(f"  Scores: {result['scores']}")
    print(f"  Saved: {out_path}")


def infer_folder(args):
    """Batch inference on a folder of images."""
    detector = OringDefectDetector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        score_threshold=args.threshold
    )

    folder = Path(args.folder)
    image_files = sorted(
        list(folder.glob("*.bmp")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.png"))
    )

    if not image_files:
        print(f"No images found in {folder}")
        return

    out_dir = OUTPUT_ROOT / args.model / "inference_results" / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    defect_count = 0
    total = len(image_files)

    print(f"\nProcessing {total} images from {folder}")
    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        vis, result = detector.predict_and_visualize(image)
        out_path = out_dir / f"{img_path.stem}_result.png"
        cv2.imwrite(str(out_path), vis)

        if result["has_defect"]:
            defect_count += 1

        results.append({
            "filename": img_path.name,
            "has_defect": result["has_defect"],
            "num_detections": result["num_detections"],
            "scores": result["scores"].tolist() if len(result["scores"]) > 0 else []
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} ({defect_count} defects found)")

    # Save summary
    summary = {
        "total_images": total,
        "defect_images": defect_count,
        "ok_images": total - defect_count,
        "defect_rate": defect_count / max(total, 1),
        "results": results
    }
    with open(out_dir / "inference_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Defects: {defect_count}/{total} ({defect_count/max(total,1)*100:.1f}%)")
    print(f"Results saved to {out_dir}")


def infer_full_image(args):
    """Inference on full-resolution 2448x2048 image."""
    detector = OringDefectDetector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        score_threshold=args.threshold
    )

    vis, binary_mask, results = detector.predict_full_image(Path(args.full_image))

    out_dir = OUTPUT_ROOT / args.model / "inference_results" / "full_res"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.full_image).stem
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), vis)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), binary_mask * 255)

    with open(out_dir / f"{stem}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResult: {'DEFECT' if results['has_defect'] else 'OK'}")
    print(f"  Total detections: {results['total_detections']}")
    print(f"  Defect area: {results['defect_area_percent']:.2f}%")
    print(f"  Saved to: {out_dir}")


def evaluate_test(args):
    """Evaluate on test set."""
    detector = OringDefectDetector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        score_threshold=args.threshold
    )

    model_dir = OUTPUT_ROOT / args.model
    test_img_dir = model_dir / "images" / "test"
    test_ann_path = model_dir / "annotations" / "test.json"

    if not test_ann_path.exists():
        print(f"ERROR: {test_ann_path} not found")
        return

    with open(test_ann_path, 'r') as f:
        coco = json.load(f)

    # Group annotations by image_id
    img_anns = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        img_anns.setdefault(img_id, []).append(ann)

    tp = fp = fn = tn = 0
    all_ious = []

    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_path = test_img_dir / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = detector.predict(image)
        has_gt = img_id in img_anns and len(img_anns[img_id]) > 0

        if has_gt and result["has_defect"]:
            tp += 1
        elif has_gt and not result["has_defect"]:
            fn += 1
        elif not has_gt and result["has_defect"]:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"  Total images: {total}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"{'='*60}")

    # Save metrics
    metrics = {
        "total": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
    }
    with open(model_dir / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN inference for o-ring defect detection")
    parser.add_argument("--model", type=str, required=True,
                        choices=["model1", "model2", "combined"],
                        help="O-ring model to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best_model.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for detections")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    group.add_argument("--full-image", type=str, help="Path to full-res 2448x2048 image")
    group.add_argument("--evaluate", action="store_true", help="Evaluate on test set")

    args = parser.parse_args()

    if args.image:
        infer_single(args)
    elif args.folder:
        infer_folder(args)
    elif args.full_image:
        infer_full_image(args)
    elif args.evaluate:
        evaluate_test(args)


if __name__ == "__main__":
    main()
