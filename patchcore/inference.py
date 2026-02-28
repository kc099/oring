"""
PatchCore Inference Script
==========================
Load a trained PatchCore model and run inference on images.
Supports single-image, directory, and batch inference with
optional anomaly map visualization.

Usage:
    # Evaluate a saved model on a directory
    python -m patchcore.inference \
        --model-path patchcore/results/model1_resnet50/model1_resnet50_patchcore.pkl \
        --image-dir  binned/model1defect \
        --backbone   resnet50 \
        --visualize

    # Single image inference
    python -m patchcore.inference \
        --model-path patchcore/results/model2_resnet101/model2_resnet101_patchcore.pkl \
        --image      binned/notok/some_image.bmp \
        --backbone   resnet101

    # Set a custom anomaly threshold
    python -m patchcore.inference \
        --model-path patchcore/results/model1_resnet50/model1_resnet50_patchcore.pkl \
        --image-dir  binned/model1defect \
        --backbone   resnet50 \
        --threshold  5.0

Author: GitHub Copilot
Date:   February 27, 2026
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import (
    ModelConfig,
    RESIZE_SIZE,
    CENTER_CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    WORKSPACE,
)
from .dataset import get_transform, OringDataset
from .patchcore_model import PatchCore


def load_model(model_path: str, backbone: str = "resnet50") -> PatchCore:
    """Load a trained PatchCore model from a pickle file.

    Restores resize/center_crop from the saved state so inference uses
    the same image size as training automatically.
    """
    import pickle
    with open(model_path, "rb") as f:
        state = pickle.load(f)
    cfg = ModelConfig(
        name=f"inference_{backbone}",
        backbone=state.get("backbone", backbone),
        resize=state.get("resize", RESIZE_SIZE),
        center_crop=state.get("center_crop", CENTER_CROP_SIZE),
        output_dir=Path(model_path).parent,
    )
    model = PatchCore(cfg)
    model.load(Path(model_path))
    print(f"    Image size: resize={cfg.resize} → crop={cfg.center_crop}")
    return model


def visualize_anomaly_map(image_path: str,
                          anomaly_map: np.ndarray,
                          score: float,
                          save_path: Optional[Path] = None,
                          threshold: Optional[float] = None) -> np.ndarray:
    """Overlay anomaly heatmap on the original image.

    Returns
    -------
    overlay : (H, W, 3) uint8 BGR image
    """
    # Read and resize original image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img[16:240, 16:240]  # center crop to 224x224

    # Normalize anomaly map to 0-255
    H, W = anomaly_map.shape
    amap = anomaly_map.copy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255
    amap = amap.astype(np.uint8)

    # Resize anomaly map to image size
    amap_resized = cv2.resize(amap, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Add score text
    status = "ANOMALY" if (threshold and score > threshold) else "OK" if threshold else ""
    color = (0, 0, 255) if status == "ANOMALY" else (0, 255, 0)
    text = f"Score: {score:.3f} {status}"
    cv2.putText(overlay, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), overlay)

    return overlay


def infer_directory(model: PatchCore,
                    image_dir: str,
                    output_dir: Optional[str] = None,
                    threshold: Optional[float] = None,
                    visualize: bool = False) -> dict:
    """Run inference on all images in a directory."""
    tfm = get_transform(model.cfg.resize, model.cfg.center_crop)
    ds = OringDataset(Path(image_dir), label=0, label_name="infer", transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=model.cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"\n  Inference on {len(ds)} images from {image_dir}")
    t0 = time.time()
    scores, _, paths, maps = model.predict(loader)
    elapsed = time.time() - t0

    # Optional visualization
    if visualize and output_dir:
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for i, (path, score, amap) in enumerate(zip(paths, scores, maps)):
            fname = Path(path).stem + "_anomaly.png"
            visualize_anomaly_map(path, amap, score,
                                  save_path=vis_dir / fname,
                                  threshold=threshold)
        print(f"  Visualizations saved → {vis_dir}")

    # Summary
    results = {
        "image_dir": str(image_dir),
        "n_images": len(scores),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "inference_time_s": round(elapsed, 2),
        "per_image_ms": round(elapsed / len(scores) * 1000, 1),
    }

    if threshold is not None:
        n_anomaly = int((scores > threshold).sum())
        results["threshold"] = threshold
        results["n_anomaly"] = n_anomaly
        results["n_good"] = len(scores) - n_anomaly
        results["anomaly_rate"] = round(n_anomaly / len(scores), 4)

    # Per-image scores
    per_image = []
    for path, score in zip(paths, scores):
        entry = {"file": Path(path).name, "score": round(float(score), 5)}
        if threshold is not None:
            entry["anomaly"] = bool(score > threshold)
        per_image.append(entry)
    results["per_image"] = per_image

    # Save results
    if output_dir:
        out_path = Path(output_dir) / "inference_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {out_path}")

    print(f"  Scores: mean={results['score_mean']:.4f}  "
          f"std={results['score_std']:.4f}  "
          f"min={results['score_min']:.4f}  max={results['score_max']:.4f}")
    if threshold:
        print(f"  Threshold={threshold}  anomalies={results.get('n_anomaly', '?')}"
              f"/{results['n_images']}")
    print(f"  Time: {elapsed:.2f}s ({results['per_image_ms']}ms/image)")

    return results


def infer_single_image(model: PatchCore,
                       image_path: str,
                       threshold: Optional[float] = None,
                       visualize: bool = False,
                       output_dir: Optional[str] = None) -> dict:
    """Run inference on a single image."""
    tfm = get_transform(model.cfg.resize, model.cfg.center_crop)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = tfm(img_rgb).unsqueeze(0)  # (1, 3, crop, crop)

    # Use model device
    tensor = tensor.to(model.device)

    with torch.no_grad():
        feat_maps = model.extractor(tensor)
        from .patchcore_model import aggregate_features
        patches = aggregate_features(feat_maps)
        patches_gpu = patches.to(model.device)
        bank = torch.from_numpy(model.memory_bank).to(model.device)
        dists = torch.cdist(patches_gpu, bank)
        topk, _ = dists.topk(model.cfg.n_neighbors, dim=1, largest=False)
        min_dists = topk.mean(dim=1).cpu().numpy()

    H, W = model.spatial_shape
    amap = min_dists.reshape(H, W)
    from scipy.ndimage import gaussian_filter
    amap = gaussian_filter(amap, sigma=4)
    score = float(amap.max())

    status = "ANOMALY" if (threshold and score > threshold) else "OK" if threshold else ""
    print(f"  {Path(image_path).name}:  score={score:.4f}  {status}")

    if visualize and output_dir:
        vis_path = Path(output_dir) / (Path(image_path).stem + "_anomaly.png")
        visualize_anomaly_map(image_path, amap, score,
                              save_path=vis_path, threshold=threshold)
        print(f"  Visualization → {vis_path}")

    return {"file": image_path, "score": score, "anomaly_map_shape": list(amap.shape)}


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore inference on O-ring images"
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to saved .pkl PatchCore model")
    parser.add_argument("--backbone", required=True,
                        choices=["resnet50", "resnet101"],
                        help="Backbone used in training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir", help="Directory of images to infer")
    group.add_argument("--image", help="Single image path")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output results/visualizations")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Anomaly score threshold")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate anomaly map visualizations")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1, low VRAM)")
    args = parser.parse_args()

    model = load_model(args.model_path, args.backbone)
    model.cfg.batch_size = args.batch_size

    output_dir = args.output_dir or str(Path(args.model_path).parent / "inference")

    if args.image_dir:
        infer_directory(model, args.image_dir, output_dir,
                        args.threshold, args.visualize)
    else:
        infer_single_image(model, args.image, args.threshold,
                           args.visualize, output_dir)


if __name__ == "__main__":
    main()
