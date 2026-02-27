"""
Export Mask R-CNN (torchvision) to ONNX — FP32, opset 17.

Torchvision's Mask R-CNN supports ONNX export natively since v0.12.
The exported model takes a [1, 3, H, W] float32 tensor (RGB, 0-1 scaled)
and returns:
    - boxes:  [N, 4]  float32  (x1, y1, x2, y2)
    - labels: [N]     int64
    - scores: [N]     float32
    - masks:  [N, 1, H, W] float32  (soft masks, threshold at 0.5)

N is dynamic (varies per image depending on detections).

Usage:
    cd "F:\\standard elastomers"
    conda activate dl
    python onnx_export/export_onnx.py

Author: GitHub Copilot
Date: February 25, 2026
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import onnx
import onnxruntime as ort

# ─── Paths ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
MASKRCNN_DIR = WORKSPACE / "maskrcnn"
CHECKPOINT = MASKRCNN_DIR / "dataset" / "combined" / "checkpoints" / "best_model.pth"
OUTPUT_ONNX = Path(__file__).resolve().parent / "maskrcnn_oring.onnx"

# ─── Add maskrcnn to path ───────────────────────────────────────────────
sys.path.insert(0, str(MASKRCNN_DIR))
from train import build_maskrcnn


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load the trained Mask R-CNN model in eval mode on CPU for export."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    num_classes = checkpoint.get("config", {}).get("num_classes", 2)
    model = build_maskrcnn(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Classes: {num_classes}, Epoch: {checkpoint.get('epoch', '?')}")
    return model, num_classes


def export_onnx(model, output_path: Path, opset: int = 17):
    """Export the model to ONNX format (FP32, dynamic batch & detections)."""
    # Dummy input: single 720×720 RGB image
    dummy = torch.randn(1, 3, 720, 720, dtype=torch.float32)

    print(f"\nExporting to ONNX (opset {opset}, FP32)...")
    t0 = time.perf_counter()

    torch.onnx.export(
        model,
        (dummy,),                       # model expects a list of tensors
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["boxes", "labels", "scores", "masks"],
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "boxes":  {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "masks":  {0: "num_detections", 2: "height", 3: "width"},
        },
    )

    elapsed = time.perf_counter() - t0
    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path}")
    print(f"  Size:  {file_mb:.1f} MB")
    print(f"  Time:  {elapsed:.1f}s")

    # Validate
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid.")

    # Print model inputs/outputs
    print(f"\n  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print(f"  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    return onnx_model


def verify_with_onnxruntime(onnx_path: Path, model_pytorch, num_iterations: int = 5):
    """Run the same image through PyTorch and ONNX Runtime; compare outputs."""
    print("\n" + "=" * 60)
    print("Verifying ONNX vs PyTorch (CPU, FP32)")
    print("=" * 60)

    # Create a test image (random or load a real one)
    test_images = list((WORKSPACE / "oring_crops" / "model1defect").glob("*.bmp"))[:1]
    if not test_images:
        test_images = list((WORKSPACE / "oring_crops" / "good").glob("*.bmp"))[:1]

    if test_images:
        img_bgr = cv2.imread(str(test_images[0]))
        # Resize to 720x720 for testing
        img_bgr = cv2.resize(img_bgr, (720, 720))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.astype(np.float32) / 255.0
        img_tensor_np = np.transpose(img_np, (2, 0, 1))[np.newaxis]  # [1, 3, 720, 720]
        print(f"  Test image: {test_images[0].name}")
    else:
        img_tensor_np = np.random.rand(1, 3, 720, 720).astype(np.float32)
        print("  Test image: random noise")

    # PyTorch inference
    img_tensor_pt = torch.as_tensor(img_tensor_np)
    with torch.no_grad():
        pt_out = model_pytorch(img_tensor_pt)[0]
    pt_boxes = pt_out["boxes"].numpy()
    pt_scores = pt_out["scores"].numpy()
    pt_labels = pt_out["labels"].numpy()
    print(f"\n  PyTorch:  {len(pt_scores)} detections")
    if len(pt_scores) > 0:
        print(f"    Top score: {pt_scores[0]:.4f}")
        print(f"    Top box:   {pt_boxes[0]}")

    # ONNX Runtime inference
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_out = session.run(None, {"image": img_tensor_np})
    ort_boxes, ort_labels, ort_scores, ort_masks = ort_out
    print(f"  ONNX RT:  {len(ort_scores)} detections")
    if len(ort_scores) > 0:
        print(f"    Top score: {ort_scores[0]:.4f}")
        print(f"    Top box:   {ort_boxes[0]}")

    # Compare
    if len(pt_scores) > 0 and len(ort_scores) > 0:
        n = min(len(pt_scores), len(ort_scores))
        score_diff = np.abs(pt_scores[:n] - ort_scores[:n]).max()
        box_diff = np.abs(pt_boxes[:n] - ort_boxes[:n]).max()
        print(f"\n  Max score difference: {score_diff:.6f}")
        print(f"  Max box difference:   {box_diff:.4f} px")
        if score_diff < 0.001 and box_diff < 1.0:
            print("  ✓ Outputs match (within tolerance).")
        else:
            print("  ⚠ Outputs differ — check carefully.")
    elif len(pt_scores) == 0 and len(ort_scores) == 0:
        print("\n  ✓ Both produce zero detections — consistent.")
    else:
        print(f"\n  ⚠ Detection count mismatch: PT={len(pt_scores)} vs ORT={len(ort_scores)}")

    # Benchmark ONNX Runtime
    print(f"\n  Benchmarking ONNX Runtime ({num_iterations} runs)...")
    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        session.run(None, {"image": img_tensor_np})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print(f"    CPU inference: mean={arr.mean():.0f}ms  std={arr.std():.0f}ms  "
          f"min={arr.min():.0f}ms  max={arr.max():.0f}ms")


def main():
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    model, num_classes = load_model(CHECKPOINT, device="cpu")
    export_onnx(model, OUTPUT_ONNX)
    verify_with_onnxruntime(OUTPUT_ONNX, model)

    print("\n" + "=" * 60)
    print(f"ONNX model ready: {OUTPUT_ONNX}")
    print("=" * 60)


if __name__ == "__main__":
    main()
