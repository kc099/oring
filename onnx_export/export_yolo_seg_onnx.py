"""
Export YOLO11-seg O-ring segmentation model to ONNX.

Finds the latest best.pt from the YOLO training runs and exports it
to this folder (onnx_export/) alongside the PatchCore ONNX models.

Input:  [1, 3, 640, 640]  float32, RGB, 0-255
Output: standard YOLO segmentation outputs (boxes, scores, masks, protos)

Usage:
    cd "F:\\standard elastomers"
    conda activate dl
    python onnx_export/export_yolo_seg_onnx.py
    python onnx_export/export_yolo_seg_onnx.py --model path/to/best.pt
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
YOLO_SEG_DIR = WORKSPACE / "patchcore" / "yolo_seg"


def find_best_model() -> str:
    """Auto-detect best.pt from the latest YOLO training run."""
    runs_dir = YOLO_SEG_DIR / "runs"
    if not runs_dir.exists():
        return ""
    best_pts = sorted(
        runs_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(best_pts[0]) if best_pts else ""


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11-seg to ONNX")
    parser.add_argument("--model", type=str, default="",
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Export image size")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not Path(model_path).exists():
        print("ERROR: No YOLO model found. Train first or provide --model.")
        return

    from ultralytics import YOLO

    print("=" * 70)
    print("YOLO11-seg ONNX EXPORT")
    print("=" * 70)
    print(f"  Source model : {model_path}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Opset        : {args.opset}")
    print(f"  Output dir   : {SCRIPT_DIR}")

    model = YOLO(model_path)

    t0 = time.perf_counter()
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=False,
        dynamic=False,
        half=False,
    )
    elapsed = time.perf_counter() - t0

    # Move to onnx_export/ folder
    src = Path(export_path)
    dst = SCRIPT_DIR / "yolo11_seg_oring.onnx"
    shutil.move(str(src), str(dst))
    print(f"\n  Moved: {src.name} → {dst}")

    file_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  File size    : {file_mb:.1f} MB")
    print(f"  Export time  : {elapsed:.1f}s")

    # ── Verify with ONNX Runtime ──
    print("\n" + "-" * 50)
    print("Verifying with ONNX Runtime...")
    try:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(dst), providers=providers)
        active = session.get_providers()
        provider = "GPU" if "CUDAExecutionProvider" in active else "CPU"

        print(f"\n  Provider: {provider}")
        print("  Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        print("  Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")

        # Dummy inference
        inp_name = session.get_inputs()[0].name
        inp_shape = session.get_inputs()[0].shape
        shape = [d if isinstance(d, int) else 1 for d in inp_shape]
        dummy = np.random.rand(*shape).astype(np.float32)

        # Warmup
        session.run(None, {inp_name: dummy})

        # Benchmark
        n_iter = 10
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            session.run(None, {inp_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"\n  Benchmark ({provider}, {n_iter} runs):")
        print(f"    mean={arr.mean():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")

        # Also test with a real image if available
        import cv2
        test_dirs = [
            WORKSPACE / "patchcore" / "data" / "patchcore-model1",
            WORKSPACE / "patchcore" / "data" / "patchcore-model2",
        ]
        for d in test_dirs:
            imgs = sorted(d.glob("*.bmp"))[:1]
            if imgs:
                img = cv2.imread(str(imgs[0]))
                img_resized = cv2.resize(img, (640, 480))
                # Pad to 640x640 (YOLO letterbox)
                padded = np.zeros((640, 640, 3), dtype=np.uint8)
                padded[:480, :640] = img_resized
                # HWC→CHW, scale to 0-1
                blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
                out = session.run(None, {inp_name: blob})
                print(f"\n  Real image test ({imgs[0].name}):")
                print(f"    Output shapes: {[o.shape for o in out]}")
                break

        del session

    except ImportError:
        print("  onnxruntime not installed — skipping verification.")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print(f"  ONNX file: {dst}")
    print("=" * 70)


if __name__ == "__main__":
    main()
