"""
Export trained YOLO11-seg model to ONNX format.

Exports with opset 17, FP32, dynamic batch axis.
Input:  [1, 3, 480, 640]  float32  (RGB, 0-255)
Outputs: standard YOLO segmentation outputs (boxes, scores, masks, protos).

Usage:
    python export_onnx.py                               # auto-find best.pt
    python export_onnx.py --model path/to/best.pt       # explicit model
    python export_onnx.py --imgsz 640                   # custom size
"""

import argparse
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def find_best_model() -> str:
    """Auto-detect the best.pt from the latest training run."""
    runs_dir = SCRIPT_DIR / "runs"
    if not runs_dir.exists():
        return ""
    best_pts = sorted(
        runs_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(best_pts[0]) if best_pts else ""


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO11-seg to ONNX")
    p.add_argument("--model", type=str, default="",
                    help="Path to best.pt (auto-detected if omitted)")
    p.add_argument("--output", type=str, default="",
                    help="Output ONNX path (default: same dir as model)")
    p.add_argument("--imgsz", type=int, default=640,
                    help="Export image size")
    p.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version")
    p.add_argument("--simplify", action="store_true",
                    help="Simplify ONNX model with onnx-simplifier")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not Path(model_path).exists():
        print("ERROR: No model found. Train first or provide --model path.")
        return

    from ultralytics import YOLO

    print("=" * 70)
    print("YOLO11-seg ONNX EXPORT")
    print("=" * 70)
    print(f"  Source model : {model_path}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Opset        : {args.opset}")

    model = YOLO(model_path)

    t0 = time.perf_counter()

    # Ultralytics built-in export handles all YOLO-specific ops
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,       # fixed batch=1 for simpler C# integration
        half=False,          # FP32
    )

    elapsed = time.perf_counter() - t0
    export_file = Path(export_path)

    # Move to custom output path if specified
    if args.output:
        import shutil
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(export_file), str(out))
        export_file = out

    file_mb = export_file.stat().st_size / (1024 * 1024)

    print(f"\n  ONNX saved   : {export_file}")
    print(f"  File size    : {file_mb:.1f} MB")
    print(f"  Export time  : {elapsed:.1f}s")

    # Verify with ONNX Runtime
    print("\nVerifying with ONNX Runtime (CPU)...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(export_file),
            providers=["CPUExecutionProvider"],
        )

        # Print input/output info
        print("\n  Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        print("  Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")

        # Dummy inference to verify it runs
        inp_name = session.get_inputs()[0].name
        inp_shape = session.get_inputs()[0].shape
        # Replace dynamic dims with concrete values
        shape = [d if isinstance(d, int) else 1 for d in inp_shape]
        dummy = np.random.rand(*shape).astype(np.float32)

        t0 = time.perf_counter()
        _ = session.run(None, {inp_name: dummy})
        latency = (time.perf_counter() - t0) * 1000
        print(f"\n  Dummy inference OK — {latency:.0f} ms (CPU)")

    except ImportError:
        print("  onnxruntime not installed — skipping verification.")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
