"""
Export PatchCore to ONNX — Self-contained model (backbone + bank + k-NN).

Creates a single ONNX model containing:
  1. ResNet-50/101 feature extractor (layers 3 & 4 → 3072-d)
  2. Memory bank embedded as constant tensor
  3. k-NN anomaly scoring (k=9, decomposed squared distances)
  4. Gaussian smoothing + bilinear upsampling to input resolution

Input:   image  [1, 3, 640, 640]  float32, RGB, [0-1] scaled
Output:  anomaly_score  [1]              float32
         anomaly_map    [1, 1, 640, 640] float32

Usage:
    cd "F:\\standard elastomers"
    conda activate dl
    python onnx_export/export_patchcore_onnx.py --model model1
    python onnx_export/export_patchcore_onnx.py --model model2
    python onnx_export/export_patchcore_onnx.py --all
    python onnx_export/export_patchcore_onnx.py --results-dir patchcore/results_cropped --model model1_cropped
    python onnx_export/export_patchcore_onnx.py --results-dir patchcore/results_cropped --model model2_cropped

Author: GitHub Copilot
Date:   February 28, 2026
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─── Ensure cuDNN DLLs are on PATH for onnxruntime-gpu ─────────────────
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

# ─── Paths ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE / "patchcore" / "results"  # default; overridden by --results-dir
OUTPUT_DIR = Path(__file__).resolve().parent

# Allow importing from workspace root
sys.path.insert(0, str(WORKSPACE))


# ─── Gaussian kernel helper ─────────────────────────────────────────────

def _make_gaussian_kernel(sigma: float = 4.0) -> torch.Tensor:
    """Create a 2D Gaussian kernel for Conv2d smoothing."""
    ksize = 2 * int(4 * sigma + 0.5) + 1
    coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # outer product
    kernel = kernel / kernel.sum()
    return kernel


# ─── ONNX-exportable PatchCore Module ───────────────────────────────────

class PatchCoreONNX(nn.Module):
    """Full PatchCore as a single ONNX-exportable module.

    Includes ImageNet normalization, ResNet backbone, feature aggregation,
    k-NN scoring against embedded memory bank, and Gaussian smoothing.

    Parameters
    ----------
    backbone_name : "resnet50" or "resnet101"
    memory_bank   : (M, 3072) float32 numpy array
    spatial_shape  : (H, W) feature map spatial dims (e.g. (40, 40))
    n_neighbors   : k for k-NN scoring (default 9)
    sigma         : Gaussian smoothing sigma (default 4.0)
    input_size    : spatial size of the input image (default 640)
    """

    def __init__(self, backbone_name: str, memory_bank: np.ndarray,
                 spatial_shape: tuple, n_neighbors: int = 9,
                 sigma: float = 4.0, input_size: int = 640):
        super().__init__()

        # ── Backbone ──
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3   # feature_layers index 2 → 1024-d
        self.layer4 = backbone.layer4   # feature_layers index 3 → 2048-d

        # ── ImageNet normalization ──
        self.register_buffer(
            'img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ── Memory bank ──
        bank_t = torch.from_numpy(memory_bank).float()   # (M, D)
        self.register_buffer('bank', bank_t)
        self.register_buffer('bank_sq_norms', (bank_t ** 2).sum(dim=1))  # (M,)

        self.n_neighbors = n_neighbors
        self.spatial_h = spatial_shape[0]
        self.spatial_w = spatial_shape[1]
        self.input_size = input_size

        # ── Gaussian smoothing kernel ──
        kernel = _make_gaussian_kernel(sigma)
        ksize = kernel.shape[0]
        self.gauss_pad = ksize // 2
        # Store as (1, 1, K, K) Conv2d weight
        self.register_buffer('gauss_kernel', kernel.unsqueeze(0).unsqueeze(0))

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (1, 3, 640, 640) float32, RGB, [0, 1]

        Returns
        -------
        anomaly_score : (1,) float32
        anomaly_map   : (1, 1, 640, 640) float32
        """
        # ── ImageNet normalization ──
        x = (x - self.img_mean) / self.img_std

        # ── Feature extraction ──
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_l3 = self.layer3(x)        # (1, 1024, H, W)   H=40 for 640 input
        feat_l4 = self.layer4(feat_l3)  # (1, 2048, H/2, W/2)

        # Upsample layer4 to match layer3 spatial size
        feat_l4_up = F.interpolate(
            feat_l4,
            size=(self.spatial_h, self.spatial_w),
            mode='bilinear', align_corners=False
        )

        # Concatenate → (1, 3072, H, W)
        features = torch.cat([feat_l3, feat_l4_up], dim=1)

        # Reshape to patch embeddings: (H*W, D)
        C = features.shape[1]
        P = self.spatial_h * self.spatial_w
        patches = features.permute(0, 2, 3, 1).reshape(P, C)

        # ── k-NN scoring ──
        # dist² = ||q||² + ||b||² − 2·q·bᵀ
        q_sq = (patches ** 2).sum(dim=1, keepdim=True)    # (P, 1)
        dot = torch.mm(patches, self.bank.t())              # (P, M)
        dists_sq = q_sq + self.bank_sq_norms.unsqueeze(0) - 2.0 * dot
        dists_sq = dists_sq.clamp(min=0.0)

        # Top-k smallest distances → use negation trick for ONNX compatibility
        neg_dists = -dists_sq
        topk_neg, _ = torch.topk(neg_dists, k=self.n_neighbors, dim=1)
        topk_sq = -topk_neg                                # (P, k) positive

        # Mean k-NN distance (take sqrt for interpretable scores)
        patch_scores = topk_sq.sqrt().mean(dim=1)          # (P,)

        # ── Spatial anomaly map ──
        score_map = patch_scores.reshape(1, 1, self.spatial_h, self.spatial_w)

        # Gaussian smoothing
        score_map = F.pad(score_map, [self.gauss_pad] * 4, mode='constant', value=0)
        score_map = F.conv2d(score_map, self.gauss_kernel)

        # Upsample to input resolution for visualization
        score_map = F.interpolate(
            score_map,
            size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False
        )

        # Image-level score = max of anomaly map
        anomaly_score = score_map.reshape(-1).max().unsqueeze(0)

        return anomaly_score, score_map


# ─── Export functions ────────────────────────────────────────────────────

def load_patchcore_state(model_name: str, backbone: str) -> dict:
    """Load the trained PatchCore .pkl state."""
    pkl_name = f"{model_name}_{backbone}_patchcore.pkl"
    pkl_path = RESULTS_DIR / f"{model_name}_{backbone}" / pkl_name
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trained model not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    print(f"  Loaded: {pkl_path}")
    print(f"    Backbone:       {state['backbone']}")
    print(f"    Memory bank:    {state['memory_bank'].shape}")
    print(f"    Spatial shape:  {state['spatial_shape']}")
    print(f"    Feature dim:    {state['feat_dim']}")
    print(f"    n_neighbors:    {state['n_neighbors']}")
    print(f"    Resize:         {state.get('resize', 660)}")
    print(f"    Center crop:    {state.get('center_crop', 640)}")
    return state


def build_onnx_module(state: dict) -> PatchCoreONNX:
    """Build the ONNX-exportable module from a saved state."""
    model = PatchCoreONNX(
        backbone_name=state["backbone"],
        memory_bank=state["memory_bank"],
        spatial_shape=state["spatial_shape"],
        n_neighbors=state["n_neighbors"],
        input_size=state.get("center_crop", 640),
    )
    model.eval()
    return model


def export_to_onnx(model: PatchCoreONNX, output_path: Path, opset: int = 17):
    """Export the PatchCoreONNX module to ONNX."""
    input_size = model.input_size
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"\n  Exporting to ONNX (opset {opset}, FP32)...")
    t0 = time.perf_counter()

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["anomaly_score", "anomaly_map"],
        dynamic_axes={
            "image": {0: "batch"},
            "anomaly_map": {0: "batch"},
        },
    )

    elapsed = time.perf_counter() - t0
    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved:  {output_path}")
    print(f"  Size:   {file_mb:.1f} MB")
    print(f"  Export: {elapsed:.1f}s")

    # Validate
    import onnx
    print("\n  Validating ONNX graph...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid.")

    print(f"  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print(f"  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    return onnx_model


def verify_onnx(onnx_path: Path, pt_model: PatchCoreONNX, state: dict):
    """Compare PyTorch vs ONNX Runtime outputs and benchmark."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("  Verifying ONNX vs PyTorch")
    print("=" * 60)

    input_size = state.get("center_crop", 640)

    # Try to load a real test image
    test_dirs = [
        WORKSPACE / "binned" / "model1defect",
        WORKSPACE / "binned" / "notok",
        WORKSPACE / "binned" / "model1good",
        WORKSPACE / "binned" / "good",
    ]
    test_image = None
    for d in test_dirs:
        imgs = list(d.glob("*.bmp"))[:1]
        if imgs:
            test_image = imgs[0]
            break

    if test_image:
        img_bgr = cv2.imread(str(test_image))
        # Apply same preprocessing: resize → center crop → scale to [0,1]
        resize_size = state.get("resize", 660)
        img = cv2.resize(img_bgr, (resize_size, resize_size),
                         interpolation=cv2.INTER_CUBIC)
        # Center crop
        margin = (resize_size - input_size) // 2
        img = img[margin:margin + input_size, margin:margin + input_size]
        # BGR → RGB, scale to [0,1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_np = np.transpose(img_rgb, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]
        print(f"  Test image: {test_image.name}")
    else:
        img_np = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        print("  Test image: random noise")

    # ── PyTorch inference ──
    img_tensor = torch.from_numpy(img_np)
    with torch.no_grad():
        pt_score, pt_map = pt_model(img_tensor)
    pt_score_val = pt_score.item()
    print(f"\n  PyTorch score:  {pt_score_val:.4f}")
    print(f"  PyTorch map:    shape={tuple(pt_map.shape)}, "
          f"min={pt_map.min().item():.4f}, max={pt_map.max().item():.4f}")

    # ── ONNX Runtime inference (GPU) ──
    gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=gpu_providers)
    active = session.get_providers()
    if "CUDAExecutionProvider" in active:
        print("  ONNX Runtime provider: CUDA GPU")
    else:
        print("  ⚠ CUDA not available, falling back to CPU")

    ort_out = session.run(None, {"image": img_np})
    ort_score = ort_out[0][0]
    ort_map = ort_out[1]
    print(f"\n  ONNX RT score:  {ort_score:.4f}")
    print(f"  ONNX RT map:    shape={ort_map.shape}, "
          f"min={ort_map.min():.4f}, max={ort_map.max():.4f}")

    # ── Compare ──
    score_diff = abs(pt_score_val - ort_score)
    map_diff = np.abs(pt_map.numpy() - ort_map).max()
    print(f"\n  Score difference: {score_diff:.6f}")
    print(f"  Map max diff:     {map_diff:.6f}")
    if score_diff < 0.05 and map_diff < 0.1:
        print("  ✓ Outputs match (within tolerance).")
    else:
        print("  ⚠ Outputs differ — review carefully.")

    # ── Benchmark (GPU only) ──
    n_iter = 5
    provider_name = "GPU" if "CUDAExecutionProvider" in active else "CPU"
    # Warmup run
    session.run(None, {"image": img_np})
    print(f"\n  Benchmarking ONNX Runtime {provider_name} ({n_iter} runs)...")
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        session.run(None, {"image": img_np})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print(f"    {provider_name}: mean={arr.mean():.0f}ms  min={arr.min():.0f}ms  max={arr.max():.0f}ms")

    # Explicitly release ORT session to free GPU memory
    del session


def save_metadata(model_name: str, backbone: str, state: dict,
                  onnx_path: Path, results_json_path: Path):
    """Save a metadata JSON alongside the ONNX model."""
    # Load training results for thresholds
    good_max = None
    defect_min = None
    if results_json_path.exists():
        with open(results_json_path) as f:
            results = json.load(f)
        # Find good_max (max score across good samples)
        if "good" in results:
            good_max = results["good"].get("score_max")
        # Find defect_min (min score across all defect categories)
        defect_mins = []
        for key, val in results.items():
            if key.startswith(("defect", "notok")) and isinstance(val, dict):
                if "score_min" in val:
                    defect_mins.append(val["score_min"])
        if defect_mins:
            defect_min = min(defect_mins)

    threshold = None
    if good_max is not None and defect_min is not None:
        threshold = (good_max + defect_min) / 2

    meta = {
        "model_name": f"{model_name}_{backbone}",
        "backbone": backbone,
        "input_shape": [1, 3, state.get("center_crop", 640),
                        state.get("center_crop", 640)],
        "input_format": "RGB float32 [0, 1]",
        "resize": state.get("resize", 660),
        "center_crop": state.get("center_crop", 640),
        "original_image_size": 720,
        "spatial_shape": list(state["spatial_shape"]),
        "n_neighbors": state["n_neighbors"],
        "memory_bank_size": state["memory_bank"].shape[0],
        "feature_dim": state["feat_dim"],
        "coreset_ratio": state.get("coreset_ratio", 0.25),
        "good_score_max": good_max,
        "defect_score_min": defect_min,
        "recommended_threshold": round(threshold, 2) if threshold else None,
        "onnx_file": onnx_path.name,
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 1),
        "export_date": datetime.now().strftime("%Y-%m-%d"),
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "note": "ImageNet normalization is embedded in the ONNX model. "
                "Input only needs to be scaled to [0, 1]."
    }

    meta_path = onnx_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata → {meta_path}")
    return meta


def export_model(model_name: str, backbone: str = "resnet50", opset: int = 17):
    """Full export pipeline for one model."""
    print("\n" + "=" * 70)
    print(f"  Exporting PatchCore: {model_name}_{backbone}")
    print("=" * 70)

    # 1. Load trained state
    state = load_patchcore_state(model_name, backbone)

    # 2. Build ONNX module
    pt_model = build_onnx_module(state)
    bank_mb = state["memory_bank"].nbytes / (1024 * 1024)
    print(f"\n  Memory bank in ONNX: {bank_mb:.0f} MB (fp32)")

    # 3. Export to ONNX
    onnx_name = f"patchcore_{model_name}_{backbone}.onnx"
    onnx_path = OUTPUT_DIR / onnx_name
    export_to_onnx(pt_model, onnx_path, opset=opset)

    # 4. Verify
    verify_onnx(onnx_path, pt_model, state)

    # 5. Save metadata
    results_json = (RESULTS_DIR / f"{model_name}_{backbone}" /
                    f"{model_name}_{backbone}_results.json")
    meta = save_metadata(model_name, backbone, state, onnx_path, results_json)

    print(f"\n  ✓ Export complete: {onnx_path}")
    print(f"    Threshold: score > {meta.get('recommended_threshold', '?')} → DEFECT")
    return onnx_path


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Export trained PatchCore model(s) to ONNX")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name prefix, e.g. model1, model1_cropped (default: all available)")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--all", action="store_true",
                        help="Export all available models")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (default: patchcore/results)")
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir).resolve()

    print("PatchCore → ONNX Export")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 70)

    if args.all or args.model is None:
        # Find all available models
        exported = []
        for model_dir in sorted(RESULTS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            parts = model_dir.name.rsplit("_", 1)
            if len(parts) == 2:
                mname = parts[0]
                bbone = parts[1]
                pkl = model_dir / f"{mname}_{bbone}_patchcore.pkl"
                if pkl.exists():
                    try:
                        path = export_model(mname, bbone, args.opset)
                        exported.append(path)
                    except Exception as e:
                        print(f"  ✗ Failed: {mname}_{bbone}: {e}")
        if not exported:
            print("No trained models found in", RESULTS_DIR)
            sys.exit(1)
        print(f"\n{'='*70}")
        print(f"  Exported {len(exported)} model(s):")
        for p in exported:
            print(f"    {p}")
    else:
        export_model(args.model, args.backbone, args.opset)

    print(f"\n{'='*70}")
    print("  Done.")

    # Force exit — onnxruntime-gpu CUDA cleanup can deadlock on Windows
    os._exit(0)


if __name__ == "__main__":
    main()
