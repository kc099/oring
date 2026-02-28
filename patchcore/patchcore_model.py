"""
PatchCore Model
===============
Full PatchCore implementation:
  1. Feature extraction from pre-trained ResNet (layers 2 & 3)
  2. Local-aware patch feature aggregation
  3. Greedy coreset subsampling
  4. k-NN anomaly scoring at inference

Reference:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
    CVPR 2022.  https://arxiv.org/abs/2106.08265

Author: GitHub Copilot
Date:   February 27, 2026
"""

import time
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from .config import ModelConfig


# ─── Feature Extractor ───────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pre-trained ResNet.

    Hooks into the specified ``layer_indices`` (0-based ResNet blocks:
    layer1=0, layer2=1, layer3=2, layer4=3) and returns their outputs.
    """

    def __init__(self, backbone_name: str = "resnet50",
                 layer_indices: Tuple[int, ...] = (2, 3)):
        super().__init__()
        self.layer_indices = layer_indices
        self.features: dict = {}

        # Load pre-trained backbone
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Register hooks on target layers
        self.layers = nn.ModuleList()
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        for idx in layer_indices:
            layer = getattr(backbone, layer_names[idx])
            self.layers.append(layer)
            layer.register_forward_hook(self._make_hook(idx))

        # Build sequential up to the deepest required layer
        max_layer = max(layer_indices)
        modules = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        ]
        for i in range(max_layer + 1):
            modules.append(getattr(backbone, layer_names[i]))
        self.backbone = nn.Sequential(*modules)

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

    def _make_hook(self, idx: int):
        def hook_fn(module, input, output):
            self.features[idx] = output
        return hook_fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass — returns list of feature maps from hooked layers."""
        self.features.clear()
        _ = self.backbone(x)
        return [self.features[idx] for idx in self.layer_indices]


# ─── Patch Feature Aggregation ───────────────────────────────────────────

def aggregate_features(feature_maps: List[torch.Tensor],
                       target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Concatenate multi-scale feature maps into patch-level descriptors.

    All feature maps are upsampled (bilinear) to the spatial resolution of
    the largest map, then concatenated channel-wise and reshaped to
    ``(N_patches, C_total)`` where ``N_patches = B * H * W``.

    Parameters
    ----------
    feature_maps : list of (B, C_i, H_i, W_i) tensors
    target_size : (H, W) to align to; defaults to the largest spatial size.

    Returns
    -------
    (N_patches, C_total) tensor on CPU
    """
    if target_size is None:
        target_size = feature_maps[0].shape[2:]
        for fm in feature_maps[1:]:
            if fm.shape[2] > target_size[0]:
                target_size = fm.shape[2:]

    aligned = []
    for fm in feature_maps:
        if fm.shape[2:] != target_size:
            fm = F.interpolate(fm, size=target_size,
                               mode="bilinear", align_corners=False)
        aligned.append(fm)

    # (B, C_total, H, W) → (B*H*W, C_total)
    concat = torch.cat(aligned, dim=1)                   # (B, C, H, W)
    B, C, H, W = concat.shape
    concat = concat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    return concat


# ─── Greedy Coreset Subsampling ──────────────────────────────────────────

def greedy_coreset(embedding: np.ndarray,
                   sampling_ratio: float = 0.25,
                   seed: int = 0,
                   device: str = "auto") -> np.ndarray:
    """GPU-accelerated greedy coreset selection (approximate k-Center).

    Iteratively selects the point that is farthest from the current
    coreset.  Uses PyTorch on GPU for fast distance computation.

    Parameters
    ----------
    embedding : (N, D) array
    sampling_ratio : fraction of N to keep
    seed : random seed for initial point
    device : "auto" (use GPU if available), "cuda", or "cpu"

    Returns
    -------
    (M, D) coreset array   where M ≈ sampling_ratio * N
    """
    N, D = embedding.shape
    M = max(1, int(N * sampling_ratio))

    if M >= N:
        return embedding.copy()

    # Pick device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"    Coreset: {N} → {M} patches ({sampling_ratio:.0%})  [device={dev}]")

    # Move embedding to GPU as float32
    emb = torch.from_numpy(embedding).to(dev, dtype=torch.float32)  # (N, D)
    # Pre-compute squared norms for fast distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    sq_norms = (emb ** 2).sum(dim=1)  # (N,)

    rng = np.random.RandomState(seed)
    selected_indices = [rng.randint(N)]
    min_distances = torch.full((N,), float("inf"), device=dev, dtype=torch.float32)

    for i in range(1, M):
        # Squared distance from last selected point to all points
        last = emb[selected_indices[-1]]  # (D,)
        # ||emb - last||^2 = sq_norms + ||last||^2 - 2 * emb @ last
        dists_sq = sq_norms + (last ** 2).sum() - 2.0 * (emb @ last)
        dists_sq.clamp_(min=0.0)  # numerical safety

        # Update running minimum distances (squared is monotonic, skip sqrt)
        min_distances = torch.minimum(min_distances, dists_sq)

        # Select the farthest point
        next_idx = int(min_distances.argmax().item())
        selected_indices.append(next_idx)

        if (i + 1) % 1000 == 0 or i == M - 1:
            print(f"      [{i+1}/{M}] selected")

    return embedding[selected_indices]


# ─── PatchCore ────────────────────────────────────────────────────────────

class PatchCore:
    """PatchCore anomaly detection model.

    Attributes
    ----------
    cfg : ModelConfig
    extractor : FeatureExtractor
    memory_bank : np.ndarray or None   — (M, D) coreset embedding
    image_level_score_type : str
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(
            backbone_name=cfg.backbone,
            layer_indices=cfg.feature_layers,
        ).to(self.device).eval()
        self.memory_bank: Optional[np.ndarray] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None  # (H, W) of feature map
        self.feat_dim: Optional[int] = None
        # GPU-cached bank for fast inference (set by _prepare_bank)
        self._bank_gpu: Optional[torch.Tensor] = None        # (M, D) fp16
        self._bank_sq_norms: Optional[torch.Tensor] = None   # (M,)  fp16
        self._bank_cached_size: int = 0                       # track size for invalidation

    # ── Training (feature extraction + coreset) ──────────────────────────

    @torch.no_grad()
    def fit(self, dataloader: DataLoader) -> None:
        """Build the memory bank from normal/good images."""
        print(f"\n{'='*60}")
        print(f"  Fitting PatchCore  [{self.cfg.name}]")
        print(f"  Backbone: {self.cfg.backbone}  |  Coreset: {self.cfg.coreset_ratio:.0%}")
        print(f"{'='*60}")
        t0 = time.time()

        all_patches = []
        for batch_idx, (images, _, _) in enumerate(dataloader):
            images = images.to(self.device)
            feat_maps = self.extractor(images)
            patches = aggregate_features(feat_maps)  # (B*H*W, C)
            all_patches.append(patches.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                print(f"    Extracted batch {batch_idx+1}/{len(dataloader)}")

        all_patches = np.concatenate(all_patches, axis=0)  # (N, C)
        self.feat_dim = all_patches.shape[1]
        print(f"    Total patches: {all_patches.shape[0]:,}  dim={self.feat_dim}")

        # Infer spatial shape from a single forward pass
        sample = next(iter(dataloader))[0][:1].to(self.device)
        fmaps = self.extractor(sample)
        target_h = max(fm.shape[2] for fm in fmaps)
        target_w = max(fm.shape[3] for fm in fmaps)
        self.spatial_shape = (target_h, target_w)
        print(f"    Feature map spatial shape: {self.spatial_shape}")

        # Greedy coreset subsampling
        self.memory_bank = greedy_coreset(
            all_patches,
            sampling_ratio=self.cfg.coreset_ratio,
        )
        elapsed = time.time() - t0
        print(f"    Memory bank: {self.memory_bank.shape}  ({elapsed:.1f}s)")

    # ── GPU bank caching ─────────────────────────────────────────────

    def _prepare_bank(self) -> None:
        """Cache memory bank on GPU as fp16 with pre-computed squared norms.

        Uses fp16 to halve VRAM (e.g. 80k×3072: 940MB fp32 → 470MB fp16).
        Pre-computes ||b||² so distance = ||q||² + ||b||² − 2·q·bᵀ
        only needs one matmul per chunk, avoiding torch.cdist overhead.
        """
        current_size = self.memory_bank.shape[0]
        if self._bank_gpu is not None and self._bank_cached_size == current_size:
            return  # already cached and same size
        # Invalidate stale cache (bank changed)
        if self._bank_gpu is not None:
            del self._bank_gpu, self._bank_sq_norms
            torch.cuda.empty_cache()
        bank_f16 = torch.from_numpy(self.memory_bank).to(
            self.device, dtype=torch.float16
        )  # (M, D)
        self._bank_gpu = bank_f16
        self._bank_sq_norms = (bank_f16 ** 2).sum(dim=1)  # (M,)
        self._bank_cached_size = current_size
        mb = bank_f16.nelement() * 2 / 1e6
        print(f"    Bank cached on GPU: {bank_f16.shape}  fp16  ({mb:.0f} MB)")

    def release_bank(self) -> None:
        """Free GPU bank cache (e.g. when switching models)."""
        self._bank_gpu = None
        self._bank_sq_norms = None
        self._bank_cached_size = 0
        torch.cuda.empty_cache()

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """Score every image in the dataloader.

        Returns
        -------
        image_scores : (N_images,) — max patch distance per image
        labels       : (N_images,) — ground truth labels
        paths        : list of file paths
        anomaly_maps : (N_images, H_feat, W_feat) — per-patch distance maps
        """
        assert self.memory_bank is not None, "Call fit() first"
        self._prepare_bank()

        all_scores = []
        all_labels = []
        all_paths = []
        all_maps = []

        bank = self._bank_gpu              # (M, D) fp16
        bank_sq = self._bank_sq_norms      # (M,)   fp16
        k = self.cfg.n_neighbors

        for images, labels, paths in dataloader:
            images = images.to(self.device)
            feat_maps = self.extractor(images)
            patches = aggregate_features(feat_maps)  # (B*H*W, D) fp32

            B = images.shape[0]
            H, W = self.spatial_shape

            # Convert query patches to fp16
            q = patches.to(self.device, dtype=torch.float16)  # (P, D)
            q_sq = (q ** 2).sum(dim=1, keepdim=True)          # (P, 1)

            # Compute distances in chunks:
            # ||q-b||² = ||q||² + ||b||² - 2·q·bᵀ
            chunk_size = 2048
            min_dists = []
            for start in range(0, q.shape[0], chunk_size):
                end = min(start + chunk_size, q.shape[0])
                q_chunk = q[start:end]               # (C, D)
                q_sq_chunk = q_sq[start:end]          # (C, 1)
                # (C, M) = (C, 1) + (1, M) - 2*(C, D)@(D, M)
                dist_sq = q_sq_chunk + bank_sq.unsqueeze(0) - 2.0 * (q_chunk @ bank.t())
                dist_sq.clamp_(min=0.0)
                topk_sq, _ = dist_sq.topk(k, dim=1, largest=False)
                # Use sqrt for final distance (more interpretable scores)
                min_dists.append(topk_sq.sqrt().mean(dim=1))

            min_dists = torch.cat(min_dists, dim=0).float().cpu().numpy()  # (B*H*W,)

            # Reshape to (B, H, W)
            score_maps = min_dists.reshape(B, H, W)

            # Apply Gaussian smoothing for cleaner anomaly maps
            for i in range(B):
                score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

            # Image-level score = max of the anomaly map
            image_scores = score_maps.reshape(B, -1).max(axis=1)

            all_scores.append(image_scores)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
            all_maps.append(score_maps)

        return (
            np.concatenate(all_scores),
            np.concatenate(all_labels),
            all_paths,
            np.concatenate(all_maps),
        )

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, dataloader: DataLoader, label_name: str = "") -> dict:
        """Run prediction and compute metrics."""
        scores, labels, paths, maps = self.predict(dataloader)
        results = {
            "label": label_name,
            "n_samples": len(scores),
            "scores": scores,
            "labels": labels,
            "paths": paths,
            "anomaly_maps": maps,
        }
        # Image-level AUROC (only meaningful if we have both classes)
        unique = np.unique(labels)
        if len(unique) > 1:
            auroc = roc_auc_score(labels, scores)
            results["auroc"] = auroc
            print(f"    [{label_name}] AUROC = {auroc:.4f}")
        else:
            kind = "good" if unique[0] == 0 else "defect"
            results["auroc"] = None
            print(f"    [{label_name}] {len(scores)} {kind} samples  "
                  f"score: mean={scores.mean():.4f}  std={scores.std():.4f}  "
                  f"min={scores.min():.4f}  max={scores.max():.4f}")
        return results

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist memory bank and metadata."""
        if path is None:
            path = self.cfg.output_dir / f"{self.cfg.name}_patchcore.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "memory_bank": self.memory_bank,
            "spatial_shape": self.spatial_shape,
            "feat_dim": self.feat_dim,
            "backbone": self.cfg.backbone,
            "feature_layers": self.cfg.feature_layers,
            "coreset_ratio": self.cfg.coreset_ratio,
            "n_neighbors": self.cfg.n_neighbors,
            "resize": self.cfg.resize,
            "center_crop": self.cfg.center_crop,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"    Model saved → {path}")
        return path

    def load(self, path: Path) -> None:
        """Load memory bank and metadata."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.memory_bank = state["memory_bank"]
        self.spatial_shape = state["spatial_shape"]
        self.feat_dim = state["feat_dim"]
        # Clear GPU cache so _prepare_bank re-uploads
        self._bank_gpu = None
        self._bank_sq_norms = None
        print(f"    Loaded memory bank: {self.memory_bank.shape} from {path}")
