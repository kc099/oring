"""
PatchCore Anomaly Detection GUI — O-Ring Inspection  (Dear PyGui)

Upload an image (or pick from file dialog), select Model 1/2 and
ResNet-50/101 backbone, run PatchCore anomaly scoring, and display
the original image with an anomaly-map overlay side by side.

Auto-discovers trained .pkl models under  patchcore/results/.

Usage:
    python -m patchcore.gui

Author: GitHub Copilot
Date:   February 27, 2026
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import torch
import dearpygui.dearpygui as dpg

# ─── Resolve workspace ─────────────────────────────────────────────────── 
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE  = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"

# ─── Lazy imports (avoid circular) ───────────────────────────────────────
_patchcore_model = None
_dataset_module = None
_config_module = None


def _lazy_imports():
    global _patchcore_model, _dataset_module, _config_module
    if _patchcore_model is None:
        from patchcore import patchcore_model as pm, dataset as ds, config as cfg
        _patchcore_model = pm
        _dataset_module = ds
        _config_module = cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_W, DISPLAY_H = 480, 480
WINDOW_W, WINDOW_H = 1080, 700

# Colors
COL_OK     = (0, 200, 0)
COL_FAIL   = (255, 60, 60)
COL_WARN   = (255, 200, 0)
COL_GRAY   = (180, 180, 180)
COL_BG     = (30, 30, 30)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_models() -> Dict[str, Path]:
    """Scan patchcore/results/ for saved .pkl models."""
    models = {}
    if RESULTS_DIR.exists():
        for pkl in RESULTS_DIR.rglob("*.pkl"):
            # Use parent folder + stem as display name
            key = pkl.stem.replace("_patchcore", "")
            models[key] = pkl
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Application State
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    """Mutable application state shared across callbacks."""

    def __init__(self):
        self.loaded_model: Optional[object] = None
        self.loaded_model_key: str = ""
        self.loaded_image_path: Optional[str] = None
        self.loaded_image_bgr: Optional[np.ndarray] = None
        self.available_models: Dict[str, Path] = {}
        self.score: Optional[float] = None
        self.anomaly_map: Optional[np.ndarray] = None
        self.is_busy: bool = False
        self.threshold: float = 13.0   # default, adjustable

    def refresh_models(self):
        self.available_models = discover_models()


STATE = AppState()


# ═══════════════════════════════════════════════════════════════════════════
#  Image Texture Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bgr_to_texture_data(img_bgr: np.ndarray, size: Tuple[int, int] = (DISPLAY_W, DISPLAY_H)):
    """Convert BGR image to RGBA float32 array for Dear PyGui texture."""
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    return img_rgba.flatten()


def create_blank_texture():
    """Return flat RGBA float array for a blank dark image."""
    blank = np.full((DISPLAY_H, DISPLAY_W, 4), 0.12, dtype=np.float32)
    blank[:, :, 3] = 1.0
    return blank.flatten()


def update_texture(tag: str, data):
    """Update an existing dynamic texture."""
    dpg.set_value(tag, data)


# ═══════════════════════════════════════════════════════════════════════════
#  Core Functions
# ═══════════════════════════════════════════════════════════════════════════

def load_model_from_pkl(model_key: str, pkl_path: Path):
    """Load a PatchCore model from pickle, determine backbone from filename."""
    _lazy_imports()

    # Infer backbone from filename
    if "resnet101" in model_key.lower():
        backbone = "resnet101"
    else:
        backbone = "resnet50"

    # Load state to get saved config
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    cfg = _config_module.ModelConfig(
        name=model_key,
        backbone=state.get("backbone", backbone),
        resize=state.get("resize", _config_module.RESIZE_SIZE),
        center_crop=state.get("center_crop", _config_module.CENTER_CROP_SIZE),
        n_neighbors=state.get("n_neighbors", 9),
        output_dir=pkl_path.parent,
        batch_size=1,
    )
    model = _patchcore_model.PatchCore(cfg)
    model.load(pkl_path)
    return model


def run_inference(model, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    """Run PatchCore on a single BGR image. Returns (score, anomaly_map).

    Uses the model's _prepare_bank() for fp16 cached bank + pre-computed norms.
    """
    _lazy_imports()

    tfm = _dataset_module.get_transform(model.cfg.resize, model.cfg.center_crop)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = tfm(img_rgb).unsqueeze(0).to(model.device)  # (1, 3, C, C)

    with torch.no_grad():
        feat_maps = model.extractor(tensor)
        patches = _patchcore_model.aggregate_features(feat_maps)

        # Ensure GPU bank is cached (fp16 + pre-computed norms)
        model._prepare_bank()
        bank = model._bank_gpu              # (M, D) fp16
        bank_sq = model._bank_sq_norms      # (M,)   fp16
        k = model.cfg.n_neighbors

        q = patches.to(model.device, dtype=torch.float16)  # (P, D)
        q_sq = (q ** 2).sum(dim=1, keepdim=True)            # (P, 1)

        # ||q-b||² = ||q||² + ||b||² - 2·q·bᵀ
        chunk_size = 2048
        min_dists = []
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            q_chunk = q[start:end]
            q_sq_chunk = q_sq[start:end]
            dist_sq = q_sq_chunk + bank_sq.unsqueeze(0) - 2.0 * (q_chunk @ bank.t())
            dist_sq.clamp_(min=0.0)
            topk_sq, _ = dist_sq.topk(k, dim=1, largest=False)
            min_dists.append(topk_sq.sqrt().mean(dim=1))
        min_dists = torch.cat(min_dists, dim=0).float().cpu().numpy()

    H, W = model.spatial_shape
    amap = min_dists.reshape(H, W)
    from scipy.ndimage import gaussian_filter
    amap = gaussian_filter(amap, sigma=4)
    score = float(amap.max())
    return score, amap


def make_overlay(image_bgr: np.ndarray,
                 anomaly_map: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """Overlay anomaly heatmap on original image. Returns BGR."""
    h, w = image_bgr.shape[:2]
    display_size = min(h, w, DISPLAY_W)

    # Resize image
    img = cv2.resize(image_bgr, (display_size, display_size), interpolation=cv2.INTER_AREA)

    # Normalize anomaly map to 0–255
    amap = anomaly_map.copy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255
    amap = amap.astype(np.uint8)
    amap = cv2.resize(amap, (display_size, display_size), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════
#  GUI Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def _set_status(text: str, color=COL_GRAY):
    dpg.set_value("status_text", text)
    dpg.configure_item("status_text", color=color)


def _set_result(score: Optional[float], threshold: float):
    if score is None:
        dpg.set_value("result_text", "No result")
        dpg.configure_item("result_text", color=COL_GRAY)
        dpg.set_value("score_text", "")
        return

    dpg.set_value("score_text", f"Anomaly Score: {score:.4f}")
    if score > threshold:
        dpg.set_value("result_text", "DEFECT")
        dpg.configure_item("result_text", color=COL_FAIL)
    else:
        dpg.set_value("result_text", "OK")
        dpg.configure_item("result_text", color=COL_OK)


def cb_file_dialog_ok(sender, app_data):
    """File dialog OK callback."""
    selections = app_data.get("selections", {})
    if not selections:
        file_path = app_data.get("file_path_name", "")
    else:
        file_path = list(selections.values())[0]

    if file_path and os.path.isfile(file_path):
        _load_image(file_path)


def cb_file_dialog_cancel(sender, app_data):
    pass


def _load_image(path: str):
    """Load an image from disk and display it."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        _set_status(f"Failed to read: {Path(path).name}", COL_FAIL)
        return

    STATE.loaded_image_path = path
    STATE.loaded_image_bgr = img
    STATE.score = None
    STATE.anomaly_map = None

    # Update left texture (original)
    tex_data = bgr_to_texture_data(img)
    update_texture("tex_original", tex_data)

    # Clear right texture
    update_texture("tex_overlay", create_blank_texture())

    _set_status(f"Loaded: {Path(path).name}  ({img.shape[1]}×{img.shape[0]})", COL_GRAY)
    _set_result(None, STATE.threshold)
    dpg.set_value("filepath_label", str(path))


def cb_open_file(sender, app_data):
    dpg.show_item("file_dialog")


def cb_load_model(sender, app_data):
    """Load the selected PatchCore model."""
    model_key = dpg.get_value("model_combo")
    if not model_key or model_key not in STATE.available_models:
        _set_status("No model selected", COL_WARN)
        return

    if STATE.loaded_model_key == model_key:
        _set_status(f"Already loaded: {model_key}", COL_GRAY)
        return

    def _do():
        STATE.is_busy = True
        _set_status(f"Loading model: {model_key} ...", COL_WARN)
        try:
            pkl_path = STATE.available_models[model_key]
            model = load_model_from_pkl(model_key, pkl_path)
            STATE.loaded_model = model
            STATE.loaded_model_key = model_key
            crop = model.cfg.center_crop
            _set_status(f"Model loaded: {model_key}  (image: {crop}×{crop})", COL_OK)
        except Exception as e:
            _set_status(f"Load failed: {e}", COL_FAIL)
        finally:
            STATE.is_busy = False

    threading.Thread(target=_do, daemon=True).start()


def cb_run_inference(sender, app_data):
    """Run PatchCore on the loaded image."""
    if STATE.is_busy:
        return
    if STATE.loaded_model is None:
        _set_status("Load a model first!", COL_WARN)
        return
    if STATE.loaded_image_bgr is None:
        _set_status("Load an image first!", COL_WARN)
        return

    STATE.threshold = dpg.get_value("threshold_slider")

    def _do():
        STATE.is_busy = True
        _set_status("Running inference ...", COL_WARN)
        t0 = time.time()
        try:
            score, amap = run_inference(STATE.loaded_model, STATE.loaded_image_bgr)
            elapsed = time.time() - t0

            STATE.score = score
            STATE.anomaly_map = amap

            # Build overlay
            overlay_bgr = make_overlay(STATE.loaded_image_bgr, amap,
                                       alpha=dpg.get_value("alpha_slider"))
            tex_data = bgr_to_texture_data(overlay_bgr)
            update_texture("tex_overlay", tex_data)

            _set_status(f"Done in {elapsed:.2f}s  |  {STATE.loaded_model_key}", COL_OK)
            _set_result(score, STATE.threshold)

        except Exception as e:
            _set_status(f"Inference error: {e}", COL_FAIL)
            import traceback; traceback.print_exc()
        finally:
            STATE.is_busy = False

    threading.Thread(target=_do, daemon=True).start()


def cb_refresh_models(sender, app_data):
    STATE.refresh_models()
    items = list(STATE.available_models.keys())
    dpg.configure_item("model_combo", items=items)
    if items:
        dpg.set_value("model_combo", items[0])
    _set_status(f"Found {len(items)} model(s)", COL_GRAY)


def cb_alpha_changed(sender, app_data):
    """Re-render overlay when alpha slider changes."""
    if STATE.anomaly_map is not None and STATE.loaded_image_bgr is not None:
        overlay_bgr = make_overlay(STATE.loaded_image_bgr, STATE.anomaly_map,
                                   alpha=dpg.get_value("alpha_slider"))
        update_texture("tex_overlay", bgr_to_texture_data(overlay_bgr))


def cb_threshold_changed(sender, app_data):
    """Update OK/DEFECT label when threshold changes."""
    STATE.threshold = dpg.get_value("threshold_slider")
    if STATE.score is not None:
        _set_result(STATE.score, STATE.threshold)


# ═══════════════════════════════════════════════════════════════════════════
#  GUI Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_gui():
    dpg.create_context()

    # ── File dialog ──
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=cb_file_dialog_ok,
        cancel_callback=cb_file_dialog_cancel,
        tag="file_dialog",
        width=600,
        height=400,
    ):
        dpg.add_file_extension(".bmp", color=(0, 255, 0))
        dpg.add_file_extension(".png", color=(0, 200, 255))
        dpg.add_file_extension(".jpg", color=(0, 200, 255))
        dpg.add_file_extension(".jpeg", color=(0, 200, 255))
        dpg.add_file_extension(".tif", color=(200, 200, 0))
        dpg.add_file_extension(".*")

    # ── Texture registry ──
    blank = create_blank_texture()
    with dpg.texture_registry():
        dpg.add_dynamic_texture(DISPLAY_W, DISPLAY_H, blank, tag="tex_original")
        dpg.add_dynamic_texture(DISPLAY_W, DISPLAY_H, blank, tag="tex_overlay")

    # ── Theme ──
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)

    # ── Main Window ──
    with dpg.window(tag="main_window"):
        # Top bar
        dpg.add_text("PatchCore  O-Ring Anomaly Detection",
                      color=(100, 200, 255))
        dpg.add_separator()

        # Controls row
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open Image", callback=cb_open_file, width=110)
            dpg.add_combo(
                items=[], tag="model_combo", width=220,
                default_value="(no models found)",
            )
            dpg.add_button(label="Load Model", callback=cb_load_model, width=110)
            dpg.add_button(label="Refresh", callback=cb_refresh_models, width=80)
            dpg.add_button(label="  RUN  ", callback=cb_run_inference, width=100)

        dpg.add_spacer(height=4)

        # Sliders row
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                label="Threshold", tag="threshold_slider",
                default_value=13.0, min_value=1.0, max_value=30.0,
                width=200, callback=cb_threshold_changed,
            )
            dpg.add_slider_float(
                label="Overlay Alpha", tag="alpha_slider",
                default_value=0.5, min_value=0.0, max_value=1.0,
                width=200, callback=cb_alpha_changed,
            )

        dpg.add_spacer(height=4)

        # Images side by side
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original", color=COL_GRAY)
                dpg.add_image("tex_original", width=DISPLAY_W, height=DISPLAY_H)
            dpg.add_spacer(width=16)
            with dpg.group():
                dpg.add_text("Anomaly Overlay", color=COL_GRAY)
                dpg.add_image("tex_overlay", width=DISPLAY_W, height=DISPLAY_H)

        dpg.add_spacer(height=6)
        dpg.add_separator()

        # Result
        with dpg.group(horizontal=True):
            dpg.add_text("", tag="score_text", color=COL_GRAY)
            dpg.add_spacer(width=30)
            dpg.add_text("No result", tag="result_text", color=COL_GRAY)

        # Status bar
        dpg.add_text("", tag="filepath_label", color=(120, 120, 120))
        dpg.add_text("Ready", tag="status_text", color=COL_GRAY)

    dpg.bind_theme(global_theme)
    dpg.create_viewport(
        title="PatchCore O-Ring Inspector",
        width=WINDOW_W,
        height=WINDOW_H,
        resizable=True,
    )
    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)

    # Auto-discover models at startup
    STATE.refresh_models()
    items = list(STATE.available_models.keys())
    dpg.configure_item("model_combo", items=items)
    if items:
        dpg.set_value("model_combo", items[0])
        _set_status(f"Found {len(items)} model(s). Select and load to begin.", COL_GRAY)
    else:
        _set_status("No trained models found in patchcore/results/. Train first.", COL_WARN)

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_gui()
