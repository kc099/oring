"""
PatchCore Inference GUI  (PySide6)

Full pipeline: Upload BMP (2048×1536) → 4×4 bin to 512×384 → YOLO seg →
Crop mask region → PatchCore anomaly detection (variable-size) → Heatmap.

Usage:
    python patchcore_inference_gui.py
    python patchcore_inference_gui.py --yolo path/to/best.pt
    python patchcore_inference_gui.py --patchcore path/to/model.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QSizePolicy,
    QStatusBar, QScrollArea, QComboBox, QSlider, QSpinBox,
)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtCore import Qt, QThread, Signal

from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_SEG_DIR = SCRIPT_DIR / "yolo_seg"
RESULTS_DIR = SCRIPT_DIR / "results_cropped"

# Add parent to path for patchcore imports
sys.path.insert(0, str(SCRIPT_DIR.parent))
from patchcore.config import ModelConfig, RESIZE_SIZE, CENTER_CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD
from patchcore.patchcore_model import PatchCore, FeatureExtractor, aggregate_features
from patchcore.dataset import get_transform

# ── Constants ────────────────────────────────────────────────────────────
YOLO_W, YOLO_H = 512, 384          # 4×4 binning of 2048×1536
YOLO_CONF = 0.25

# Per-model 2σ thresholds (will be re-evaluated after retraining)
MODEL_THRESHOLDS = {
    "model1_cropped_resnet50": 29.76,
    "model2_cropped_resnet50": 30.60,
}
DEFAULT_THRESHOLD = 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_best_yolo() -> str:
    runs_dir = YOLO_SEG_DIR / "runs"
    if not runs_dir.exists():
        return ""
    pts = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pts[0]) if pts else ""


def discover_patchcore_models():
    """Find all .pkl PatchCore models in results_cropped/."""
    models = {}
    for d in (RESULTS_DIR, SCRIPT_DIR / "results"):
        if d.exists():
            for pkl in d.rglob("*.pkl"):
                key = pkl.stem.replace("_patchcore", "")
                models[key] = pkl
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  PatchCore model loader
# ═══════════════════════════════════════════════════════════════════════════

def load_patchcore(pkl_path: Path) -> PatchCore:
    """Load PatchCore from pickle, infer backbone from filename."""
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    backbone = state.get("backbone", "resnet50")
    cfg = ModelConfig(
        name=pkl_path.stem,
        backbone=backbone,
        resize=state.get("resize", RESIZE_SIZE),
        center_crop=state.get("center_crop", CENTER_CROP_SIZE),
        n_neighbors=state.get("n_neighbors", 9),
        output_dir=pkl_path.parent,
        batch_size=1,
    )
    model = PatchCore(cfg)
    model.load(pkl_path)
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Inference pipeline
# ═══════════════════════════════════════════════════════════════════════════

def yolo_crop(yolo_model: YOLO, img_bgr: np.ndarray, padding: int = 0):
    """Resize → YOLO → return (cropped_image, bbox_on_resized).

    Returns:
        crop: cropped region from resized image
        bbox: (x1, y1, x2, y2) on the 640×480 resized image
        img_resized: the 640×480 image
    """
    img_resized = cv2.resize(img_bgr, (YOLO_W, YOLO_H), interpolation=cv2.INTER_AREA)

    results = yolo_model.predict(
        source=img_resized,
        imgsz=[YOLO_H, YOLO_W],
        conf=YOLO_CONF,
        device="cpu",
        verbose=False,
    )
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        # No mask → return full resized image
        return img_resized, (0, 0, YOLO_W, YOLO_H), img_resized

    all_pts = []
    for mask_xy in result.masks.xy:
        if len(mask_xy) > 0:
            all_pts.append(mask_xy)

    if not all_pts:
        return img_resized, (0, 0, YOLO_W, YOLO_H), img_resized

    all_pts = np.concatenate(all_pts, axis=0)
    x1 = max(0, int(all_pts[:, 0].min()) - padding)
    y1 = max(0, int(all_pts[:, 1].min()) - padding)
    x2 = min(YOLO_W, int(all_pts[:, 0].max()) + padding)
    y2 = min(YOLO_H, int(all_pts[:, 1].max()) + padding)

    if x2 <= x1 or y2 <= y1:
        return img_resized, (0, 0, YOLO_W, YOLO_H), img_resized

    crop = img_resized[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2), img_resized


def run_patchcore_inference(patchcore_model: PatchCore, crop_bgr: np.ndarray):
    """Run PatchCore on a single crop. Returns (score, anomaly_map)."""
    tfm = get_transform(patchcore_model.cfg.resize, patchcore_model.cfg.center_crop)
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = tfm(img_rgb).unsqueeze(0).to(patchcore_model.device)

    with torch.no_grad():
        feat_maps = patchcore_model.extractor(tensor)
        patches = aggregate_features(feat_maps)

        patchcore_model._prepare_bank()
        bank = patchcore_model._bank_gpu
        bank_sq = patchcore_model._bank_sq_norms
        k = patchcore_model.cfg.n_neighbors

        q = patches.to(patchcore_model.device, dtype=torch.float16)
        q_sq = (q ** 2).sum(dim=1, keepdim=True)

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

    H, W = patchcore_model.spatial_shape
    amap = min_dists.reshape(H, W)
    amap = gaussian_filter(amap, sigma=4)
    score = float(amap.max())
    return score, amap


def make_overlay(img_bgr: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5):
    """Overlay JET heatmap on image. Returns BGR image."""
    h, w = img_bgr.shape[:2]
    amap = anomaly_map.copy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255
    amap = amap.astype(np.uint8)
    amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════
#  Worker thread
# ═══════════════════════════════════════════════════════════════════════════

class InferenceWorker(QThread):
    finished = Signal(float, np.ndarray, np.ndarray, np.ndarray, tuple, float)
    error = Signal(str)

    def __init__(self, yolo_model, patchcore_model, img_bgr, padding=0):
        super().__init__()
        self.yolo_model = yolo_model
        self.patchcore_model = patchcore_model
        self.img_bgr = img_bgr
        self.padding = padding

    def run(self):
        try:
            t0 = time.time()
            crop, bbox, img_resized = yolo_crop(
                self.yolo_model, self.img_bgr, self.padding
            )
            score, amap = run_patchcore_inference(self.patchcore_model, crop)
            elapsed = time.time() - t0
            self.finished.emit(score, amap, crop, img_resized, bbox, elapsed)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════

def cv2_to_qpixmap(cv_img: np.ndarray, max_w: int = 0) -> QPixmap:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    if max_w > 0 and rgb.shape[1] > max_w:
        scale = max_w / rgb.shape[1]
        new_h = int(rgb.shape[0] * scale)
        rgb = cv2.resize(rgb, (max_w, new_h))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MainWindow(QMainWindow):
    def __init__(self, yolo_path: str, patchcore_path: str):
        super().__init__()
        self.setWindowTitle("PatchCore O-Ring Inspector")
        self.setMinimumSize(1200, 750)

        self.yolo_model = None
        self.patchcore_model = None
        self.patchcore_models = {}
        self.current_img = None
        self.worker = None
        self.threshold = DEFAULT_THRESHOLD

        # Load YOLO
        if yolo_path and Path(yolo_path).exists():
            self.yolo_model = YOLO(yolo_path)
            print(f"YOLO loaded: {yolo_path}")

        # Discover PatchCore models
        self.patchcore_models = discover_patchcore_models()

        # Load specific patchcore if given
        if patchcore_path and Path(patchcore_path).exists():
            self.patchcore_model = load_patchcore(Path(patchcore_path))

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── Top controls ─────────────────────────────────────────────────
        top = QHBoxLayout()

        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_upload.clicked.connect(self._on_upload)
        top.addWidget(self.btn_upload)

        top.addWidget(QLabel("PatchCore Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(250)
        model_keys = list(self.patchcore_models.keys())
        self.combo_model.addItems(model_keys)
        top.addWidget(self.combo_model)

        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self._on_load_model)
        top.addWidget(self.btn_load)

        self.btn_run = QPushButton("RUN")
        self.btn_run.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_run.clicked.connect(self._on_run)
        top.addWidget(self.btn_run)

        top.addStretch()
        layout.addLayout(top)

        # ── Threshold slider ─────────────────────────────────────────────
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(10, 300)  # 1.0 to 30.0
        self.slider_thresh.setValue(int(DEFAULT_THRESHOLD * 10))
        self.slider_thresh.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.slider_thresh)
        self.lbl_thresh = QLabel(f"{DEFAULT_THRESHOLD:.1f}")
        thresh_row.addWidget(self.lbl_thresh)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # ── Image panels ─────────────────────────────────────────────────
        panels = QHBoxLayout()

        # Original panel
        grp1 = QGroupBox("Original (resized)")
        lay1 = QVBoxLayout(grp1)
        self.lbl_original = QLabel("No image loaded")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setMinimumSize(380, 300)
        lay1.addWidget(self.lbl_original)
        panels.addWidget(grp1)

        # Crop panel
        grp2 = QGroupBox("YOLO Crop")
        lay2 = QVBoxLayout(grp2)
        self.lbl_crop = QLabel("—")
        self.lbl_crop.setAlignment(Qt.AlignCenter)
        self.lbl_crop.setMinimumSize(380, 300)
        lay2.addWidget(self.lbl_crop)
        panels.addWidget(grp2)

        # Anomaly overlay panel
        grp3 = QGroupBox("Anomaly Overlay")
        lay3 = QVBoxLayout(grp3)
        self.lbl_overlay = QLabel("—")
        self.lbl_overlay.setAlignment(Qt.AlignCenter)
        self.lbl_overlay.setMinimumSize(380, 300)
        lay3.addWidget(self.lbl_overlay)
        panels.addWidget(grp3)

        layout.addWidget(QWidget())  # spacer
        layout.addLayout(panels)

        # ── Verdict + info ───────────────────────────────────────────────
        info_row = QHBoxLayout()
        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setFont(QFont("Segoe UI", 18, QFont.Bold))
        info_row.addWidget(self.lbl_verdict)

        self.lbl_info = QLabel("")
        self.lbl_info.setFont(QFont("Consolas", 10))
        info_row.addWidget(self.lbl_info)
        info_row.addStretch()
        layout.addLayout(info_row)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        if self.patchcore_model:
            self.status_bar.showMessage("PatchCore model loaded", 3000)
        elif model_keys:
            self.status_bar.showMessage(
                f"Found {len(model_keys)} PatchCore model(s) — select and load", 5000
            )
        if not self.yolo_model:
            self.status_bar.showMessage("WARNING: No YOLO model found!", 5000)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.bmp *.png *.jpg *.jpeg *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_bar.showMessage(f"Failed to read: {path}", 5000)
            return
        self.current_img = img
        # Show resized preview
        preview = cv2.resize(img, (YOLO_W, YOLO_H), interpolation=cv2.INTER_AREA)
        self.lbl_original.setPixmap(cv2_to_qpixmap(preview, max_w=380))
        self.lbl_crop.setPixmap(QPixmap())
        self.lbl_crop.setText("—")
        self.lbl_overlay.setPixmap(QPixmap())
        self.lbl_overlay.setText("—")
        self.lbl_verdict.setText("")
        self.lbl_info.setText(f"{Path(path).name}  |  {img.shape[1]}×{img.shape[0]}")
        self.status_bar.showMessage(f"Loaded: {Path(path).name}", 3000)

    def _on_load_model(self):
        key = self.combo_model.currentText()
        if not key or key not in self.patchcore_models:
            self.status_bar.showMessage("No model selected", 3000)
            return
        pkl_path = self.patchcore_models[key]
        self.status_bar.showMessage(f"Loading {key}...")
        QApplication.processEvents()
        self.patchcore_model = load_patchcore(pkl_path)

        # Auto-set 2σ threshold for this model
        if key in MODEL_THRESHOLDS:
            self.threshold = MODEL_THRESHOLDS[key]
            self.slider_thresh.setValue(int(self.threshold * 10))
            self.lbl_thresh.setText(f"{self.threshold:.1f}")
            self.status_bar.showMessage(
                f"Loaded: {key}  |  Threshold auto-set to {self.threshold:.1f} (2σ)", 5000
            )
        else:
            self.status_bar.showMessage(f"Loaded: {key}", 5000)

    def _on_run(self):
        if self.current_img is None:
            self.status_bar.showMessage("Upload an image first!", 3000)
            return
        if self.yolo_model is None:
            self.status_bar.showMessage("No YOLO model loaded!", 3000)
            return
        if self.patchcore_model is None:
            self.status_bar.showMessage("Load a PatchCore model first!", 3000)
            return

        self.btn_run.setEnabled(False)
        self.status_bar.showMessage("Running pipeline: resize → YOLO → crop → PatchCore ...")
        QApplication.processEvents()

        self.worker = InferenceWorker(
            self.yolo_model, self.patchcore_model, self.current_img
        )
        self.worker.finished.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, score, amap, crop, img_resized, bbox, elapsed):
        # Show crop
        self.lbl_crop.setPixmap(cv2_to_qpixmap(crop, max_w=380))

        # Show overlay on crop
        overlay = make_overlay(crop, amap, alpha=0.5)
        self.lbl_overlay.setPixmap(cv2_to_qpixmap(overlay, max_w=380))

        # Also draw bbox on resized image
        img_with_box = img_resized.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.lbl_original.setPixmap(cv2_to_qpixmap(img_with_box, max_w=380))

        # Verdict
        if score > self.threshold:
            self.lbl_verdict.setText("DEFECT")
            self.lbl_verdict.setStyleSheet("color: red")
        else:
            self.lbl_verdict.setText("OK")
            self.lbl_verdict.setStyleSheet("color: green")

        self.lbl_info.setText(
            f"Score: {score:.4f}  |  Threshold: {self.threshold:.1f}  |  "
            f"Crop: {crop.shape[1]}×{crop.shape[0]}  |  Time: {elapsed:.2f}s"
        )

        self.btn_run.setEnabled(True)
        self.status_bar.showMessage(f"Done in {elapsed:.2f}s", 5000)

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.status_bar.showMessage(f"Error: {msg}", 10000)
        self.lbl_verdict.setText("ERROR")
        self.lbl_verdict.setStyleSheet("color: orange")

    def _on_threshold_changed(self, val):
        self.threshold = val / 10.0
        self.lbl_thresh.setText(f"{self.threshold:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PatchCore Inference GUI")
    p.add_argument("--yolo", type=str, default="",
                    help="Path to YOLO best.pt (auto-detected if omitted)")
    p.add_argument("--patchcore", type=str, default="",
                    help="Path to PatchCore .pkl (pick from GUI if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    yolo_path = args.yolo or find_best_yolo()
    patchcore_path = args.patchcore

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(yolo_path, patchcore_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
