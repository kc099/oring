"""
PySide6 Inference GUI for YOLO11-seg Defect Detection.

Upload a 2048x1536 BMP image, the model resizes it to 640x480 internally,
runs YOLO segmentation inference, and displays the result overlaid on the
original-resolution image.

Usage:
    python inference_gui.py                              # auto-find best.pt
    python inference_gui.py --model path/to/best.pt      # explicit model
"""

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QSizePolicy,
    QStatusBar, QScrollArea,
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
INFER_IMGSZ = 640  # model input size
CONF_THRESHOLD = 0.25


def find_best_model() -> str:
    """Auto-detect the best.pt from the latest training run."""
    runs_dir = SCRIPT_DIR / "runs"
    if not runs_dir.exists():
        return ""
    # Find all best.pt files and pick the newest
    best_pts = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(best_pts[0]) if best_pts else ""


def cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Convert BGR OpenCV image to QPixmap."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class InferenceWindow(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.setWindowTitle("YOLO11-seg Defect Inspection")
        self.setMinimumSize(1000, 700)

        self.model = None
        self.model_path = model_path
        self.current_image_path = ""
        self.original_img = None

        self._load_model()
        self._build_ui()

    def _load_model(self):
        if self.model_path and Path(self.model_path).exists():
            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
        else:
            print("No model loaded — use 'Load Model' button or --model flag.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = QHBoxLayout()

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self._on_load_model)
        toolbar.addWidget(self.btn_load_model)

        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_upload.clicked.connect(self._on_upload)
        toolbar.addWidget(self.btn_upload)

        self.lbl_model_info = QLabel("Model: (none)")
        toolbar.addWidget(self.lbl_model_info)
        toolbar.addStretch()

        main_layout.addLayout(toolbar)

        # ── Image display area ───────────────────────────────────────────
        img_group = QGroupBox("Result")
        img_layout = QVBoxLayout(img_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.lbl_image = QLabel("Upload an image to start inference")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setWidget(self.lbl_image)
        img_layout.addWidget(scroll_area)

        main_layout.addWidget(img_group)

        # ── Info labels ──────────────────────────────────────────────────
        self.lbl_info = QLabel("")
        self.lbl_info.setFont(QFont("Consolas", 10))
        main_layout.addWidget(self.lbl_info)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        if self.model:
            self.lbl_model_info.setText(f"Model: {Path(self.model_path).name}")

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", str(SCRIPT_DIR), "PyTorch Model (*.pt)"
        )
        if path:
            self.model_path = path
            self.model = YOLO(path)
            self.lbl_model_info.setText(f"Model: {Path(path).name}")
            self.status_bar.showMessage(f"Model loaded: {path}", 5000)

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.bmp *.png *.jpg *.jpeg *.tiff)",
        )
        if not path:
            return
        self.current_image_path = path
        self._run_inference(path)

    def _run_inference(self, image_path: str):
        if self.model is None:
            self.status_bar.showMessage("No model loaded!", 5000)
            return

        self.status_bar.showMessage("Running inference...")
        QApplication.processEvents()

        # Read original image
        img = cv2.imread(image_path)
        if img is None:
            self.status_bar.showMessage(f"Failed to read image: {image_path}", 5000)
            return
        self.original_img = img.copy()
        orig_h, orig_w = img.shape[:2]

        # Resize to model input size for inference
        img_resized = cv2.resize(img, (INFER_IMGSZ, int(INFER_IMGSZ * 480 / 640)),
                                  interpolation=cv2.INTER_AREA)

        # Run YOLO inference (CPU only)
        results = self.model.predict(
            source=img_resized,
            imgsz=INFER_IMGSZ,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
        )

        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0

        # Draw results on original-resolution image
        annotated = img.copy()

        if result.masks is not None and num_detections > 0:
            # Scale masks back to original resolution
            for i, mask in enumerate(result.masks.xy):
                # mask is array of (x, y) polygon points in resized coords
                # Scale from resized → original
                scale_x = orig_w / img_resized.shape[1]
                scale_y = orig_h / img_resized.shape[0]
                scaled_pts = mask.copy()
                scaled_pts[:, 0] *= scale_x
                scaled_pts[:, 1] *= scale_y
                pts = scaled_pts.astype(np.int32)

                # Draw filled semi-transparent mask
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

                # Draw polygon outline
                cv2.polylines(annotated, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                # Confidence label
                conf = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                cls_name = result.names.get(cls_id, "defect")
                label = f"{cls_name} {conf:.2f}"
                x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
                cv2.putText(annotated, label, (x_min, max(y_min - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Display result
        # Scale for display (fit to ~900px width)
        display_w = min(900, orig_w)
        scale = display_w / orig_w
        display_h = int(orig_h * scale)
        display_img = cv2.resize(annotated, (display_w, display_h))

        pixmap = cv2_to_qpixmap(display_img)
        self.lbl_image.setPixmap(pixmap)

        # Info text
        verdict = "DEFECT DETECTED" if num_detections > 0 else "OK (no defects)"
        color = "red" if num_detections > 0 else "green"
        self.lbl_info.setText(
            f"File: {Path(image_path).name}  |  "
            f"Original: {orig_w}x{orig_h}  |  "
            f"Detections: {num_detections}  |  "
            f'<span style="color:{color}; font-weight:bold">{verdict}</span>'
        )
        self.lbl_info.setTextFormat(Qt.RichText)

        self.status_bar.showMessage(
            f"Inference complete — {num_detections} detection(s)", 5000
        )


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11-seg Inference GUI")
    p.add_argument("--model", type=str, default="",
                    help="Path to best.pt (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model or find_best_model()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = InferenceWindow(model_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
