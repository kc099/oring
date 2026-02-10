"""
GUI for reviewing rework measurements.

Workflow:
1) Select a folder of images
2) Click an image in the list
3) Press Detect to compute contours + stats
4) Review overlay and stats
5) Accept or Reject result
6) Export accepted measurements and averages

Author: GitHub Copilot
Date: February 6, 2026
"""

import sys
import csv
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QTextEdit, QSplitter, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap


def _largest_component(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(binary)
    mask[labels == largest_label] = 255
    return mask


def build_mask(image: np.ndarray, method: str, bg_value: int = 0, threshold: int = 30) -> np.ndarray:
    """Build a binary mask using different color space thresholds."""
    method = method.lower()

    if method == "otsu (gray)":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "hsv value":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "lab lightness":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        _, binary = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive (gray)":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5
        )
    elif method == "background subtraction":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg = np.full_like(gray, int(bg_value))
        diff = cv2.absdiff(gray, bg)
        _, binary = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If background becomes white, invert
    if np.mean(binary == 255) > 0.75:
        binary = cv2.bitwise_not(binary)

    # Cleanup (close small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return _largest_component(binary)


def find_inner_outer_contours(mask: np.ndarray):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, None

    areas = [cv2.contourArea(c) for c in contours]
    outer_idx = int(np.argmax(areas))
    outer = contours[outer_idx]

    inner = None
    h = hierarchy[0]
    for i, hinfo in enumerate(h):
        if hinfo[3] == outer_idx:
            if inner is None or cv2.contourArea(contours[i]) > cv2.contourArea(inner):
                inner = contours[i]

    return outer, inner


def fit_circle(contour: np.ndarray):
    (x, y), r = cv2.minEnclosingCircle(contour)
    return (float(x), float(y)), float(r)


def measure_ring(image: np.ndarray, method: str, bg_value: int = 0, threshold: int = 30) -> Optional[Dict]:
    mask = build_mask(image, method, bg_value=bg_value, threshold=threshold)
    outer, inner = find_inner_outer_contours(mask)
    if outer is None or inner is None:
        return None

    annular_area = int(np.count_nonzero(mask))

    (ox, oy), orad = fit_circle(outer)
    (ix, iy), irad = fit_circle(inner)

    center_dx = ox - ix
    center_dy = oy - iy
    center_dist = (center_dx ** 2 + center_dy ** 2) ** 0.5
    ring_thickness = orad - irad
    mean_radius = (orad + irad) / 2.0

    thickness_stats = compute_contour_distances(outer, inner)

    return {
        'method': method,
        'outer_radius': orad,
        'inner_radius': irad,
        'center_dx': center_dx,
        'center_dy': center_dy,
        'center_dist': center_dist,
        'ring_thickness': ring_thickness,
        'mean_radius': mean_radius,
        'min_thickness': thickness_stats['min_thickness'],
        'max_thickness': thickness_stats['max_thickness'],
        'annular_area': annular_area,
        'mask': mask,
        'outer_contour': outer,
        'inner_contour': inner
    }


def draw_overlay(image: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    vis = image.copy()
    cv2.drawContours(vis, [outer], -1, (0, 255, 0), 3)
    cv2.drawContours(vis, [inner], -1, (0, 0, 255), 3)
    return vis


def draw_overlay_on_mask(mask: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [outer], -1, (0, 255, 0), 2)
    cv2.drawContours(vis, [inner], -1, (0, 0, 255), 2)
    return vis


def compute_contour_distances(outer: np.ndarray, inner: np.ndarray) -> Dict[str, float]:
    """Compute min/max distances between outer and inner contours."""
    min_dist = float('inf')
    max_dist = 0.0

    for pt in outer:
        x, y = pt[0]
        d = abs(cv2.pointPolygonTest(inner, (float(x), float(y)), True))
        min_dist = min(min_dist, d)
        max_dist = max(max_dist, d)

    for pt in inner:
        x, y = pt[0]
        d = abs(cv2.pointPolygonTest(outer, (float(x), float(y)), True))
        min_dist = min(min_dist, d)
        max_dist = max(max_dist, d)

    if min_dist == float('inf'):
        min_dist = 0.0

    return {
        'min_thickness': float(min_dist),
        'max_thickness': float(max_dist)
    }


class ClickableImageLabel(QLabel):
    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_shape = None
        self._pixmap_size = None

    def set_display_info(self, image_shape, pixmap_size):
        self._image_shape = image_shape
        self._pixmap_size = pixmap_size

    def mousePressEvent(self, event):
        if self._image_shape is None or self._pixmap_size is None:
            return

        pixmap = self.pixmap()
        if pixmap is None:
            return

        # Compute offsets due to aspect-fit
        x_offset = (self.width() - self._pixmap_size.width()) // 2
        y_offset = (self.height() - self._pixmap_size.height()) // 2

        x = event.pos().x() - x_offset
        y = event.pos().y() - y_offset

        if x < 0 or y < 0 or x >= self._pixmap_size.width() or y >= self._pixmap_size.height():
            return

        img_h, img_w = self._image_shape[:2]
        scale_x = img_w / self._pixmap_size.width()
        scale_y = img_h / self._pixmap_size.height()

        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        self.clicked.emit(img_x, img_y)


class ReworkReviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rework Review - Concentricity QC")
        self.setGeometry(100, 100, 1400, 850)

        self.image_paths: List[Path] = []
        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.current_measurements: Optional[Dict] = None
        self.accepted: List[Dict] = []
        self.pick_bg_mode = False
        self.bg_value = 0
        self.threshold_value = 30
        self.image_min = None
        self.image_max = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel: folder + list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.select_folder_btn = QPushButton("📁 Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        left_layout.addWidget(self.select_folder_btn)

        self.image_list = QListWidget()
        self.image_list.itemSelectionChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list)

        splitter.addWidget(left_panel)

        # Center panel: image view + binary mask view
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        self.image_label = ClickableImageLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: #2b2b2b; color: #ddd; }")
        self.image_label.clicked.connect(self.on_image_click)
        center_layout.addWidget(self.image_label)

        self.mask_label = QLabel("Binary mask preview")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("QLabel { background: #1f1f1f; color: #ddd; }")
        center_layout.addWidget(self.mask_label)

        splitter.addWidget(center_panel)

        # Right panel: controls + stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Otsu (Gray)",
            "HSV Value",
            "LAB Lightness",
            "Adaptive (Gray)",
            "Background Subtraction"
        ])
        detect_layout.addWidget(self.method_combo)

        bg_controls = QHBoxLayout()
        self.pick_bg_btn = QPushButton("🎯 Pick BG")
        self.pick_bg_btn.clicked.connect(self.enable_pick_bg)
        bg_controls.addWidget(self.pick_bg_btn)

        self.bg_spin = QSpinBox()
        self.bg_spin.setRange(0, 255)
        self.bg_spin.setValue(self.bg_value)
        self.bg_spin.valueChanged.connect(self.on_bg_value_changed)
        bg_controls.addWidget(QLabel("BG"))
        bg_controls.addWidget(self.bg_spin)
        detect_layout.addLayout(bg_controls)

        thresh_controls = QHBoxLayout()
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(0, 255)
        self.thresh_spin.setValue(self.threshold_value)
        self.thresh_spin.valueChanged.connect(self.on_threshold_changed)
        thresh_controls.addWidget(QLabel("Threshold"))
        thresh_controls.addWidget(self.thresh_spin)
        detect_layout.addLayout(thresh_controls)

        self.detect_btn = QPushButton("🔍 Detect")
        self.detect_btn.clicked.connect(self.detect)
        detect_layout.addWidget(self.detect_btn)
        detect_group.setLayout(detect_layout)
        right_layout.addWidget(detect_group)

        stats_group = QGroupBox("Measurements")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        decision_group = QGroupBox("Decision")
        decision_layout = QHBoxLayout()
        self.accept_btn = QPushButton("✅ Accept")
        self.reject_btn = QPushButton("❌ Reject")
        self.accept_btn.clicked.connect(self.accept_current)
        self.reject_btn.clicked.connect(self.reject_current)
        decision_layout.addWidget(self.accept_btn)
        decision_layout.addWidget(self.reject_btn)
        decision_group.setLayout(decision_layout)
        right_layout.addWidget(decision_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        self.export_btn = QPushButton("💾 Export Accepted CSV + Averages")
        self.export_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)

        splitter.addWidget(right_panel)

        splitter.setSizes([250, 800, 350])

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.image_paths = []
        self.image_list.clear()

        for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png"):
            self.image_paths.extend(folder_path.glob(ext))

        self.image_paths = sorted(self.image_paths)
        for p in self.image_paths:
            self.image_list.addItem(p.name)

        if not self.image_paths:
            QMessageBox.information(self, "No Images", "No images found in the folder.")

    def on_image_selected(self):
        items = self.image_list.selectedItems()
        if not items:
            return

        name = items[0].text()
        match = next((p for p in self.image_paths if p.name == name), None)
        if match is None:
            return

        self.current_path = match
        self.current_image = cv2.imread(str(match))
        self.current_measurements = None
        self.stats_text.clear()
        self.show_image(self.current_image)
        self.show_mask(None)

    def on_image_click(self, x: int, y: int):
        if not self.pick_bg_mode or self.current_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
            value = int(gray[y, x])
            self.bg_spin.setValue(value)
            self.pick_bg_mode = False
            QMessageBox.information(self, "Background Picked", f"BG value set to: {value}")

    def enable_pick_bg(self):
        if self.current_image is None:
            return
        self.pick_bg_mode = True
        QMessageBox.information(self, "Pick Background", "Click on the background in the image.")

    def on_bg_value_changed(self, value: int):
        self.bg_value = int(value)

    def on_threshold_changed(self, value: int):
        self.threshold_value = int(value)

    def detect(self):
        if self.current_image is None:
            return

        method = self.method_combo.currentText()
        result = measure_ring(
            self.current_image,
            method,
            bg_value=self.bg_value,
            threshold=self.threshold_value
        )
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.image_min = int(gray.min())
        self.image_max = int(gray.max())
        if result is None:
            QMessageBox.warning(self, "Detection Failed", "Could not find inner/outer contours.")
            return

        self.current_measurements = result

        overlay = draw_overlay(self.current_image, result['outer_contour'], result['inner_contour'])
        self.show_image(overlay)

        self.show_mask(result['mask'])

        self.stats_text.setPlainText(
            f"method: {result['method']}\n"
            f"bg_value: {self.bg_value}\n"
            f"threshold: {self.threshold_value}\n"
            f"image_min: {self.image_min}\n"
            f"image_max: {self.image_max}\n"
            f"outer_radius: {result['outer_radius']:.2f}\n"
            f"inner_radius: {result['inner_radius']:.2f}\n"
            f"center_dx: {result['center_dx']:.2f}\n"
            f"center_dy: {result['center_dy']:.2f}\n"
            f"center_dist: {result['center_dist']:.2f}\n"
            f"ring_thickness (circle): {result['ring_thickness']:.2f}\n"
            f"min_thickness (contour): {result['min_thickness']:.2f}\n"
            f"max_thickness (contour): {result['max_thickness']:.2f}\n"
            f"mean_radius: {result['mean_radius']:.2f}\n"
            f"annular_area (mask nonzero): {result['annular_area']}"
        )

    def accept_current(self):
        if self.current_path is None or self.current_measurements is None:
            return

        row = {
            'image': self.current_path.name,
            'method': self.current_measurements['method'],
            'bg_value': self.bg_value,
            'threshold': self.threshold_value,
            'image_min': self.image_min,
            'image_max': self.image_max,
            'outer_radius': self.current_measurements['outer_radius'],
            'inner_radius': self.current_measurements['inner_radius'],
            'center_dx': self.current_measurements['center_dx'],
            'center_dy': self.current_measurements['center_dy'],
            'center_dist': self.current_measurements['center_dist'],
            'ring_thickness': self.current_measurements['ring_thickness'],
            'min_thickness': self.current_measurements['min_thickness'],
            'max_thickness': self.current_measurements['max_thickness'],
            'mean_radius': self.current_measurements['mean_radius'],
            'annular_area': self.current_measurements['annular_area']
        }
        self.accepted.append(row)
        QMessageBox.information(self, "Accepted", f"Accepted: {self.current_path.name}")

    def reject_current(self):
        if self.current_path is None:
            return
        QMessageBox.information(self, "Rejected", f"Rejected: {self.current_path.name}")

    def export_csv(self):
        if not self.accepted:
            QMessageBox.information(self, "No Data", "No accepted measurements to export.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "rework_measurements.csv", "CSV Files (*.csv)")
        if not save_path:
            return

        fieldnames = list(self.accepted[0].keys())
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.accepted)

        # Compute averages
        avg = {}
        for key in fieldnames:
            if key in ('image', 'method', 'bg_value', 'threshold'):
                continue
            values = [row[key] for row in self.accepted if row.get(key) is not None]
            avg[key] = float(np.mean(values)) if values else None

        avg_path = str(Path(save_path).with_name(Path(save_path).stem + "_avg.csv"))
        with open(avg_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
            writer.writeheader()
            for k, v in avg.items():
                writer.writerow({'metric': k, 'value': v})

        QMessageBox.information(self, "Exported", f"Saved CSV:\n{save_path}\n\nAverages:\n{avg_path}")

    def show_image(self, image: np.ndarray):
        if image is None:
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.set_display_info(rgb.shape, scaled.size())

    def show_mask(self, mask: Optional[np.ndarray]):
        if mask is None:
            self.mask_label.setText("Binary mask preview")
            self.mask_label.setPixmap(QPixmap())
            return

        if len(mask.shape) == 2:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            qformat = QImage.Format_BGR888
        else:
            vis = mask
            qformat = QImage.Format_BGR888

        h, w, ch = vis.shape
        bytes_per_line = ch * w
        qimg = QImage(vis.data, w, h, bytes_per_line, qformat)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.mask_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mask_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            self.show_image(self.current_image)
        if self.current_measurements is not None and 'mask' in self.current_measurements:
            self.show_mask(self.current_measurements['mask'])


def main():
    app = QApplication(sys.argv)
    window = ReworkReviewWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
