"""
PySide6-based GUI for YOLO Segmentation Inference.
Allows users to load images, run inference, and visualize segmentation results.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QGroupBox, QGridLayout, QTextEdit, QSplitter, QMessageBox,
    QProgressBar, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

from ultralytics import YOLO


class InferenceThread(QThread):
    """Background thread for running inference on patches."""
    
    finished = Signal(object)  # Emits results list
    error = Signal(str)  # Emits error message
    
    def __init__(self, model, patches, conf_threshold, iou_threshold):
        super().__init__()
        self.model = model
        self.patches = patches
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def run(self):
        """Run inference in background."""
        try:
            results_list = []
            for patch in self.patches:
                results = self.model.predict(
                    source=patch,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                results_list.append(results[0] if results else None)
            self.finished.emit(results_list)
        except Exception as e:
            self.error.emit(str(e))


class YOLOSegmentationGUI(QMainWindow):
    """Main GUI window for YOLO segmentation inference."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.current_results = None
        self.current_image = None
        self.patches = []
        self.patch_coords = []
        self.patch_results = []
        self.current_patch_index = 0
        self.inference_thread = None
        
        self.setWindowTitle("YOLO Segmentation - Defect Detection")
        self.setGeometry(100, 100, 1400, 900)
        
        self.init_ui()
        self.load_default_model()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("YOLO Segmentation - Defect Detection System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Top controls
        control_group = self.create_control_panel()
        main_layout.addWidget(control_group)
        
        # Image display area (splitter for original and result)
        splitter = QSplitter(Qt.Horizontal)
        
        # Original image
        self.original_image_label = self.create_image_display("Original Image")
        splitter.addWidget(self.original_image_label)
        
        # Result image
        self.result_image_label = self.create_image_display("Segmentation Result")
        splitter.addWidget(self.result_image_label)
        
        splitter.setSizes([700, 700])
        main_layout.addWidget(splitter, stretch=1)
        
        # Bottom info panel
        info_group = self.create_info_panel()
        main_layout.addWidget(info_group)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_control_panel(self) -> QGroupBox:
        """Create the control panel with buttons and settings."""
        group = QGroupBox("Controls")
        layout = QGridLayout()
        
        # Model selection
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Select a model..."])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo, 0, 1, 1, 2)
        
        self.browse_model_btn = QPushButton("Browse Model...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        layout.addWidget(self.browse_model_btn, 0, 3)
        
        # Load image button
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_btn, 1, 0)
        
        # Run inference button
        self.run_inference_btn = QPushButton("Run Inference")
        self.run_inference_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.run_inference_btn.setEnabled(False)
        self.run_inference_btn.clicked.connect(self.run_inference)
        layout.addWidget(self.run_inference_btn, 1, 1)
        
        # Save result button
        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self.save_result)
        layout.addWidget(self.save_result_btn, 1, 2)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(self.clear_btn, 1, 3)

        # Patch navigation
        self.prev_patch_btn = QPushButton("Prev Patch")
        self.prev_patch_btn.setEnabled(False)
        self.prev_patch_btn.clicked.connect(self.show_prev_patch)
        layout.addWidget(self.prev_patch_btn, 2, 0)

        self.next_patch_btn = QPushButton("Next Patch")
        self.next_patch_btn.setEnabled(False)
        self.next_patch_btn.clicked.connect(self.show_next_patch)
        layout.addWidget(self.next_patch_btn, 2, 1)

        self.patch_index_label = QLabel("Patch: 0/0")
        layout.addWidget(self.patch_index_label, 2, 2, 1, 2)
        
        # Confidence threshold
        layout.addWidget(QLabel("Confidence:"), 3, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        layout.addWidget(self.conf_slider, 3, 1, 1, 2)
        
        self.conf_value_label = QLabel("0.50")
        layout.addWidget(self.conf_value_label, 3, 3)
        
        # IOU threshold
        layout.addWidget(QLabel("IOU Threshold:"), 4, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.on_iou_changed)
        layout.addWidget(self.iou_slider, 4, 1, 1, 2)
        
        self.iou_value_label = QLabel("0.45")
        layout.addWidget(self.iou_value_label, 4, 3)
        
        group.setLayout(layout)
        return group
    
    def create_image_display(self, title: str) -> QGroupBox:
        """Create an image display widget."""
        group = QGroupBox(title)
        layout = QVBoxLayout()
        
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(600, 600)
        label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        label.setText("No image loaded")
        
        layout.addWidget(label)
        group.setLayout(layout)
        
        # Store reference to the label
        if "Original" in title:
            self.original_display = label
        else:
            self.result_display = label
        
        return group
    
    def create_info_panel(self) -> QGroupBox:
        """Create information panel showing detection results."""
        group = QGroupBox("Detection Information")
        layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setPlainText("No results yet. Load an image and run inference.")
        
        layout.addWidget(self.info_text)
        group.setLayout(layout)
        
        return group
    
    def load_default_model(self):
        """Load available models from the runs directory."""
        self.model_combo.clear()
        self.model_combo.addItem("Select a model...")
        
        # Look for trained models
        runs_dir = Path(__file__).parent / "runs"
        if runs_dir.exists():
            best_models = list(runs_dir.glob("*/weights/best.pt"))
            last_models = list(runs_dir.glob("*/weights/last.pt"))
            
            all_models = []
            for model_path in best_models + last_models:
                model_name = f"{model_path.parent.parent.name}/{model_path.name}"
                all_models.append((model_name, str(model_path)))
            
            for display_name, model_path in all_models:
                self.model_combo.addItem(display_name, model_path)
            
            if all_models:
                self.model_combo.setCurrentIndex(1)  # Select first model
                self.status_label.setText(f"Found {len(all_models)} trained model(s)")
        else:
            self.status_label.setText("No trained models found. Train a model first.")
    
    def browse_model(self):
        """Browse for a model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "PyTorch Model (*.pt);;All Files (*.*)"
        )
        
        if file_path:
            model_name = Path(file_path).name
            self.model_combo.addItem(f"Custom: {model_name}", file_path)
            self.model_combo.setCurrentText(f"Custom: {model_name}")
    
    def on_model_changed(self, text):
        """Handle model selection change."""
        if text == "Select a model...":
            self.model = None
            return
        
        model_path = self.model_combo.currentData()
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.status_label.setText(f"Loaded model: {text}")
                
                # Enable inference if image is loaded
                if self.current_image_path:
                    self.run_inference_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
                self.model = None
    
    def load_image(self):
        """Load an image for inference."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*.*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                QMessageBox.warning(self, "Warning", "Failed to load image.")
                return
            
            # Display original image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.original_display.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.original_display.setPixmap(scaled_pixmap)
                
                self.status_label.setText(f"Loaded: {Path(file_path).name}")
                
                # Enable inference if model is loaded
                if self.model:
                    self.run_inference_btn.setEnabled(True)
                
                # Clear previous results
                self.result_display.clear()
                self.result_display.setText("Run inference to see results")
                self.current_results = None
                self.save_result_btn.setEnabled(False)
                self.patches = []
                self.patch_coords = []
                self.patch_results = []
                self.current_patch_index = 0
                self.patch_index_label.setText("Patch: 0/0")
                self.prev_patch_btn.setEnabled(False)
                self.next_patch_btn.setEnabled(False)
    
    def run_inference(self):
        """Run YOLO inference on the current image."""
        if not self.model or self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load both a model and an image.")
            return
        
        # Disable buttons during inference
        self.run_inference_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running inference...")
        
        # Get thresholds
        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        
        # Build patches (3x3 grid) from background-subtracted image
        fg_image = self._preprocess_background_subtract(self.current_image)
        self.patches, self.patch_coords = self._create_3x3_patches(fg_image, patch_size=640)

        if not self.patches:
            QMessageBox.warning(self, "Warning", "Failed to create patches from the image.")
            self.progress_bar.setVisible(False)
            self.run_inference_btn.setEnabled(True)
            self.load_image_btn.setEnabled(True)
            return

        # Run inference in background thread
        self.inference_thread = InferenceThread(
            self.model,
            self.patches,
            conf_threshold,
            iou_threshold
        )
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_finished(self, results_list):
        """Handle inference completion."""
        self.progress_bar.setVisible(False)
        self.run_inference_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        
        if results_list is None or len(results_list) == 0:
            QMessageBox.warning(self, "Error", "Inference failed to produce results.")
            self.status_label.setText("Inference failed")
            return
        
        self.patch_results = results_list
        self.current_patch_index = 0
        self._show_patch_result(self.current_patch_index)
        self.prev_patch_btn.setEnabled(len(self.patch_results) > 1)
        self.next_patch_btn.setEnabled(len(self.patch_results) > 1)
        
        self.status_label.setText("Inference completed")
        self.save_result_btn.setEnabled(True)
    
    def on_inference_error(self, error_msg):
        """Handle inference error."""
        self.progress_bar.setVisible(False)
        self.run_inference_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Inference Error", f"Error during inference:\n{error_msg}")
        self.status_label.setText("Inference error")
    
    def update_info_panel(self, results):
        """Update the information panel with detection results."""
        info_text = []
        info_text.append("=" * 60)
        info_text.append("DETECTION RESULTS")
        info_text.append("=" * 60)
        info_text.append("")

        if self.patch_coords:
            x, y, w, h = self.patch_coords[self.current_patch_index]
            info_text.append(f"Patch index: {self.current_patch_index + 1}/{len(self.patch_coords)}")
            info_text.append(f"Patch coords: x={x}, y={y}, w={w}, h={h}")
            info_text.append("")
        
        # Get detections
        if results.masks is not None and len(results.masks) > 0:
            n_detections = len(results.masks)
            info_text.append(f"Number of defects detected: {n_detections}")
            info_text.append("")
            
            # Get boxes and confidences
            boxes = results.boxes
            if boxes is not None:
                info_text.append("Defect Details:")
                info_text.append("-" * 60)
                
                for i, box in enumerate(boxes):
                    conf = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                    cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                    
                    # Get bounding box
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    info_text.append(f"  Defect #{i+1}:")
                    info_text.append(f"    Confidence: {conf:.3f} ({conf*100:.1f}%)")
                    info_text.append(f"    Bounding Box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                    info_text.append(f"    Area: {area:.0f} pixels²")
                    info_text.append("")
            
            # Get mask statistics
            if results.masks is not None:
                total_mask_area = 0
                for mask in results.masks.data:
                    total_mask_area += mask.sum().item()
                
                info_text.append(f"Total defect area: {total_mask_area:.0f} pixels")
        else:
            info_text.append("✓ No defects detected")
            info_text.append("")
            info_text.append("The image appears to be defect-free!")
        
        info_text.append("")
        info_text.append("=" * 60)
        
        self.info_text.setPlainText("\n".join(info_text))
    
    def on_conf_changed(self, value):
        """Handle confidence slider change."""
        self.conf_value_label.setText(f"{value/100:.2f}")
    
    def on_iou_changed(self, value):
        """Handle IOU slider change."""
        self.iou_value_label.setText(f"{value/100:.2f}")
    
    def save_result(self):
        """Save the result image."""
        if self.patch_results is None or len(self.patch_results) == 0:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            f"result_{Path(self.current_image_path).stem}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            result_image = self.patch_results[self.current_patch_index].plot()
            cv2.imwrite(file_path, result_image)
            self.status_label.setText(f"Saved: {Path(file_path).name}")
            QMessageBox.information(self, "Success", f"Result saved to:\n{file_path}")
    
    def clear_all(self):
        """Clear all displays and reset."""
        self.current_image_path = None
        self.current_results = None
        self.current_image = None
        self.patches = []
        self.patch_coords = []
        self.patch_results = []
        self.current_patch_index = 0
        
        self.original_display.clear()
        self.original_display.setText("No image loaded")
        
        self.result_display.clear()
        self.result_display.setText("No results")
        
        self.info_text.setPlainText("No results yet. Load an image and run inference.")
        
        self.run_inference_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)
        self.prev_patch_btn.setEnabled(False)
        self.next_patch_btn.setEnabled(False)
        self.patch_index_label.setText("Patch: 0/0")
        
        self.status_label.setText("Ready")

    def show_prev_patch(self):
        if not self.patch_results:
            return
        self.current_patch_index = (self.current_patch_index - 1) % len(self.patch_results)
        self._show_patch_result(self.current_patch_index)

    def show_next_patch(self):
        if not self.patch_results:
            return
        self.current_patch_index = (self.current_patch_index + 1) % len(self.patch_results)
        self._show_patch_result(self.current_patch_index)

    def _show_patch_result(self, index: int):
        results = self.patch_results[index]
        if results is None:
            self.result_display.setText("No results for this patch")
            return

        result_image = results.plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        h, w, ch = result_image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(result_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(
            self.result_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_display.setPixmap(scaled_pixmap)

        self.patch_index_label.setText(f"Patch: {index + 1}/{len(self.patch_results)}")
        self.update_info_panel(results)

    def _preprocess_background_subtract(self, image: np.ndarray) -> np.ndarray:
        mask = self._otsu_mask(image)
        result = image.copy()
        result[mask == 0] = 0
        return result

    def _otsu_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_ratio = np.mean(binary == 255)
        if white_ratio > 0.75:
            binary = cv2.bitwise_not(binary)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            mask = binary
        else:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(binary)
            mask[labels == largest_label] = 255

        return self._fill_holes(mask)

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        flood = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = mask | flood_inv
        return filled

    def _create_3x3_patches(self, image: np.ndarray, patch_size: int = 640):
        h, w = image.shape[:2]
        if w < patch_size or h < patch_size:
            return [], []

        xs = [0, max(0, (w - patch_size) // 2), max(0, w - patch_size)]
        ys = [0, max(0, (h - patch_size) // 2), max(0, h - patch_size)]

        patches = []
        coords = []
        for y in ys:
            for x in xs:
                patch = image[y:y + patch_size, x:x + patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)
                    coords.append((x, y, patch_size, patch_size))
        return patches, coords


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = YOLOSegmentationGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
