"""
Image Segmentation Mask Labeler - GUI Version
A GUI tool for creating segmentation masks by drawing polygons on images.

Author: AI Assistant
Date: February 2, 2026
"""

import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional
import glob
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap, QColor


class ImageCanvas(QLabel):
    """Custom Qt widget for displaying and drawing on images with zoom/pan support.
    
    All polygon coordinates are stored in ORIGINAL image space.
    Zoom and pan only affect the display — coordinates are always mapped back
    to the original resolution before storing, and mapped forward for rendering.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        
        self.current_points = []        # points in ORIGINAL image coords
        self.completed_polygons = []    # polygons in ORIGINAL image coords
        self.selected_polygon_index = None
        self.image = None               # original image (never modified)
        self.display_pixmap = None
        
        # Zoom / Pan state
        self.zoom_level = 1.0           # 1.0 = fit to widget
        self.min_zoom = 1.0             # set when image is loaded (fit-to-widget)
        self.max_zoom = 10.0
        self.pan_offset_x = 0.0         # pan in DISPLAY (widget) pixels
        self.pan_offset_y = 0.0
        self._pan_active = False
        self._pan_start = None
        self._pan_start_offset = None
        
        # Colors for polygons
        self.colors = [
            QColor(0, 255, 0),      # Green
            QColor(255, 0, 0),      # Blue
            QColor(0, 0, 255),      # Red
            QColor(255, 255, 0),    # Cyan
            QColor(255, 0, 255),    # Magenta
            QColor(0, 255, 255),    # Yellow
        ]
        
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")
    
    def set_image(self, cv_image):
        """Set the image to display (resets zoom/pan)."""
        self.image = cv_image.copy()
        self.current_points = []
        self.completed_polygons = []
        self.selected_polygon_index = None
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update_display()
    
    def _base_scale(self):
        """Scale factor to fit the full image into the widget (zoom_level=1.0)."""
        if self.image is None:
            return 1.0
        ih, iw = self.image.shape[:2]
        ww, wh = self.width(), self.height()
        if iw == 0 or ih == 0 or ww == 0 or wh == 0:
            return 1.0
        return min(ww / iw, wh / ih)
    
    def _effective_scale(self):
        """Total scale from original-image pixels to display pixels."""
        return self._base_scale() * self.zoom_level
    
    def _image_to_widget(self, img_x, img_y):
        """Convert original-image coords → widget coords."""
        s = self._effective_scale()
        ih, iw = self.image.shape[:2]
        # Displayed image size
        dw = iw * s
        dh = ih * s
        # Centering offset
        cx = (self.width() - dw) / 2.0 + self.pan_offset_x
        cy = (self.height() - dh) / 2.0 + self.pan_offset_y
        wx = cx + img_x * s
        wy = cy + img_y * s
        return wx, wy
    
    def _widget_to_image(self, wx, wy):
        """Convert widget coords → original-image coords (float). Returns None if outside."""
        if self.image is None:
            return None
        s = self._effective_scale()
        ih, iw = self.image.shape[:2]
        dw = iw * s
        dh = ih * s
        cx = (self.width() - dw) / 2.0 + self.pan_offset_x
        cy = (self.height() - dh) / 2.0 + self.pan_offset_y
        ix = (wx - cx) / s
        iy = (wy - cy) / s
        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
            return None
        return (int(round(ix)), int(round(iy)))
    
    def update_display(self):
        """Render the image + drawings at the current zoom/pan and set the pixmap."""
        if self.image is None:
            return
        
        s = self._effective_scale()
        ih, iw = self.image.shape[:2]
        dw = int(iw * s)
        dh = int(ih * s)
        
        # Resize the image for display
        display_image = cv2.resize(self.image, (dw, dh), interpolation=cv2.INTER_LINEAR)
        
        # Draw completed polygons (scale coords)
        for i, polygon in enumerate(self.completed_polygons):
            color_idx = i % len(self.colors)
            color = self.colors[color_idx]
            
            pts = np.array([(int(px * s), int(py * s)) for (px, py) in polygon], dtype=np.int32)
            
            overlay = display_image.copy()
            cv2.fillPoly(overlay, [pts], (color.blue(), color.green(), color.red()))
            display_image = cv2.addWeighted(display_image, 0.6, overlay, 0.4, 0)
            
            if i == self.selected_polygon_index:
                cv2.polylines(display_image, [pts], True, (255, 255, 255), 8)
                cv2.polylines(display_image, [pts], True, (color.blue(), color.green(), color.red()), 5)
            else:
                cv2.polylines(display_image, [pts], True, (color.blue(), color.green(), color.red()), 3)
        
        # Draw current polygon (in-progress)
        if len(self.current_points) > 0:
            disp_pts = [(int(px * s), int(py * s)) for (px, py) in self.current_points]
            for j in range(len(disp_pts) - 1):
                cv2.line(display_image, disp_pts[j], disp_pts[j + 1], (0, 255, 255), 3)
            for dp in disp_pts:
                cv2.circle(display_image, dp, 6, (0, 255, 255), -1)
                cv2.circle(display_image, dp, 8, (255, 255, 255), 2)
        
        # Convert to QPixmap
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        full_pixmap = QPixmap.fromImage(q_image)
        
        # Create a widget-sized canvas and paint the zoomed image with pan offset
        canvas_pixmap = QPixmap(self.size())
        canvas_pixmap.fill(QColor(43, 43, 43))  # dark background
        
        from PySide6.QtGui import QPainter
        painter = QPainter(canvas_pixmap)
        # Where to draw the image on the widget
        cx = (self.width() - dw) / 2.0 + self.pan_offset_x
        cy = (self.height() - dh) / 2.0 + self.pan_offset_y
        painter.drawPixmap(int(cx), int(cy), full_pixmap)
        
        # Draw zoom indicator
        zoom_pct = int(self.zoom_level * 100)
        painter.setPen(QColor(255, 255, 255))
        from PySide6.QtGui import QFont as QF
        painter.setFont(QF("Arial", 10))
        painter.drawText(10, 20, f"Zoom: {zoom_pct}%  (Scroll to zoom, Middle-drag to pan)")
        
        painter.end()
        
        self.setPixmap(canvas_pixmap)
    
    def wheelEvent(self, event):
        """Zoom in/out centered on cursor position."""
        if self.image is None:
            return
        
        # Mouse position in widget
        mx, my = event.position().x(), event.position().y()
        
        # Image coord under cursor BEFORE zoom
        img_pt = self._widget_to_image(mx, my)
        
        # Compute new zoom
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        
        new_zoom = self.zoom_level * factor
        new_zoom = max(1.0, min(self.max_zoom, new_zoom))
        
        if new_zoom == self.zoom_level:
            return
        
        # If we had a valid image point, keep it under the cursor after zoom
        if img_pt is not None:
            old_s = self._effective_scale()
            self.zoom_level = new_zoom
            new_s = self._effective_scale()
            ih, iw = self.image.shape[:2]
            
            # Where that image point would end up with new zoom + old pan
            dw_new = iw * new_s
            dh_new = ih * new_s
            cx_new = (self.width() - dw_new) / 2.0 + self.pan_offset_x
            cy_new = (self.height() - dh_new) / 2.0 + self.pan_offset_y
            wx_new = cx_new + img_pt[0] * new_s
            wy_new = cy_new + img_pt[1] * new_s
            
            # Adjust pan so the point stays under cursor
            self.pan_offset_x += mx - wx_new
            self.pan_offset_y += my - wy_new
        else:
            self.zoom_level = new_zoom
        
        # Reset pan if zoomed back to fit
        if self.zoom_level <= 1.0:
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
        
        self.update_display()
    
    def mousePressEvent(self, event):
        """Handle mouse press: left=draw/select, middle=pan."""
        if event.button() == Qt.MiddleButton:
            self._pan_active = True
            self._pan_start = event.pos()
            self._pan_start_offset = (self.pan_offset_x, self.pan_offset_y)
            self.setCursor(Qt.ClosedHandCursor)
            return
        
        if event.button() == Qt.LeftButton and self.image is not None:
            pos = self._widget_to_image(event.pos().x(), event.pos().y())
            if pos:
                if len(self.current_points) == 0:
                    clicked_polygon = self.find_polygon_at_point(pos)
                    if clicked_polygon is not None:
                        self.selected_polygon_index = clicked_polygon
                        self.update_display()
                        if hasattr(self.parent(), 'update_polygon_count'):
                            self.parent().update_polygon_count()
                        return
                
                self.current_points.append(pos)
                self.selected_polygon_index = None
                self.update_display()
                if hasattr(self.parent(), 'update_polygon_count'):
                    self.parent().update_polygon_count()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for panning."""
        if self._pan_active and self._pan_start is not None:
            dx = event.pos().x() - self._pan_start.x()
            dy = event.pos().y() - self._pan_start.y()
            self.pan_offset_x = self._pan_start_offset[0] + dx
            self.pan_offset_y = self._pan_start_offset[1] + dy
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """Stop panning."""
        if event.button() == Qt.MiddleButton:
            self._pan_active = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
    
    def find_polygon_at_point(self, point):
        """Find which polygon contains the given point (original image coords)."""
        for i in range(len(self.completed_polygons) - 1, -1, -1):
            pts = np.array(self.completed_polygons[i], dtype=np.int32)
            if cv2.pointPolygonTest(pts, point, False) >= 0:
                return i
        return None
    
    def map_to_image(self, widget_pos):
        """Map widget coordinates to original image coordinates (legacy compat)."""
        return self._widget_to_image(widget_pos.x(), widget_pos.y())
    
    def reset_zoom(self):
        """Reset zoom to fit-to-widget."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update_display()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_display()


class LabelingWindow(QMainWindow):
    """Labeling window with buttons and better UI."""
    
    def __init__(self, image_folders, mask_folder):
        super().__init__()
        self.image_folders = image_folders
        self.mask_folder = mask_folder
        self.image_files = self._load_image_files()
        self.current_image_idx = 0
        
        if not self.image_files:
            raise ValueError("No images found in specified folders!")
        
        self.init_ui()
        self._load_current_image()
    
    def _load_image_files(self):
        """Load all image files from specified folders."""
        image_files = []
        seen_files = set()  # Track unique file paths to avoid duplicates
        extensions = ['*.bmp', '*.BMP']
        
        for folder in self.image_folders:
            if not os.path.exists(folder):
                continue
                
            for ext in extensions:
                pattern = os.path.join(folder, ext)
                for filepath in glob.glob(pattern):
                    # Use full path to check for duplicates
                    if filepath not in seen_files:
                        filename = os.path.basename(filepath)
                        image_files.append((folder, filename))
                        seen_files.add(filepath)
        
        print(f"Found {len(image_files)} unique images")
        return sorted(image_files)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Segmentation Labeler - Draw Polygons")
        self.setGeometry(50, 50, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side - Image canvas
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        
        self.canvas = ImageCanvas(self)
        self.canvas.setMinimumSize(800, 600)
        canvas_layout.addWidget(self.canvas)
        
        main_layout.addWidget(canvas_container, stretch=4)
        
        # Right side - Control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(350)
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-size: 11pt;
            }
        """)
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("🎨 Control Panel")
        title.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #333;
                padding: 10px;
                background-color: #e3f2fd;
                border-radius: 5px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        # Image info
        info_group = QGroupBox("📁 Current Image")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout()
        
        self.image_info_label = QLabel()
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        info_layout.addWidget(self.image_info_label)
        
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #1976D2;
                padding: 8px;
                background-color: #e3f2fd;
                border-radius: 5px;
            }
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.progress_label)
        
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        # Drawing controls
        draw_group = QGroupBox("✏️ Drawing Controls")
        draw_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        draw_layout = QVBoxLayout()
        
        self.polygon_count_label = QLabel("Polygons: 0")
        self.polygon_count_label.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                padding: 5px;
                background-color: #fff3e0;
                border-radius: 3px;
            }
        """)
        draw_layout.addWidget(self.polygon_count_label)
        
        btn_style = """
            QPushButton {
                padding: 12px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """
        
        self.close_polygon_btn = QPushButton("🔒 Close Polygon (C)")
        self.close_polygon_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.close_polygon_btn.clicked.connect(self.close_polygon)
        draw_layout.addWidget(self.close_polygon_btn)
        
        self.undo_btn = QPushButton("↩️ Undo (Z)")
        self.undo_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #ff9800;
                color: white;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        self.undo_btn.clicked.connect(self.undo)
        draw_layout.addWidget(self.undo_btn)
        
        self.delete_selected_btn = QPushButton("❌ Delete Selected (Del)")
        self.delete_selected_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #E91E63;
                color: white;
            }
            QPushButton:hover {
                background-color: #C2185B;
            }
        """)
        self.delete_selected_btn.clicked.connect(self.delete_selected_polygon)
        draw_layout.addWidget(self.delete_selected_btn)
        
        self.clear_all_btn = QPushButton("🗑️ Clear All Polygons")
        self.clear_all_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #f44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_all_btn.clicked.connect(self.clear_all)
        draw_layout.addWidget(self.clear_all_btn)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("🔍+ Zoom In")
        self.zoom_in_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #795548;
                color: white;
            }
            QPushButton:hover {
                background-color: #6D4C41;
            }
        """)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("🔍- Zoom Out")
        self.zoom_out_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #795548;
                color: white;
            }
            QPushButton:hover {
                background-color: #6D4C41;
            }
        """)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        draw_layout.addLayout(zoom_layout)
        
        self.reset_zoom_btn = QPushButton("🔄 Reset Zoom (R)")
        self.reset_zoom_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #9E9E9E;
                color: white;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        draw_layout.addWidget(self.reset_zoom_btn)
        
        draw_group.setLayout(draw_layout)
        control_layout.addWidget(draw_group)
        
        # Navigation controls
        nav_group = QGroupBox("🔄 Navigation")
        nav_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        nav_layout = QVBoxLayout()
        
        self.save_next_btn = QPushButton("💾 Save & Next (S)")
        self.save_next_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.save_next_btn.clicked.connect(self.save_and_next)
        nav_layout.addWidget(self.save_next_btn)
        
        nav_buttons = QHBoxLayout()
        
        self.prev_btn = QPushButton("⬅️ Previous (P)")
        self.prev_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #607D8B;
                color: white;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.prev_btn.clicked.connect(self.previous_image)
        nav_buttons.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ➡️ (N)")
        self.next_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #607D8B;
                color: white;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.next_btn.clicked.connect(self.next_image)
        nav_buttons.addWidget(self.next_btn)
        
        nav_layout.addLayout(nav_buttons)
        
        nav_group.setLayout(nav_layout)
        control_layout.addWidget(nav_group)
        
        # Instructions
        instructions_group = QGroupBox("ℹ️ Quick Help")
        instructions_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        instructions_layout = QVBoxLayout()
        
        instructions = QLabel(
            "• Click on image to add points\n"
            "• Click 'Close Polygon' or press C\n"
            "• Click on polygon to select it\n"
            "• Press Delete to remove selected\n"
            "• Use 'Undo' to remove last action\n"
            "• Scroll wheel to zoom in/out\n"
            "• Middle-click drag to pan\n"
            "• Press R to reset zoom\n"
            "• Save & Next to save mask\n"
            "• Press Q or Escape to quit"
        )
        instructions.setStyleSheet("""
            QLabel {
                background-color: #fff9e6;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ffd966;
                line-height: 1.5;
            }
        """)
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        instructions_group.setLayout(instructions_layout)
        control_layout.addWidget(instructions_group)
        
        control_layout.addStretch()
        
        # Quit button
        self.quit_btn = QPushButton("❌ Quit Labeling (Q)")
        self.quit_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        self.quit_btn.clicked.connect(self.close)
        control_layout.addWidget(self.quit_btn)
        
        main_layout.addWidget(control_panel, stretch=1)
        
        # Keyboard shortcuts
        self.installEventFilter(self)
    
    def _load_current_image(self):
        """Load the current image."""
        folder, filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(folder, filename)
        
        print(f"  [Index {self.current_image_idx}] Loading: {filename} from {os.path.basename(folder)}/")
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            QMessageBox.warning(self, "Error", f"Failed to load: {image_path}")
            return
        
        self.current_folder = folder
        self.current_filename = filename
        self.height, self.width = self.original_image.shape[:2]
        
        # Initialize mask
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Update canvas - this clears previous polygons
        self.canvas.set_image(self.original_image)
        
        # Try to load existing mask if it exists
        self._load_existing_mask()
        
        # Update info labels
        folder_name = os.path.basename(folder)
        self.image_info_label.setText(
            f"<b>Folder:</b> {folder_name}<br>"
            f"<b>File:</b> {filename}<br>"
            f"<b>Size:</b> {self.width}x{self.height}"
        )
        
        self.progress_label.setText(
            f"Image {self.current_image_idx + 1} of {len(self.image_files)}"
        )
        
        self.update_polygon_count()
        
        # Force UI update
        self.canvas.update()
        QApplication.processEvents()
        
        print(f"  ✓ Image loaded successfully")
    
    def _load_existing_mask(self):
        """Load existing mask JSON if it exists."""
        # Create mask filename
        mask_filename = os.path.splitext(self.current_filename)[0] + '_mask.json'
        mask_path = os.path.join(self.mask_folder, mask_filename)
        
        # Check if mask file exists
        if os.path.exists(mask_path):
            try:
                with open(mask_path, 'r', encoding='utf-8') as f:
                    mask_data = json.load(f)
                
                # Load polygons from JSON
                self.canvas.completed_polygons = []
                for polygon_data in mask_data.get('polygons', []):
                    points = [(pt['x'], pt['y']) for pt in polygon_data['points']]
                    self.canvas.completed_polygons.append(points)
                    
                    # Update mask
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(self.mask, [pts], 255)
                
                # Update display
                self.canvas.update_display()
                
                num_polygons = len(self.canvas.completed_polygons)
                print(f"  ✓ Loaded existing mask with {num_polygons} polygon(s)")
                
            except Exception as e:
                print(f"  ⚠ Failed to load existing mask: {e}")
        else:
            print(f"  ℹ No existing mask found")

    def _resolve_mask_subfolder(self, folder_name: str) -> str:
        """Resolve mask folder path based on selected mask folder."""
        # If selected folder already points to a *_masks folder, use it
        if os.path.basename(self.mask_folder).endswith('_masks'):
            return self.mask_folder
        
        # Otherwise, use root masks folder + subfolder
        return os.path.join(self.mask_folder, f"{folder_name}_masks")
    
    def update_polygon_count(self):
        """Update polygon count label."""
        count = len(self.canvas.completed_polygons)
        points = len(self.canvas.current_points)
        
        text = f"Polygons: {count}"
        if points > 0:
            text += f" | Current points: {points}"
        elif self.canvas.selected_polygon_index is not None:
            text += f" | Selected: #{self.canvas.selected_polygon_index + 1}"
        
        self.polygon_count_label.setText(text)
    
    def close_polygon(self):
        """Close the current polygon."""
        if len(self.canvas.current_points) >= 3:
            # Add to completed polygons
            self.canvas.completed_polygons.append(self.canvas.current_points.copy())
            
            # Update mask with white (255)
            pts = np.array(self.canvas.current_points, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], 255)
            
            # Clear current points
            self.canvas.current_points = []
            self.canvas.update_display()
            self.update_polygon_count()
            
            print(f"✓ Polygon closed. Total polygons: {len(self.canvas.completed_polygons)}")
        else:
            QMessageBox.information(self, "Info", "Need at least 3 points to close a polygon")
    
    def undo(self):
        """Undo last action."""
        if self.canvas.current_points:
            removed = self.canvas.current_points.pop()
            print(f"↩ Removed point: {removed}")
        elif self.canvas.completed_polygons:
            removed = self.canvas.completed_polygons.pop()
            print(f"↩ Removed polygon with {len(removed)} points")
            
            # Clear selection if we removed the selected polygon
            if self.canvas.selected_polygon_index is not None:
                if self.canvas.selected_polygon_index >= len(self.canvas.completed_polygons):
                    self.canvas.selected_polygon_index = None
            
            # Recreate mask
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for polygon in self.canvas.completed_polygons:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(self.mask, [pts], 255)
        else:
            print("Nothing to undo")
        
        self.canvas.update_display()
        self.update_polygon_count()
    
    def delete_selected_polygon(self):
        """Delete the currently selected polygon."""
        if self.canvas.selected_polygon_index is not None:
            idx = self.canvas.selected_polygon_index
            if 0 <= idx < len(self.canvas.completed_polygons):
                removed = self.canvas.completed_polygons.pop(idx)
                print(f"❌ Deleted polygon #{idx + 1} with {len(removed)} points")
                
                # Clear selection
                self.canvas.selected_polygon_index = None
                
                # Recreate mask
                self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
                for polygon in self.canvas.completed_polygons:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(self.mask, [pts], 255)
                
                self.canvas.update_display()
                self.update_polygon_count()
        else:
            QMessageBox.information(self, "No Selection", "Please click on a polygon to select it first.")
    
    def clear_all(self):
        """Clear all polygons."""
        if self.canvas.completed_polygons or self.canvas.current_points:
            reply = QMessageBox.question(
                self, "Clear All",
                "Are you sure you want to clear all polygons?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.canvas.current_points = []
                self.canvas.completed_polygons = []
                self.canvas.selected_polygon_index = None
                self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
                self.canvas.update_display()
                self.update_polygon_count()
                print("🗑 All polygons cleared")
    
    def zoom_in(self):
        """Zoom in by 20%."""
        new_zoom = min(self.canvas.zoom_level * 1.2, self.canvas.max_zoom)
        self.canvas.zoom_level = new_zoom
        if self.canvas.zoom_level <= 1.0:
            self.canvas.pan_offset_x = 0.0
            self.canvas.pan_offset_y = 0.0
        self.canvas.update_display()
    
    def zoom_out(self):
        """Zoom out by 20%."""
        new_zoom = max(self.canvas.zoom_level / 1.2, 1.0)
        self.canvas.zoom_level = new_zoom
        if self.canvas.zoom_level <= 1.0:
            self.canvas.pan_offset_x = 0.0
            self.canvas.pan_offset_y = 0.0
        self.canvas.update_display()
    
    def reset_zoom(self):
        """Reset zoom to fit image in widget."""
        self.canvas.reset_zoom()
    
    def save_mask(self):
        """Save the current mask as JSON."""
        # Create mask filename based on original image
        mask_filename = os.path.splitext(self.current_filename)[0] + '_mask.json'
        mask_path = os.path.join(self.mask_folder, mask_filename)
        
        # Prepare JSON data
        mask_data = {
            "image_filename": self.current_filename,
            "image_folder": os.path.basename(self.current_folder),
            "image_path": os.path.join(self.current_folder, self.current_filename),
            "image_width": self.width,
            "image_height": self.height,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_polygons": len(self.canvas.completed_polygons),
            "polygons": []
        }
        
        # Convert polygons to list format
        for i, polygon in enumerate(self.canvas.completed_polygons):
            polygon_data = {
                "id": i + 1,
                "num_points": len(polygon),
                "points": [{"x": int(pt[0]), "y": int(pt[1])} for pt in polygon]
            }
            mask_data["polygons"].append(polygon_data)
        
        # Save JSON file
        try:
            with open(mask_path, 'w', encoding='utf-8') as f:
                json.dump(mask_data, f, indent=2)
            
            print(f"✓ Mask saved: {mask_path}")
            print(f"  - Polygons: {len(self.canvas.completed_polygons)}")
            print(f"  - Total points: {sum(len(p) for p in self.canvas.completed_polygons)}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save mask to:\n{mask_path}\n\nError: {str(e)}")
            print(f"❌ Error saving mask: {e}")
            return False
    
    def save_and_next(self):
        """Save current mask and move to next image."""
        num_polygons = len(self.canvas.completed_polygons)
        print(f"\n💾 Save & Next pressed...")
        print(f"  Current image index: {self.current_image_idx}")
        print(f"  Total images: {len(self.image_files)}")
        
        if self.save_mask():
            print(f"  ✓ Save successful, moving to next image...")
            # Move to next image
            if self.current_image_idx < len(self.image_files) - 1:
                self.current_image_idx += 1
                print(f"  → New image index: {self.current_image_idx}")
                self._load_current_image()
                print(f"✓ Successfully saved {num_polygons} polygon(s) and loaded next image")
            else:
                print(f"  ⚠ Already at last image")
                QMessageBox.information(self, "Complete", f"Mask saved with {num_polygons} polygon(s)!\n\nThis is the last image.")
        else:
            print(f"  ❌ Save failed, staying on current image")
            # Don't move to next if save failed
            pass
    
    def next_image(self, skip_save=False):
        """Move to next image."""
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self._load_current_image()
        else:
            QMessageBox.information(self, "Info", "Already at last image")
    
    def previous_image(self):
        """Move to previous image."""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_current_image()
        else:
            QMessageBox.information(self, "Info", "Already at first image")
    
    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts."""
        if event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_C:
                self.close_polygon()
                return True
            elif key == Qt.Key_Z:
                self.undo()
                return True
            elif key == Qt.Key_S:
                self.save_and_next()
                return True
            elif key == Qt.Key_N:
                self.next_image()
                return True
            elif key == Qt.Key_P:
                self.previous_image()
                return True
            elif key == Qt.Key_Q:
                self.close()
                return True
            elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
                self.delete_selected_polygon()
                return True
            elif key == Qt.Key_R:
                self.reset_zoom()
                return True
        
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    """Main GUI window for the segmentation labeler."""
    
    def __init__(self):
        super().__init__()
        self.selected_folders = []
        self.labeler = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Image Segmentation Labeler - GUI")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Image Segmentation Mask Labeler")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Folder selection group
        folder_group = QGroupBox("1. Select Image Folders")
        folder_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        folder_layout = QVBoxLayout()
        
        # Buttons for folder selection
        button_layout = QHBoxLayout()
        
        self.add_folder_btn = QPushButton("➕ Add Folder")
        self.add_folder_btn.setMinimumHeight(40)
        self.add_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_folder_btn.clicked.connect(self.add_folder)
        
        self.remove_folder_btn = QPushButton("➖ Remove Selected")
        self.remove_folder_btn.setMinimumHeight(40)
        self.remove_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.remove_folder_btn.clicked.connect(self.remove_folder)
        
        self.clear_folders_btn = QPushButton("🗑️ Clear All")
        self.clear_folders_btn.setMinimumHeight(40)
        self.clear_folders_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        self.clear_folders_btn.clicked.connect(self.clear_folders)
        
        button_layout.addWidget(self.add_folder_btn)
        button_layout.addWidget(self.remove_folder_btn)
        button_layout.addWidget(self.clear_folders_btn)
        
        folder_layout.addLayout(button_layout)
        
        # List widget to show selected folders
        self.folder_list = QListWidget()
        self.folder_list.setMinimumHeight(200)
        self.folder_list.setStyleSheet("""
            QListWidget {
                background-color: #f5f5f5;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 10pt;
                padding: 5px;
            }
        """)
        folder_layout.addWidget(self.folder_list)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # Masks folder selection
        mask_group = QGroupBox("2. Select Masks Folder")
        mask_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        mask_layout = QHBoxLayout()
        
        self.mask_folder_label = QLabel("Default: ./masks")
        self.mask_folder_label.setStyleSheet("font-size: 10pt; padding: 5px;")
        
        self.mask_folder_btn = QPushButton("📁 Choose Masks Folder")
        self.mask_folder_btn.setMinimumHeight(35)
        self.mask_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 10pt;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.mask_folder_btn.clicked.connect(self.select_mask_folder)
        
        mask_layout.addWidget(self.mask_folder_label, 1)
        mask_layout.addWidget(self.mask_folder_btn)
        
        mask_group.setLayout(mask_layout)
        main_layout.addWidget(mask_group)
        
        # Instructions
        instructions_group = QGroupBox("3. Instructions")
        instructions_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        instructions_layout = QVBoxLayout()
        
        instructions_text = QTextEdit()
        instructions_text.setReadOnly(True)
        instructions_text.setMaximumHeight(150)
        instructions_text.setStyleSheet("""
            QTextEdit {
                background-color: #fff9e6;
                border: 2px solid #ffd966;
                border-radius: 5px;
                font-size: 10pt;
                padding: 10px;
            }
        """)
        instructions_text.setHtml("""
            <b>How to use:</b><br>
            1. Click <b>"Add Folder"</b> to select folders containing images<br>
            2. Choose the <b>Masks Folder</b> to load existing masks (and save new ones)<br>
            3. Click <b>"Start Labeling"</b> to begin<br><br>
            <b>Labeling Controls:</b><br>
            • <b>Left Click</b> - Add point to polygon<br>
            • <b>Close Polygon button or C</b> - Close and fill polygon<br>
            • <b>Undo button or Z</b> - Undo last point/polygon<br>
            • <b>Save & Next button or S</b> - Save mask and next image<br>
            • <b>Q/ESC</b> - Quit
        """)
        
        instructions_layout.addWidget(instructions_text)
        instructions_group.setLayout(instructions_layout)
        main_layout.addWidget(instructions_group)
        
        # Start button
        self.start_btn = QPushButton("▶️ Start Labeling")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #5e35b1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_labeling)
        self.start_btn.setEnabled(False)
        main_layout.addWidget(self.start_btn)
        
        # Status label
        self.status_label = QLabel("Ready. Please select folders to begin.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 2px solid #2196F3;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
                color: #1976D2;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # Set default masks folder
        self.mask_folder = "masks"
    
    def add_folder(self):
        """Open dialog to add a folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Images",
            os.getcwd()
        )
        
        if folder:
            if folder not in self.selected_folders:
                # Check if folder has images
                has_images = self.check_folder_has_images(folder)
                if has_images:
                    self.selected_folders.append(folder)
                    self.folder_list.addItem(folder)
                    self.update_status()
                else:
                    QMessageBox.warning(
                        self,
                        "No Images Found",
                        f"No images (jpg, png, bmp) found in:\n{folder}"
                    )
            else:
                QMessageBox.information(
                    self,
                    "Already Added",
                    "This folder is already in the list."
                )
    
    def check_folder_has_images(self, folder: str) -> bool:
        """Check if folder contains any images."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        for ext in extensions:
            if glob.glob(os.path.join(folder, ext)):
                return True
        return False
    
    def remove_folder(self):
        """Remove selected folder from list."""
        current_item = self.folder_list.currentItem()
        if current_item:
            folder = current_item.text()
            self.selected_folders.remove(folder)
            self.folder_list.takeItem(self.folder_list.row(current_item))
            self.update_status()
    
    def clear_folders(self):
        """Clear all folders."""
        if self.selected_folders:
            reply = QMessageBox.question(
                self,
                "Clear All Folders",
                "Are you sure you want to clear all selected folders?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.selected_folders.clear()
                self.folder_list.clear()
                self.update_status()
    
    def select_mask_folder(self):
        """Select masks folder for loading/saving masks."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Masks Folder",
            os.getcwd()
        )
        
        if folder:
            self.mask_folder = folder
            self.mask_folder_label.setText(f"Masks: {folder}")
    
    def update_status(self):
        """Update status label and button state."""
        num_folders = len(self.selected_folders)
        
        if num_folders == 0:
            self.status_label.setText("Ready. Please select folders to begin.")
            self.start_btn.setEnabled(False)
        else:
            # Count total images
            total_images = 0
            for folder in self.selected_folders:
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
                for ext in extensions:
                    total_images += len(glob.glob(os.path.join(folder, ext)))
            
            self.status_label.setText(
                f"✓ {num_folders} folder(s) selected | {total_images} images found | Ready to start!"
            )
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 10pt;
                    color: #2E7D32;
                    font-weight: bold;
                }
            """)
            self.start_btn.setEnabled(True)
    
    def start_labeling(self):
        """Start the labeling process."""
        if not self.selected_folders:
            QMessageBox.warning(
                self,
                "No Folders Selected",
                "Please select at least one folder containing images."
            )
            return
        
        try:
            # Create labeling window
            self.labeling_window = LabelingWindow(
                image_folders=self.selected_folders,
                mask_folder=self.mask_folder
            )
            
            # Hide main window
            self.hide()
            
            # Show labeling window
            self.labeling_window.show()
            
            # Connect close event
            self.labeling_window.destroyed.connect(self.on_labeling_closed)
            
        except Exception as e:
            self.show()
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred:\n{str(e)}"
            )
    
    def on_labeling_closed(self):
        """Handle labeling window closed."""
        self.show()
        QMessageBox.information(
            self,
            "Labeling Complete",
            f"Labeling session ended.\nMasks saved to: {self.mask_folder}"
        )


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
