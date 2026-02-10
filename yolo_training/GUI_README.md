# YOLO Segmentation Inference GUI

## Overview
A user-friendly PySide6-based GUI application for running YOLO segmentation inference on defect images.

## Features
✓ **Visual Interface**: Easy-to-use graphical interface
✓ **Model Selection**: Load any trained YOLO model (.pt files)
✓ **Image Upload**: Browse and load images for inference
✓ **Real-time Inference**: Run segmentation with live results
✓ **Adjustable Thresholds**: Confidence and IOU sliders
✓ **Side-by-Side View**: Original image vs. segmented result
✓ **Detailed Results**: Shows defect count, confidence, bounding boxes, and area
✓ **Save Results**: Export annotated images
✓ **Auto-Discovery**: Automatically finds trained models in runs/ folder

## Installation

Install PySide6 if not already installed:
```bash
pip install PySide6
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Launch the GUI
```bash
python inference_gui.py
```

### Workflow

1. **Select Model**
   - The GUI automatically finds models in `runs/` folder
   - Or click "Browse Model..." to load a custom .pt file
   - Best models are typically in `runs/*/weights/best.pt`

2. **Load Image**
   - Click "Load Image" button
   - Browse and select an image (BMP, PNG, JPG, etc.)
   - Original image displays on the left

3. **Adjust Thresholds (Optional)**
   - **Confidence**: Minimum confidence to show detections (default: 0.50)
   - **IOU Threshold**: Overlap threshold for non-max suppression (default: 0.45)

4. **Run Inference**
   - Click "Run Inference" button
   - Wait for processing (typically 1-2 seconds)
   - Segmented result displays on the right

5. **View Results**
   - Number of defects detected
   - Confidence scores for each defect
   - Bounding box coordinates
   - Defect area in pixels
   - Total defect area

6. **Save Results**
   - Click "Save Result" to export annotated image
   - Choose output location and format (PNG/JPG)

7. **Clear**
   - Click "Clear" to reset and load a new image

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│          YOLO Segmentation - Defect Detection               │
├─────────────────────────────────────────────────────────────┤
│ Controls                                                     │
│ Model: [dropdown]                   [Browse Model...]       │
│ [Load Image] [Run Inference] [Save Result] [Clear]         │
│ Confidence: [slider]  0.50    IOU: [slider]  0.45          │
├──────────────────────┬──────────────────────────────────────┤
│   Original Image     │    Segmentation Result               │
│                      │                                       │
│  (loaded image)      │  (inference result with masks)       │
│                      │                                       │
├──────────────────────┴──────────────────────────────────────┤
│ Detection Information                                        │
│ - Number of defects: X                                      │
│ - Defect details (confidence, bbox, area)                   │
│ - Total defect area                                         │
└─────────────────────────────────────────────────────────────┘
```

## Confidence Threshold

The confidence threshold determines which detections are displayed:
- **Lower** (0.1-0.3): Show more detections, but may include false positives
- **Medium** (0.4-0.6): Balanced (recommended default: 0.50)
- **Higher** (0.7-0.9): Only show high-confidence detections

Adjust the slider to find the optimal threshold for your use case.

## IOU Threshold

The IOU (Intersection over Union) threshold controls non-maximum suppression:
- **Lower** (0.1-0.3): More aggressive suppression, fewer overlapping boxes
- **Medium** (0.4-0.5): Balanced (recommended default: 0.45)
- **Higher** (0.6-0.9): Less suppression, may show multiple boxes for same defect

## Model Auto-Discovery

The GUI automatically searches for trained models in:
- `runs/*/weights/best.pt`
- `runs/*/weights/last.pt`

Models are listed as: `{training_name}/{model_file}`

Example: `yolov8m-seg_training/best.pt`

## Keyboard Shortcuts

- **Ctrl+O**: Load image (same as "Load Image" button)
- **Ctrl+R**: Run inference (same as "Run Inference" button)
- **Ctrl+S**: Save result (same as "Save Result" button)
- **Ctrl+Q**: Quit application

## Result Information

The detection information panel shows:

**For images with defects:**
```
Number of defects detected: 2

Defect Details:
  Defect #1:
    Confidence: 0.856 (85.6%)
    Bounding Box: (120, 200) to (350, 450)
    Area: 57500 pixels²
  
  Defect #2:
    Confidence: 0.723 (72.3%)
    Bounding Box: (400, 100) to (550, 300)
    Area: 30000 pixels²

Total defect area: 87500 pixels
```

**For defect-free images:**
```
✓ No defects detected

The image appears to be defect-free!
```

## Tips

### Best Performance
- Use GPU if available (model loads faster)
- Use `best.pt` for better accuracy
- Use `last.pt` if `best.pt` is not available

### Adjusting Detection Sensitivity
- **Too many false positives**: Increase confidence threshold
- **Missing defects**: Decrease confidence threshold
- **Multiple boxes on same defect**: Decrease IOU threshold
- **Split defects**: Increase IOU threshold

### Batch Processing
For processing multiple images, use the command-line inference script:
```bash
python inference.py
```

## Troubleshooting

### "No trained models found"
- Run training first: `python train_yolo.py`
- Or manually select a model using "Browse Model..."

### GUI doesn't start
```bash
pip install --upgrade PySide6
```

### Inference is slow
- Check if GPU is being used
- Use a smaller model (yolov8n-seg or yolov8s-seg)
- Reduce image size in preprocessing

### Model fails to load
- Ensure the .pt file is a valid YOLO segmentation model
- Check that ultralytics package is installed
- Verify model was trained with compatible YOLO version

## Advanced Usage

### Loading Custom Models

You can load any YOLO segmentation model:
1. Click "Browse Model..."
2. Navigate to your .pt file
3. Select the model

Supported models:
- YOLOv8-seg (all sizes: n, s, m, l, x)
- Custom trained models
- Fine-tuned models

### Integration with Other Code

The GUI can be imported and used programmatically:

```python
from inference_gui import YOLOSegmentationGUI
from PySide6.QtWidgets import QApplication

app = QApplication([])
gui = YOLOSegmentationGUI()
gui.show()
app.exec()
```

## Screenshot Description

The GUI consists of:
- **Top Section**: Control panel with model selection, buttons, and threshold sliders
- **Middle Section**: Side-by-side image display (original | result)
- **Bottom Section**: Detection information panel with detailed results
- **Status Bar**: Shows current operation status and progress

## Example Workflow

1. Launch: `python inference_gui.py`
2. GUI opens with models auto-detected
3. Click "Load Image", select `test_image.bmp`
4. Adjust confidence to 0.6 if needed
5. Click "Run Inference"
6. View segmentation results on the right
7. Check detection details in info panel
8. Click "Save Result" if satisfied
9. Load next image or adjust thresholds

## Dependencies

- PySide6: GUI framework
- ultralytics: YOLO framework
- opencv-python: Image processing
- numpy: Array operations
- PyTorch: Deep learning backend

All dependencies are in `requirements.txt`.

## Notes

- The GUI runs inference in a background thread to keep the interface responsive
- Large images are automatically scaled to fit the display
- Original image aspect ratio is preserved
- Segmentation masks are overlaid with transparency
- Bounding boxes show confidence scores
- Different colors indicate different defect instances

## Future Enhancements

Potential features for future versions:
- Batch processing mode
- Export results to CSV/JSON
- Real-time webcam inference
- Model comparison view
- Custom color schemes for masks
- Zoom and pan for large images
- Annotation editing tools
