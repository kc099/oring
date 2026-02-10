# Image and Mask Splitting for YOLOv8 Segmentation

This project contains scripts to split large (2kx2k) images and their corresponding segmentation masks into smaller patches suitable for YOLOv8 training.

## Overview

The dataset contains O-ring images with three classes:
- **Good**: No defects (no masks)
- **Rework**: Minor defects requiring rework
- **Bad/Defect**: Major defects

Original images are 2448x2048 pixels, which are too large for YOLOv8. These scripts split them into 640x640 patches (standard YOLOv8 input size).

## Scripts

### 1. split_images_and_masks.py
Splits large images and transforms mask polygon coordinates for each patch.

**Features:**
- Splits images into configurable patch sizes (default 640x640)
- Transforms JSON mask polygons to patch-local coordinates
- Handles images without masks (good samples)
- Preserves directory structure
- Optional overlap between patches

**Usage:**
```python
python split_images_and_masks.py
```

**Configuration** (edit in script):
```python
SOURCE_ROOT = r"F:\standard elastomers"
OUTPUT_ROOT = r"F:\standard elastomers\split_dataset"
PATCH_SIZE = 640  # YOLOv8 standard size
OVERLAP = 0       # Increase for better boundary detection
```

### 2. visualize_split_results.py
Interactive visualization tool to verify splitting results.

**Features:**
- Summary report of split dataset
- Visualize random samples from each folder
- View all patches from a specific original image
- Display masks overlaid on images
- Interactive menu system

**Usage:**
```python
python visualize_split_results.py
```

## Workflow

### Step 1: Install Requirements
```bash
pip install opencv-python numpy matplotlib
```

### Step 2: Split Images and Masks
```bash
python split_images_and_masks.py
```

This will create a new `split_dataset` directory with:
```
split_dataset/
├── model1defect/
│   ├── Image_20260130144046195_patch_000.bmp
│   ├── Image_20260130144046195_patch_001.bmp
│   └── ...
├── model1rework/
├── model1good/
├── notok/
├── Rework/
├── good/
└── masks/
    ├── model1defect_masks/
    │   ├── Image_20260130144046195_patch_000_mask.json
    │   ├── Image_20260130144046195_patch_001_mask.json
    │   └── ...
    ├── model1rework_masks/
    ├── notok_masks/
    └── Rework_masks/
```

### Step 3: Visualize Results
```bash
python visualize_split_results.py
```

The visualization tool provides:
- **Option 1**: Random sample visualization (6 patches with masks)
- **Option 2**: Grid view of all patches from one original image

### Step 4: Convert to YOLOv8 Format
After verifying the splits, you'll need to:
1. Convert JSON masks to YOLOv8 segmentation format (normalized polygon coordinates)
2. Create train/val/test splits
3. Generate YOLOv8 dataset.yaml

## Output Structure

### Split Images
- Each original image is split into multiple patches
- Patches are named: `{original_name}_patch_{index:03d}.ext`
- Patches maintain original image format (BMP, JPG, PNG)

### Split Masks
Each mask JSON contains:
```json
{
  "image_filename": "Image_20260130144046195_patch_000.bmp",
  "image_folder": "model1defect",
  "image_width": 640,
  "image_height": 640,
  "patch_info": {
    "original_image": "Image_20260130144046195.bmp",
    "patch_index": 0,
    "x_start": 0,
    "y_start": 0,
    "x_end": 640,
    "y_end": 640
  },
  "num_polygons": 1,
  "polygons": [
    {
      "id": 1,
      "num_points": 11,
      "points": [
        {"x": 412, "y": 928},
        {"x": 470, "y": 850},
        ...
      ]
    }
  ]
}
```

## Key Features

### Intelligent Patch Creation
- Automatically calculates optimal number of patches
- Adjusts edge patches to maintain consistent size
- Configurable overlap for better boundary detection

### Mask Transformation
- Transforms polygon coordinates to patch-local space
- Filters out polygons that don't intersect with patch
- Preserves polygon structure and IDs
- Stores original image metadata for reference

### Good Images Handling
- Processes images without masks (good samples)
- Maintains consistent directory structure
- No mask files created for good images

## Next Steps for YOLOv8

To use this dataset with YOLOv8, you'll need to:

1. **Convert JSON to YOLO format**:
   - Create `.txt` files with normalized polygon coordinates
   - Format: `class_id x1 y1 x2 y2 x3 y3 ...` (normalized 0-1)

2. **Organize dataset**:
   ```
   yolo_dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

3. **Create dataset.yaml**:
   ```yaml
   path: /path/to/yolo_dataset
   train: images/train
   val: images/val
   test: images/test
   
   nc: 2  # number of classes (rework, bad)
   names: ['rework', 'bad']
   ```

4. **Train YOLOv8**:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n-seg.pt')
   model.train(data='dataset.yaml', epochs=100, imgsz=640)
   ```

## Troubleshooting

### Issue: No masks found
- Check that mask files are in `masks/{folder}_masks/` directory
- Verify mask filenames match image filenames with `_mask.json` suffix

### Issue: Polygons not visible in patches
- Increase `OVERLAP` parameter to capture boundary objects
- Verify original mask coordinates are correct

### Issue: Too many/few patches
- Adjust `PATCH_SIZE` parameter
- Larger patches = fewer patches, more context
- Smaller patches = more patches, better for small objects

## Parameters to Tune

```python
PATCH_SIZE = 640   # Increase for fewer, larger patches (416, 640, 800)
OVERLAP = 0        # Add overlap for boundary objects (50, 100, 150)
```

## License

MIT
