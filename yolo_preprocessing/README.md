# YOLO Segmentation Dataset Preprocessing

## Overview
This script converts JSON-formatted masks from the split_dataset into YOLO segmentation format for training a YOLO segmentation model.

## What It Does
1. **Reads JSON masks** from `split_dataset/masks/model1defect_masks/` and `split_dataset/masks/notok_masks/`
2. **Converts polygon coordinates** to normalized YOLO format (0-1 range)
3. **Creates label files** in YOLO format (.txt files with normalized coordinates)
4. **Handles images without segments** by creating empty label files
5. **Copies images** to a new organized directory structure

## Input Structure
```
split_dataset/
├── model1defect/          (images)
├── notok/                  (images)
└── masks/
    ├── model1defect_masks/  (JSON masks)
    └── notok_masks/         (JSON masks)
```

## Output Structure
```
yolo_dataset/
├── images/   (all images from model1defect and notok)
└── labels/   (corresponding YOLO format .txt files)
```

## YOLO Format
Each label file contains polygon segmentation in the format:
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id` = 0 (defect)
- `x1, y1, x2, y2, ...` = normalized coordinates in range [0, 1]

For images without segments, empty label files are created.

## JSON Mask Format (Input)
```json
{
  "image_filename": "Image_XXXXX_patch_004.bmp",
  "image_width": 640,
  "image_height": 640,
  "polygons": [
    {
      "id": 1,
      "num_points": 11,
      "points": [
        {"x": 412, "y": 288},
        {"x": 470, "y": 210},
        ...
      ]
    }
  ]
}
```

## Usage
```bash
python convert_masks_to_yolo.py
```

## Output
The script will:
- Create `yolo_dataset/images/` with all images (model1defect + notok)
- Create `yolo_dataset/labels/` with corresponding YOLO format label files
- Print statistics about processing:
  - Total images processed
  - Images with segments (converted labels)
  - Images without segments (empty labels)
  - Any errors encountered

## Notes
- All coordinates are normalized to [0, 1] range as required by YOLO
- Images without mask files or with empty polygons get empty label files
- The script handles missing masks gracefully
- All images are copied to ensure the dataset is self-contained

## YOLO Dataset Statistics
After running the script, the dataset will be ready for YOLO segmentation training with:
- **Class 0**: Defects (from both model1defect and notok folders)
- **Total images**: ~2226 (1041 from model1defect + 1185 from notok)
