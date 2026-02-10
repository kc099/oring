# YOLO Segmentation Training Suite

This folder contains all scripts and configurations for training and evaluating YOLO v8 segmentation models on the defect detection dataset.

## Quick Start

### 1. Prepare Dataset (Train/Val/Test Split)
```bash
python prepare_dataset.py
```
This creates a structured dataset with:
- 70% training set
- 20% validation set
- 10% test set
- Generates `data.yaml` configuration file

**Output:** `yolo_dataset_split/` folder

### 2. Train YOLO Model
```bash
python train_yolo.py
```
Trains a YOLOv8m-seg model with default settings:
- Model: YOLOv8 Medium (yolov8m-seg)
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640

**Output:** `runs/` folder with training results and checkpoints

### 3. Run Inference & Evaluation

**Option A: GUI Interface (Recommended)**
```bash
python inference_gui.py
```
Opens a user-friendly GUI to:
- Load and visualize images
- Run inference with adjustable thresholds
- View detailed detection results
- Save annotated images

**Option B: Command-Line Evaluation**
```bash
python inference.py
```
Evaluates the trained model on the test set and generates predictions.

**Output:** `inference_results/` folder with annotated images

## File Descriptions

### `prepare_dataset.py`
Prepares the YOLO dataset for training:
- Splits data into train/val/test sets
- Creates required `data.yaml` configuration
- Validates data structure

**Key Functions:**
- `create_train_val_test_split()`: Splits dataset
- `create_data_yaml()`: Creates YOLO config file

### `train_yolo.py`
Main training script for YOLO segmentation models.

**Key Class:**
- `YOLOSegmentationTrainer`: Handles model training, validation, and inference

**Usage Example:**
```python
from train_yolo import YOLOSegmentationTrainer

trainer = YOLOSegmentationTrainer('yolo_dataset_split/data.yaml')
results = trainer.train(
    model_name='yolov8m-seg',
    epochs=100,
    batch_size=16
)
```

### `inference_gui.py`
**NEW: PySide6-based GUI for inference**

Interactive graphical interface for model inference:
- Visual model and image selection
- Real-time inference with adjustable thresholds
- Side-by-side original vs. result view
- Detailed detection statistics
- Save annotated results

**Launch:**
```bash
python inference_gui.py
```

See [GUI_README.md](GUI_README.md) for detailed GUI documentation.

### `inference.py`
Inference and evaluation script for trained models.

**Key Class:**
- `YOLOSegmentationInference`: Handles inference, evaluation, and visualization

**Key Methods:**
- `predict_image()`: Single image inference
- `predict_batch()`: Batch inference
- `evaluate()`: Model evaluation
- `visualize_prediction()`: Annotated predictions

**Usage Example:**
```python
from inference import YOLOSegmentationInference

inference = YOLOSegmentationInference('runs/.../weights/best.pt')
results = inference.predict_image('path/to/image.bmp')
```

### `config.py`
Configuration file with all hyperparameters.

**Sections:**
- `MODEL_CONFIG`: Model architecture settings
- `TRAINING_CONFIG`: Training parameters
- `AUGMENTATION_CONFIG`: Data augmentation settings
- `OPTIMIZER_CONFIG`: Optimizer settings
- `DATASET_SPLIT_CONFIG`: Dataset split ratios
- `INFERENCE_CONFIG`: Inference settings
- `PATHS_CONFIG`: Directory paths

**Usage:**
```python
from config import get_train_config, print_config

config = get_train_config()
print_config()  # Print current settings
```

## Configuration

### Model Sizes
Choose from different YOLOv8 segmentation models:
- `yolov8n-seg`: Nano (fastest, smallest)
- `yolov8s-seg`: Small
- `yolov8m-seg`: Medium (recommended)
- `yolov8l-seg`: Large
- `yolov8x-seg`: Extra Large (slowest, largest)

Edit in `config.py`:
```python
MODEL_CONFIG = {
    'model_name': 'yolov8m-seg',  # Change this
    ...
}
```

### Training Parameters
Edit `config.py` to adjust:
```python
TRAINING_CONFIG = {
    'epochs': 100,          # Number of training epochs
    'batch_size': 16,       # Batch size (reduce if GPU OOM)
    'imgsz': 640,          # Input image size
    'patience': 20,        # Early stopping patience
    'device': 0,           # GPU device (0, 1, etc. or 'cpu')
    'workers': 4,          # Data loader workers
}
```

### Data Augmentation
Adjust augmentation in `config.py`:
```python
AUGMENTATION_CONFIG = {
    'degrees': 10.0,       # Rotation
    'translate': 0.1,      # Translation
    'scale': 0.5,         # Scaling
    'fliplr': 0.5,        # Horizontal flip
    ...
}
```

### Data Split Ratios
Modify split ratios in `config.py`:
```python
DATASET_SPLIT_CONFIG = {
    'train_ratio': 0.7,    # 70% training
    'val_ratio': 0.2,      # 20% validation
    'test_ratio': 0.1,     # 10% test
}
```

## Dataset Structure

After running `prepare_dataset.py`, the structure is:

```
yolo_dataset_split/
├── images/
│   ├── train/    (70% of images)
│   ├── val/      (20% of images)
│   └── test/     (10% of images)
├── labels/
│   ├── train/    (corresponding labels)
│   ├── val/
│   └── test/
└── data.yaml     (YOLO configuration)
```

## YOLO Label Format

Each image has a corresponding `.txt` label file in YOLO format:

**Format:** `class_id x1 y1 x2 y2 ... xn yn`

**Example:**
```
0 0.5 0.3 0.7 0.5 0.8 0.6 0.7 0.8
```

Where:
- `0`: Class ID (defect)
- `0.5 0.3 ... 0.8`: Normalized polygon coordinates (0-1 range)

**Empty labels:** Images without defects have empty label files.

## Training Results

Training outputs are saved to `runs/` with the structure:

```
runs/
└── yolov8m-seg_training/
    ├── weights/
    │   ├── best.pt       (Best model - saved when validation improves)
    │   └── last.pt       (Last checkpoint - saved EVERY epoch)
    ├── results.csv       (Training metrics)
    ├── confusion_matrix.png
    ├── results.png       (Training curves)
    └── ...
```

### Checkpoint Saving

**YOLO automatically saves checkpoints:**
- **`last.pt`**: Saved after **EVERY epoch** (can resume training)
- **`best.pt`**: Saved when validation metric improves

### Stopping & Resuming Training

You can **stop training anytime** (Ctrl+C) and it will save the current state.

**To resume training:**
```python
from train_yolo import YOLOSegmentationTrainer

trainer = YOLOSegmentationTrainer('yolo_dataset_split/data.yaml')
results = trainer.train(resume=True)  # Resume from last.pt
```

Or modify `train_yolo.py` and set `resume=True` in the main function.

## Inference Results

Inference outputs are saved to `inference_results/` with:
- Annotated images with predicted segments
- Segmentation masks
- Confidence scores

## Tips & Tricks

### Optimize Batch Size
If you get CUDA out-of-memory errors:
```python
TRAINING_CONFIG = {
    'batch_size': 8,  # Reduce from 16
}
```

### Faster Training
Use smaller model:
```python
MODEL_CONFIG = {
    'model_name': 'yolov8s-seg',  # Instead of yolov8m-seg
}
```

### Better Accuracy
Use larger model:
```python
MODEL_CONFIG = {
    'model_name': 'yolov8l-seg',  # Instead of yolov8m-seg
}
```

### Transfer Learning
Start with pretrained weights and fine-tune.

### More Augmentation
Increase augmentation for small datasets:
```python
AUGMENTATION_CONFIG = {
    'degrees': 20.0,
    'translate': 0.2,
    'scale': 0.7,
}
```

## Troubleshooting

### Model Download Issues
Models are automatically downloaded on first run. Ensure internet connection is available.

### Out of Memory (OOM)
Reduce batch size in `config.py`:
```python
'batch_size': 8  # Smaller batch
```

### Slow Training on CPU
Switch to GPU by modifying `config.py`:
```python
'device': 0  # GPU device 0
```

### Data Not Found
Ensure `yolo_dataset/` exists and contains images/labels folders.

## Dependencies

Required packages:
- ultralytics (YOLO)
- torch
- torchvision
- numpy
- opencv-python (cv2)

Install with:
```bash
pip install ultralytics torch torchvision opencv-python
```

## Output Statistics

After training, key metrics include:
- **Precision**: Ratio of correct detections
- **Recall**: Ratio of detected defects
- **mAP (mean Average Precision)**: Overall accuracy
- **Segmentation IoU**: Mask overlap accuracy

These are logged during training and saved in `results.csv`.

## Next Steps

1. **Prepare dataset**: `python prepare_dataset.py`
2. **Train model**: `python train_yolo.py`
3. **Evaluate**: `python inference.py`
4. **Fine-tune**: Adjust config and retrain if needed
5. **Deploy**: Use best.pt model for production

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Model Customization](https://docs.ultralytics.com/modes/train/)
