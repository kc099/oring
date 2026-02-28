# PatchCore Anomaly Detection for O-Ring Inspection

Implementation of [PatchCore](https://arxiv.org/abs/2106.08265) (Roth et al., CVPR 2022) for unsupervised anomaly detection on O-ring images from two production models.

## Overview

PatchCore is a **one-class** anomaly detection method:
- Trains on **good/normal images only** — no defect labels needed
- Builds a memory bank of patch-level features from a pre-trained CNN
- Uses **greedy coreset subsampling** to compress the memory bank
- At inference, scores each image by k-NN distance to the memory bank

## Data Layout

Images are 720×720 binned/cropped BMPs in `binned/`:

| Model   | Good (train)        | Defect (test)                                    |
|---------|---------------------|--------------------------------------------------|
| Model 1 | `binned/model1good` (202 imgs) | `model1defect`, `model1defect2`, `model1defect3` |
| Model 2 | `binned/good`       (260 imgs) | `notok`, `notok2`, `notok3`                      |

## Pipeline

1. **Resize** 720×720 → 660×660 → **center-crop** 640×640
2. **Feature extraction** from pre-trained ResNet-50/101 (layers 2 & 3)
3. **Coreset subsampling** (25%) to build compact memory bank
4. **k-NN scoring** (k=9) at inference for per-patch anomaly distance
5. **Gaussian smoothing** on anomaly maps for visualization

## File Structure

```
patchcore/
├── __init__.py
├── config.py              # Model configs, paths, hyperparameters
├── dataset.py             # PyTorch Dataset + DataLoader utilities
├── patchcore_model.py     # PatchCore: feature extraction, coreset, k-NN
├── train.py               # Training + evaluation entry point
├── inference.py           # CLI inference on new images
├── gui.py                 # Dear PyGui interactive inspector
├── requirements.txt
└── README.md
```

## Quick Start

### Install Dependencies

```bash
pip install -r patchcore/requirements.txt
```

### Train All Models (2 models × 2 backbones = 4 runs)

```bash
python -m patchcore.train
```

### Train Specific Combinations

```bash
# Model 1 with ResNet-50 only
python -m patchcore.train --model model1 --backbone resnet50

# Model 2 with both backbones
python -m patchcore.train --model model2

# Custom coreset ratio
python -m patchcore.train --model model1 --backbone resnet101 --coreset 0.10
```

### Inference

```bash
# Evaluate a directory with anomaly map visualizations
python -m patchcore.inference \
    --model-path patchcore/results/model1_resnet50/model1_resnet50_patchcore.pkl \
    --image-dir  binned/model1defect \
    --backbone   resnet50 \
    --visualize

# Single image with threshold
python -m patchcore.inference \
    --model-path patchcore/results/model2_resnet101/model2_resnet101_patchcore.pkl \
    --image      binned/notok/some_image.bmp \
    --backbone   resnet101 \
    --threshold  5.0
```

## Configuration

Key hyperparameters in `config.py`:

| Parameter        | Default   | Description                               |
|------------------|-----------|-------------------------------------------|
| `backbone`       | resnet50  | ResNet-50 or ResNet-101                   |
| `feature_layers` | (2, 3)    | ResNet blocks for feature extraction      |
| `coreset_ratio`  | 0.25      | Fraction of patches to keep (25%)         |
| `n_neighbors`    | 9         | k for k-NN anomaly scoring               |
| `resize`         | 660       | Resize before center crop                 |
| `center_crop`    | 640       | Final image size for backbone             |
| `batch_size`     | 8         | Feature extraction batch size (training)  |

## Output

Training produces per-model output under `patchcore/results/`:

```
patchcore/results/
├── model1_resnet50/
│   ├── model1_resnet50_patchcore.pkl    # Saved memory bank
│   └── model1_resnet50_results.json     # Evaluation metrics
├── model1_resnet101/
│   └── ...
├── model2_resnet50/
│   └── ...
└── model2_resnet101/
    └── ...
```

## Expected Behavior

- **Good images** → low anomaly scores
- **Defect images** → high anomaly scores
- **AUROC** reported when both classes are present in evaluation
- Per-image scores and statistics saved to JSON

## Hardware Requirements

- **Training**: ~4-5 GB GPU memory (coreset phase, 640×640 input)
- **Inference**: ~1.5 GB GPU memory (batch=1, 640×640 input) — fits 4.5 GB budget
- CPU works but is significantly slower
- Memory bank fits in CPU RAM (~100-300 MB depending on coreset size)

## Interactive GUI

```bash
python -m patchcore.gui
```

Features:
- **Open Image**: file dialog to select a BMP/PNG/JPG image
- **Model selector**: dropdown auto-discovers trained `.pkl` models
- **Run**: executes PatchCore and shows anomaly heatmap overlay
- **Threshold slider**: adjust OK/DEFECT boundary in real-time
- **Overlay alpha**: blend original and heatmap
- Side-by-side original + anomaly map display
