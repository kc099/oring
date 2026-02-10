# Mask R-CNN O-Ring Defect Detection

Instance segmentation pipeline for o-ring defect detection using **Mask R-CNN** (torchvision).
Trains a **separate model** for each o-ring type.

---

## Quick Start

```bash
cd "F:\standard elastomers\maskrcnn"

# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess — convert JSON polygon labels to COCO format + split
python preprocess_to_coco.py                 # both models
python preprocess_to_coco.py --model model1  # model1 only

# 3. Train
python train.py --model model1               # train model1
python train.py --model model2               # train model2
python train.py --model model1 --epochs 80 --batch-size 2 --lr 0.003

# 4. Monitor training
tensorboard --logdir output/model1/runs

# 5. Inference
python inference.py --model model1 --image path/to/patch.bmp
python inference.py --model model1 --folder path/to/images/
python inference.py --model model1 --full-image path/to/2448x2048.bmp
python inference.py --model model1 --evaluate   # test set metrics
```

---

## O-Ring Models

| Model | Good Folder | Defect Folder | Description |
|-------|-------------|---------------|-------------|
| **model1** | `model1good` | `model1defect` | Model 1 O-Ring |
| **model2** | `good` | `notok` | Model 2 O-Ring |

Each model is trained independently with its own checkpoint and data split.

---

## Pipeline Architecture

```
dataset/images/{folder}/*.bmp  +  dataset/labels/{folder}/*.json
                    │
                    ▼
        ┌─────────────────────┐
        │  preprocess_to_coco │  → COCO JSON + organized images
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │      train.py       │  → Mask R-CNN (ResNet50-FPN)
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │    inference.py     │  → Predictions + visualization
        └─────────────────────┘
```

### Output Structure

```
maskrcnn/output/
├── model1/
│   ├── annotations/
│   │   ├── train.json         # COCO format
│   │   ├── val.json
│   │   └── test.json
│   ├── images/
│   │   ├── train/             # 640x640 patches
│   │   ├── val/
│   │   └── test/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── latest.pth
│   ├── runs/                  # TensorBoard logs
│   ├── inference_results/
│   ├── training_history.json
│   └── dataset_summary.json
└── model2/
    └── ...
```

---

## Why Mask R-CNN Over YOLO?

| Aspect | YOLO-Seg | Mask R-CNN |
|--------|----------|------------|
| Architecture | Single-stage | Two-stage (RPN + heads) |
| Small defects | Often missed | Better via FPN multi-scale features |
| Mask quality | Coarse (prototype-based) | Per-instance high-quality masks |
| Training data | Needs many examples | Better with fewer examples (transfer learning) |
| Speed | Faster | Slower but more accurate |
| Good-image handling | Implicit | Explicit (images with no annotations in COCO) |

Your defects range from tiny holes to half-image — Mask R-CNN's **Feature Pyramid Network** handles this multi-scale challenge much better than YOLO's single-pass approach.

---

## Configuration

Edit `config.py` to tune hyperparameters:

```python
# Key parameters
batch_size = 4           # reduce to 2 if GPU memory limited
num_epochs = 50          # with early stopping patience=10
learning_rate = 0.005    # SGD with StepLR decay
trainable_backbone_layers = 3  # fine-tune last 3 ResNet layers
mixed_precision = True   # AMP for faster training
```

---

## Full-Resolution Inference

The `--full-image` mode handles the complete pipeline:

1. **Background subtraction** (Otsu) — reuses `preprocess_background_split.py`
2. **Split** into 640×640 patches with overlap
3. **Predict** on each patch independently
4. **Stitch** masks back to full 2448×2048 resolution
5. **Visualize** with overlay + bounding boxes

---

## ⚡ Alternative Approaches Worth Exploring

### 1. 🏆 Anomaly Detection (RECOMMENDED)

**Why this may work best for your case:**
- You have **~1,400+ good samples** per o-ring but only **~400 defect samples**
- Defects are diverse (holes, scratches, large deformations)
- Anomaly detection learns "what good looks like" and flags anything different

**Models to try:**

| Model | Library | Key Advantage |
|-------|---------|---------------|
| **PatchCore** | [anomalib](https://github.com/openvinotoolkit/anomalib) | State-of-art, no training needed — just a memory bank of good features |
| **EfficientAD** | anomalib | Fast, lightweight, production-ready |
| **PaDiM** | anomalib | Good for texture defects |
| **FastFlow** | anomalib | Normalizing flow — fast inference |

```bash
pip install anomalib
# PatchCore: train on ONLY good samples, zero defect labels needed!
# It extracts features from good images and flags anomalies at test time
```

**This is potentially your best option** because:
- Uses only good images (you have plenty)
- No polygon annotations needed for training
- Detects ANY type of defect, even unseen ones
- Naturally handles the "rework" vs "defect" spectrum
- State-of-the-art on industrial defect benchmarks (MVTec AD)

### 2. Binary Classification (Patch-Level)

Simplest approach: classify each 640×640 patch as **good/defect**.

- Use **ResNet-50** or **EfficientNet-B4** as backbone
- Much simpler than instance segmentation
- High accuracy for "is there a defect?" question
- Won't give you defect location/mask, but may be sufficient for pass/fail

### 3. Semantic Segmentation (U-Net)

If you need pixel-level defect masks but don't need instance separation:

- **U-Net** with ResNet/EfficientNet encoder (via `segmentation_models_pytorch`)
- Simpler than Mask R-CNN, fewer parameters
- Works well for defects that are large or blob-shaped
- Consider **DeepLabV3+** for multi-scale defects

### 4. Vision Transformer Approaches

- **Segment Anything Model (SAM)** — zero-shot segmentation, can be prompted with points/boxes
- **SegFormer** — transformer-based semantic segmentation
- May work well but require more GPU memory

### 5. Hybrid: Anomaly Detection + Mask R-CNN

Best of both worlds:
1. **PatchCore** for fast binary screening (good/defect)
2. **Mask R-CNN** only on flagged defects for detailed segmentation
3. Reduces false negatives and gives precise defect masks

---

## Recommendation Priority

Given your setup (few defect samples, diverse defect sizes, two o-ring types):

1. **🥇 Anomaly Detection (PatchCore/EfficientAD)** — try this first
2. **🥈 Mask R-CNN** (this pipeline) — for instance-level segmentation
3. **🥉 Binary Classification** — if you just need pass/fail
4. **Hybrid** — anomaly detection screening + Mask R-CNN for details

---

## Notes on Rework Samples

Rework samples are handled separately via rule-based code (not trained). This is a good approach because:
- Rework defects have measurable geometric properties
- Rule-based logic is transparent and adjustable
- Avoids contaminating the binary good/defect training signal
