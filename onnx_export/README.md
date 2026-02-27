# Mask R-CNN ONNX Export & WPF C# Integration Guide

This folder contains everything needed to export the O-Ring defect detection Mask R-CNN model to ONNX format and run inference from a **WPF / C#** application.

---

## Contents

| File | Description |
|------|-------------|
| `export_onnx.py` | Python script that exports the trained PyTorch model to ONNX |
| `maskrcnn_oring.onnx` | Exported ONNX model (~168 MB, FP32) |
| `MaskRCNNDetector.cs` | C# wrapper around ONNX Runtime for inference |
| `InspectionService.cs` | Higher-level service: preprocess → detect → overlay → verdict |
| `MainWindow.xaml` | Example WPF UI layout |
| `MainWindow.xaml.cs` | Code-behind for the example WPF window |
| `README.md` | This guide |

---

## 1.  Exporting the Model (Python)

### Prerequisites

```
pip install torch torchvision onnx onnxruntime
```

Tested versions: torch 2.8.0, torchvision 0.23.0, onnx 1.20.0, onnxruntime 1.23.2.

### Run the export

```bash
cd "F:\standard elastomers"
python onnx_export/export_onnx.py
```

The script will:
1. Build a `maskrcnn_resnet50_fpn` model (ResNet-50 backbone, 2 classes: background + defect).
2. Load the checkpoint from `maskrcnn/dataset/combined/checkpoints/best_model.pth`.
3. Export to ONNX with **opset 17** and dynamic axes.
4. Validate the ONNX graph with `onnx.checker`.
5. Verify outputs match between PyTorch and ONNX Runtime (score diff < 0.001, box diff < 1 px).
6. Print an ORT CPU benchmark (mean latency).

Output: `onnx_export/maskrcnn_oring.onnx` (~168 MB).

### Key export settings

| Setting | Value | Reason |
|---------|-------|--------|
| Opset | 17 | Required for Mask R-CNN dynamic ops (NMS, RoI) |
| Precision | FP32 | Matches training; FP16 caused accuracy loss |
| Dynamic axes | batch, height, width, num_detections | Allows variable image sizes |

---

## 2.  C# Project Setup

### Create a new WPF project

```
dotnet new wpf -n OringInspection --framework net8.0
cd OringInspection
```

(.NET 6.0 also works — replace `net8.0` with `net6.0`.)

### Install NuGet packages

**CPU inference** (simplest):
```
dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.0
dotnet add package System.Drawing.Common --version 8.0.0
```

**GPU inference** (recommended for production — requires CUDA 12 + cuDNN 9):
```
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --version 1.18.0
dotnet add package System.Drawing.Common --version 8.0.0
```

> **Note:** On .NET 6+ (Windows), `System.Drawing.Common` works out of the box.  
> Do **not** install both CPU and GPU packages — choose one.

### Add files to the project

1. Copy all `.cs` files from this folder into your project.
2. Replace `MainWindow.xaml` and `MainWindow.xaml.cs` with the ones provided.
3. Copy `maskrcnn_oring.onnx` to the output directory:
   - In your `.csproj`, add:
     ```xml
     <ItemGroup>
       <None Update="maskrcnn_oring.onnx">
         <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
       </None>
     </ItemGroup>
     ```

---

## 3.  Model Input / Output Format

### Input

| Name | Shape | Type | Notes |
|------|-------|------|-------|
| `image` | `[1, 3, H, W]` | float32 | RGB, normalized to [0, 1] |

Preprocessing steps (must match Python pipeline):
1. **2×2 binning** — Resize to half width/height using area interpolation
2. **Pad or resize to 720×720** — The model was trained on 720×720
3. **BGR → RGB** conversion (if loading with OpenCV/System.Drawing)
4. **Scale to [0, 1]** — Divide pixel values by 255.0
5. **Channel-first** layout — HWC → CHW

### Output

| Name | Shape | Type | Notes |
|------|-------|------|-------|
| `boxes` | `[N, 4]` | float32 | Bounding boxes in `[x1, y1, x2, y2]` pixel coords |
| `labels` | `[N]` | int64 | Class labels (1 = defect) |
| `scores` | `[N]` | float32 | Confidence scores (0–1) |
| `masks` | `[N, 1, H, W]` | float32 | Instance segmentation masks (0–1 probabilities) |

Apply a **score threshold** (default 0.5) to filter low-confidence detections. Binarize masks at **0.5** to get binary masks.

---

## 4.  How the C# Code Works

### `MaskRCNNDetector.cs`

Low-level ONNX Runtime wrapper:
- Creates an `InferenceSession` (CPU or GPU)
- `BitmapToTensor()` — Converts `System.Drawing.Bitmap` to float32 NCHW tensor using `LockBits` for fast pixel access
- `Detect(Bitmap, scoreThreshold)` — Runs inference, filters by score threshold, returns `List<Detection>`

### `InspectionService.cs`

Production-ready service layer:
- `Preprocess(Bitmap)` — 2×2 bin + resize to 720×720
- `Inspect(Bitmap)` — Full pipeline: preprocess → detect → draw overlay → determine verdict
- `DrawOverlay()` — Semi-transparent red mask overlay with confidence text
- `BitmapToBitmapSource()` — Converts GDI+ `Bitmap` to WPF-compatible `BitmapSource`
- Returns `InspectionResult` with verdict, cycle time, detections, and overlay image

### `MainWindow.xaml.cs`

Example UI code-behind:
- Loads model on `Window_Loaded` (async, non-blocking)
- `LoadImage_Click` — Opens file dialog, displays raw image
- `Analyze_Click` — Runs inference on background thread, updates verdict / overlay / cycle time
- Uses `Task.Run` so the UI stays responsive during inference

---

## 5.  Adapting to Your Pipeline

### Score threshold

Adjust `scoreThreshold` in `InspectionService` constructor (default: 0.5).  
Lower values catch more defects but may increase false positives.

### Preprocessing

The `InspectionService.Preprocess()` method performs:
```
Original image → Resize(W/2, H/2, INTER_AREA) → Resize(720, 720)
```

If your camera resolution or binning changes, update this method to match the Python pipeline in `binning_pipeline/bin_and_crop.py`.

### Verdict logic

Currently binary: any detection above threshold → **REJECT**, otherwise → **PASS**.  
For REWORK logic, add measurement-based thresholds as done in the Python `inspection_gui.py`.

---

## 6.  GPU Acceleration

### Requirements

- NVIDIA GPU with CUDA Compute Capability ≥ 6.0
- CUDA Toolkit 12.x
- cuDNN 9.x
- `Microsoft.ML.OnnxRuntime.Gpu` NuGet package (instead of CPU package)

### Enable in code

```csharp
// In InspectionService constructor or startup:
_detector = new MaskRCNNDetector(modelPath, useGpu: true);
```

This sets `OrtSessionOptions.AppendExecutionProvider_CUDA(0)`.

### Expected performance

| Provider | Latency (720×720) | Notes |
|----------|-------------------|-------|
| CPU (4 threads) | ~880 ms | Default, no GPU needed |
| CUDA GPU | ~80–150 ms | Requires CUDA 12 + cuDNN |
| TensorRT | ~40–80 ms | Via `AppendExecutionProvider_Tensorrt` |

---

## 7.  Deployment Checklist

- [ ] Copy `maskrcnn_oring.onnx` alongside the executable
- [ ] Verify NuGet package matches target (CPU vs GPU)
- [ ] If GPU: install CUDA 12 + cuDNN 9 on target machine
- [ ] Test with sample images to verify detections match Python
- [ ] Set `scoreThreshold` appropriate for production (tune on validation set)
- [ ] Handle `Dispose()` — both `MaskRCNNDetector` and `InspectionService` implement `IDisposable`
- [ ] For continuous inspection, reuse the `InspectionService` instance (don't recreate per frame)

---

## 8.  Troubleshooting

| Issue | Solution |
|-------|----------|
| `OnnxRuntimeException: CUDA failure` | Install matching CUDA/cuDNN versions, or switch to CPU |
| Output scores all zero | Check preprocessing: must be RGB [0–1], channel-first |
| Detections don't match Python | Verify image is resized to 720×720 with same interpolation |
| `DllNotFoundException` for onnxruntime | Ensure the native `onnxruntime.dll` is in the output directory |
| Slow first inference | First run includes model optimization; subsequent runs are faster |
| `System.Drawing` not available | Add `<EnableWindowsTargeting>true</EnableWindowsTargeting>` to `.csproj` on non-Windows |

---

## 9.  Quick Start Summary

```
1.  pip install torch torchvision onnx onnxruntime
2.  python onnx_export/export_onnx.py        → creates maskrcnn_oring.onnx
3.  dotnet new wpf -n OringInspection
4.  dotnet add package Microsoft.ML.OnnxRuntime
5.  dotnet add package System.Drawing.Common
6.  Copy .cs/.xaml files + .onnx model into project
7.  dotnet run
8.  Click "Load Image" → "Analyze" → see verdict + overlay
```
