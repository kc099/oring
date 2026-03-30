# YOLO11-seg ONNX — WPF C# Integration Guide

## Overview

This guide explains how to use the exported YOLO11-seg ONNX model for O-Ring defect segmentation in a **WPF / C#** application. The model expects 640×480 input and produces bounding boxes + segmentation masks.

---

## 1. Export the Model (Python)

```bash
cd "F:\standard elastomers\patchcore\yolo_seg"
python export_onnx.py
```

This creates a `.onnx` file (typically ~40-90 MB) in the same directory as `best.pt`.

To export to a specific path:
```bash
python export_onnx.py --output yolo11_seg_defect.onnx
```

---

## 2. Model Input / Output Format

### Input

| Name | Shape | Type | Notes |
|------|-------|------|-------|
| `images` | `[1, 3, 480, 640]` | float32 | RGB, normalized to [0, 1] |

**Preprocessing steps (must match in C#):**
1. Resize original image (2048×1536) to **640×480** using bilinear/area interpolation
2. Convert BGR → RGB (if loading with `System.Drawing`)
3. Scale pixel values to [0, 1] (divide by 255.0)
4. Reshape to CHW format: `[1, 3, 480, 640]`

### Output

YOLO11-seg ONNX exports produce **two** output tensors:

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `output0` | `[1, 38, 5040]` | float32 | Detection boxes + class scores + mask coefficients |
| `output1` | `[1, 32, 120, 160]` | float32 | Prototype masks (mask protos) |

#### Decoding `output0` (detections)

Each of the 5040 candidate detections is a vector of 38 values:
- `[0:4]` — bounding box: `cx, cy, w, h` (center-x, center-y, width, height in pixels)
- `[4]` — class confidence (1 class: defect)
- `[5:37]` — 32 mask coefficients

#### Decoding `output1` (prototype masks)

- Shape `[1, 32, 120, 160]` — 32 prototype masks at ¼ resolution (640/4=160, 480/4=120)
- To get a per-detection mask: matrix-multiply the 32 mask coefficients × 32 prototype masks, then sigmoid

---

## 3. C# Project Setup

### Create WPF project

```
dotnet new wpf -n OringYoloInspection --framework net8.0
cd OringYoloInspection
```

### Install NuGet packages

```
dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.0
dotnet add package System.Drawing.Common --version 8.0.0
```

### Add model to project

In `.csproj`:
```xml
<ItemGroup>
  <None Update="yolo11_seg_defect.onnx">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
</ItemGroup>
```

---

## 4. C# Inference Code

### `YoloSegDetector.cs`

```csharp
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OringYoloInspection
{
    public class YoloDetection
    {
        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Score { get; set; }
        public float[,] Mask { get; set; } // binary mask at model input resolution
    }

    public class YoloSegDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private const int InputW = 640;
        private const int InputH = 480;
        private const int NumClasses = 1;
        private const int MaskCoeffs = 32;
        private const int ProtoH = 120;  // InputH / 4
        private const int ProtoW = 160;  // InputW / 4
        private const float ConfThreshold = 0.25f;
        private const float NmsIouThreshold = 0.45f;

        public YoloSegDetector(string modelPath)
        {
            var opts = new SessionOptions();
            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            opts.IntraOpNumThreads = 4;
            opts.InterOpNumThreads = 4;
            _session = new InferenceSession(modelPath, opts);
        }

        /// <summary>
        /// Run inference on a Bitmap. Resizes internally to 640x480.
        /// Returns detections with masks scaled to original image size.
        /// </summary>
        public List<YoloDetection> Detect(Bitmap image, float scoreThreshold = 0.25f)
        {
            int origW = image.Width;
            int origH = image.Height;

            // Resize to model input
            using var resized = new Bitmap(image, new Size(InputW, InputH));
            var inputTensor = BitmapToTensor(resized);

            // Run inference
            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var results = _session.Run(inputs);
            var outputList = results.ToList();

            // output0: [1, 38, 5040] — detections
            var output0 = outputList[0].AsTensor<float>();
            // output1: [1, 32, 120, 160] — prototype masks
            var output1 = outputList[1].AsTensor<float>();

            // Parse detections
            int numPreds = output0.Dimensions[2]; // 5040
            int vecLen = output0.Dimensions[1];   // 38

            var candidates = new List<YoloDetection>();
            var coeffsList = new List<float[]>();

            for (int i = 0; i < numPreds; i++)
            {
                float conf = output0[0, 4, i]; // class confidence
                if (conf < scoreThreshold) continue;

                float cx = output0[0, 0, i];
                float cy = output0[0, 1, i];
                float w  = output0[0, 2, i];
                float h  = output0[0, 3, i];

                var det = new YoloDetection
                {
                    X1 = cx - w / 2f,
                    Y1 = cy - h / 2f,
                    X2 = cx + w / 2f,
                    Y2 = cy + h / 2f,
                    Score = conf,
                };
                candidates.Add(det);

                // Extract 32 mask coefficients
                float[] coeffs = new float[MaskCoeffs];
                for (int c = 0; c < MaskCoeffs; c++)
                    coeffs[c] = output0[0, 5 + NumClasses + c, i];
                coeffsList.Add(coeffs);
            }

            // NMS
            var kept = NonMaxSuppression(candidates, NmsIouThreshold);

            // Decode masks for kept detections
            foreach (int idx in kept)
            {
                var det = candidates[idx];
                float[] coeffs = coeffsList[idx];
                det.Mask = DecodeMask(output1, coeffs, det, origW, origH);
            }

            return kept.Select(i => candidates[i]).ToList();
        }

        private float[,] DecodeMask(
            Tensor<float> protos, float[] coeffs,
            YoloDetection det, int origW, int origH)
        {
            // Compute mask at proto resolution: sum(coeffs[c] * protos[0,c,h,w])
            var protoMask = new float[ProtoH, ProtoW];
            for (int c = 0; c < MaskCoeffs; c++)
            {
                float coeff = coeffs[c];
                for (int py = 0; py < ProtoH; py++)
                    for (int px = 0; px < ProtoW; px++)
                        protoMask[py, px] += coeff * protos[0, c, py, px];
            }

            // Sigmoid
            for (int py = 0; py < ProtoH; py++)
                for (int px = 0; px < ProtoW; px++)
                    protoMask[py, px] = 1f / (1f + MathF.Exp(-protoMask[py, px]));

            // Crop mask to bbox (in proto coords)
            float sx = (float)ProtoW / InputW;
            float sy = (float)ProtoH / InputH;
            int bx1 = Math.Max(0, (int)(det.X1 * sx));
            int by1 = Math.Max(0, (int)(det.Y1 * sy));
            int bx2 = Math.Min(ProtoW - 1, (int)(det.X2 * sx));
            int by2 = Math.Min(ProtoH - 1, (int)(det.Y2 * sy));

            // Resize mask to original image size (nearest-neighbor)
            var mask = new float[origH, origW];
            float xScale = (float)ProtoW / origW;
            float yScale = (float)ProtoH / origH;
            for (int y = 0; y < origH; y++)
            {
                int py = Math.Min((int)(y * yScale), ProtoH - 1);
                for (int x = 0; x < origW; x++)
                {
                    int px = Math.Min((int)(x * xScale), ProtoW - 1);
                    if (px >= bx1 && px <= bx2 && py >= by1 && py <= by2)
                        mask[y, x] = protoMask[py, px] >= 0.5f ? 1f : 0f;
                }
            }

            return mask;
        }

        private DenseTensor<float> BitmapToTensor(Bitmap bmp)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, InputH, InputW });
            var bmpData = bmp.LockBits(
                new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            int stride = bmpData.Stride;
            byte[] pixels = new byte[stride * bmp.Height];
            Marshal.Copy(bmpData.Scan0, pixels, 0, pixels.Length);
            bmp.UnlockBits(bmpData);

            for (int y = 0; y < InputH; y++)
            {
                int rowOffset = y * stride;
                for (int x = 0; x < InputW; x++)
                {
                    int idx = rowOffset + x * 3;
                    // BGR -> RGB, scale to [0, 1]
                    tensor[0, 0, y, x] = pixels[idx + 2] / 255f; // R
                    tensor[0, 1, y, x] = pixels[idx + 1] / 255f; // G
                    tensor[0, 2, y, x] = pixels[idx + 0] / 255f; // B
                }
            }

            return tensor;
        }

        private static List<int> NonMaxSuppression(
            List<YoloDetection> dets, float iouThreshold)
        {
            var indices = Enumerable.Range(0, dets.Count)
                .OrderByDescending(i => dets[i].Score).ToList();
            var kept = new List<int>();

            while (indices.Count > 0)
            {
                int best = indices[0];
                kept.Add(best);
                indices.RemoveAt(0);

                indices.RemoveAll(i =>
                    IoU(dets[best], dets[i]) > iouThreshold);
            }

            return kept;
        }

        private static float IoU(YoloDetection a, YoloDetection b)
        {
            float x1 = Math.Max(a.X1, b.X1);
            float y1 = Math.Max(a.Y1, b.Y1);
            float x2 = Math.Min(a.X2, b.X2);
            float y2 = Math.Min(a.Y2, b.Y2);
            float inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
            float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);
            return inter / (areaA + areaB - inter + 1e-6f);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
```

---

### `MainWindow.xaml`

```xml
<Window x:Class="OringYoloInspection.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="O-Ring YOLO Defect Inspector" Height="800" Width="1100"
        Loaded="Window_Loaded">
    <Grid Margin="10">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <!-- Toolbar -->
        <StackPanel Grid.Row="0" Orientation="Horizontal" Margin="0,0,0,10">
            <Button x:Name="BtnLoad" Content="Load Image"
                    Padding="15,8" Margin="0,0,10,0"
                    Click="LoadImage_Click"/>
            <Button x:Name="BtnAnalyze" Content="Analyze"
                    Padding="15,8" Margin="0,0,10,0"
                    IsEnabled="False"
                    Click="Analyze_Click"/>
            <TextBlock x:Name="TxtModelStatus" VerticalAlignment="Center"
                       Foreground="Gray" Text="Loading model..."/>
        </StackPanel>

        <!-- Image display -->
        <Border Grid.Row="1" BorderBrush="Gray" BorderThickness="1">
            <Image x:Name="ImgDisplay" Stretch="Uniform"/>
        </Border>

        <!-- Status bar -->
        <StackPanel Grid.Row="2" Orientation="Horizontal" Margin="0,10,0,0">
            <TextBlock x:Name="TxtVerdict" FontSize="18" FontWeight="Bold"
                       VerticalAlignment="Center" Margin="0,0,20,0"/>
            <TextBlock x:Name="TxtInfo" VerticalAlignment="Center"
                       Foreground="Gray"/>
        </StackPanel>
    </Grid>
</Window>
```

---

### `MainWindow.xaml.cs`

```csharp
using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace OringYoloInspection
{
    public partial class MainWindow : Window
    {
        private YoloSegDetector _detector;
        private Bitmap _currentImage;
        private const string ModelPath = "yolo11_seg_defect.onnx";

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // Load model on background thread
            await Task.Run(() =>
            {
                _detector = new YoloSegDetector(ModelPath);
            });

            TxtModelStatus.Text = "Model loaded (CPU)";
            TxtModelStatus.Foreground = System.Windows.Media.Brushes.Green;
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Images|*.bmp;*.png;*.jpg;*.jpeg;*.tiff"
            };
            if (dlg.ShowDialog() != true) return;

            _currentImage?.Dispose();
            _currentImage = new Bitmap(dlg.FileName);
            ImgDisplay.Source = BitmapToBitmapSource(_currentImage);
            BtnAnalyze.IsEnabled = true;
            TxtVerdict.Text = "";
            TxtInfo.Text = $"{_currentImage.Width}×{_currentImage.Height}";
        }

        private async void Analyze_Click(object sender, RoutedEventArgs e)
        {
            if (_currentImage == null || _detector == null) return;

            BtnAnalyze.IsEnabled = false;
            TxtVerdict.Text = "Analyzing...";

            var sw = Stopwatch.StartNew();
            var detections = await Task.Run(() =>
                _detector.Detect(_currentImage, scoreThreshold: 0.25f));
            sw.Stop();

            if (detections.Count > 0)
            {
                // Draw overlay
                using var overlay = new Bitmap(_currentImage);
                using var g = Graphics.FromImage(overlay);
                foreach (var det in detections)
                {
                    // Scale bbox from model coords (640x480) to original
                    float sx = (float)_currentImage.Width / 640f;
                    float sy = (float)_currentImage.Height / 480f;
                    int bx = (int)(det.X1 * sx);
                    int by = (int)(det.Y1 * sy);
                    int bw = (int)((det.X2 - det.X1) * sx);
                    int bh = (int)((det.Y2 - det.Y1) * sy);

                    // Draw mask overlay (semi-transparent red)
                    if (det.Mask != null)
                    {
                        for (int y = 0; y < _currentImage.Height; y++)
                            for (int x = 0; x < _currentImage.Width; x++)
                                if (det.Mask[y, x] > 0.5f)
                                    overlay.SetPixel(x, y,
                                        Color.FromArgb(90, 255, 0, 0));
                    }

                    // Draw bbox
                    using var pen = new Pen(Color.Red, 3);
                    g.DrawRectangle(pen, bx, by, bw, bh);
                    g.DrawString($"defect {det.Score:F2}",
                        new Font("Segoe UI", 14, System.Drawing.FontStyle.Bold),
                        System.Drawing.Brushes.Red, bx, by - 25);
                }

                ImgDisplay.Source = BitmapToBitmapSource(overlay);
                TxtVerdict.Text = "DEFECT DETECTED";
                TxtVerdict.Foreground = System.Windows.Media.Brushes.Red;
            }
            else
            {
                TxtVerdict.Text = "OK";
                TxtVerdict.Foreground = System.Windows.Media.Brushes.Green;
            }

            TxtInfo.Text = $"Detections: {detections.Count} | "
                         + $"Time: {sw.ElapsedMilliseconds} ms | "
                         + $"{_currentImage.Width}×{_currentImage.Height}";
            BtnAnalyze.IsEnabled = true;
        }

        private static BitmapSource BitmapToBitmapSource(Bitmap bmp)
        {
            using var ms = new MemoryStream();
            bmp.Save(ms, ImageFormat.Png);
            ms.Position = 0;
            var img = new BitmapImage();
            img.BeginInit();
            img.CacheOption = BitmapCacheOption.OnLoad;
            img.StreamSource = ms;
            img.EndInit();
            img.Freeze();
            return img;
        }

        protected override void OnClosed(EventArgs e)
        {
            _detector?.Dispose();
            _currentImage?.Dispose();
            base.OnClosed(e);
        }
    }
}
```

---

## 5. Preprocessing Must Match Training

The model was trained on images resized from **2048×1536 → 640×480**. In C#:

```csharp
// Resize to model input
using var resized = new Bitmap(originalImage, new Size(640, 480));
```

Then convert to float32 tensor `[1, 3, 480, 640]` with RGB channel order, values in `[0, 1]`.

---

## 6. Output Decoding Summary

```
1. Parse output0 [1, 38, 5040]:
   - For each of 5040 candidates, check conf (index 4) > threshold
   - Extract bbox (indices 0-3): cx, cy, w, h → convert to x1, y1, x2, y2
   - Extract 32 mask coefficients (indices 5-36)

2. Apply NMS to filter overlapping boxes (IoU threshold 0.45)

3. Decode masks using output1 [1, 32, 120, 160]:
   - For each kept detection: mask = sigmoid(coeffs · protos)
   - Crop mask to bounding box region
   - Resize mask to original image resolution
   - Threshold at 0.5 for binary mask
```

---

## 7. Quick Start

```
1.  Train model:       python train.py
2.  Export to ONNX:    python export_onnx.py --output yolo11_seg_defect.onnx
3.  Create WPF app:    dotnet new wpf -n OringYoloInspection
4.  Add NuGet:         dotnet add package Microsoft.ML.OnnxRuntime
5.  Add NuGet:         dotnet add package System.Drawing.Common
6.  Copy C# code from this README into your project
7.  Copy yolo11_seg_defect.onnx to project root (set CopyToOutput)
8.  dotnet run
9.  Load Image → Analyze → see defect overlay
```

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| Wrong output shape | Run `export_onnx.py` to see actual shapes; adjust constants in `YoloSegDetector.cs` |
| No detections | Lower `ConfThreshold` to 0.1 for testing |
| Slow mask overlay | Use `LockBits` instead of `SetPixel` for production |
| Colors wrong | Ensure BGR→RGB conversion in `BitmapToTensor` |
| Model file not found | Set `CopyToOutputDirectory` in `.csproj` |
| `DllNotFoundException` | Ensure `onnxruntime.dll` is in output directory |

---

## 9. Notes

- The C# code uses **CPU-only** inference (`Microsoft.ML.OnnxRuntime`, not `.Gpu`).
- Expected CPU inference time: ~200-500 ms depending on hardware.
- The `SetPixel` mask overlay loop is slow for large images — for production, use `LockBits` with pointer access.
- The output tensor dimensions (38, 5040, 32, 120, 160) depend on the model architecture and image size. If you change `--imgsz`, re-check the actual ONNX output shapes and update the constants.
