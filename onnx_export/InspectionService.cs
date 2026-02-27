using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OringInspection
{
    /// <summary>
    /// Example WPF integration showing how to use MaskRCNNDetector
    /// in a WPF application for o-ring defect inspection.
    /// 
    /// This demonstrates the full pipeline:
    ///   1. Load/capture image
    ///   2. Preprocess (bin + crop to 720×720)
    ///   3. Run Mask R-CNN inference via ONNX Runtime
    ///   4. Draw defect overlay on the image
    ///   5. Display verdict (PASS/REWORK/REJECT)
    /// </summary>
    public class InspectionService : IDisposable
    {
        private readonly MaskRCNNDetector _detector;
        private readonly float _scoreThreshold;

        public InspectionService(string onnxModelPath, float scoreThreshold = 0.5f)
        {
            _detector = new MaskRCNNDetector(onnxModelPath, useGpu: false);
            _scoreThreshold = scoreThreshold;
        }

        /// <summary>
        /// Full inspection pipeline: preprocess → detect → verdict.
        /// </summary>
        public InspectionResult Inspect(Bitmap rawImage)
        {
            var sw = Stopwatch.StartNew();

            // 1. Preprocess: 2×2 binning + crop to 720×720
            var preprocessed = Preprocess(rawImage);

            // 2. Run Mask R-CNN
            var detections = _detector.Detect(preprocessed, _scoreThreshold);

            sw.Stop();

            // 3. Build result
            bool hasDefect = detections.Count > 0;
            float topScore = detections.Count > 0
                ? detections.Max(d => d.Score)
                : 0f;

            return new InspectionResult
            {
                HasDefect = hasDefect,
                Detections = detections,
                Verdict = hasDefect ? "REJECT" : "PASS",
                TopScore = topScore,
                CycleTimeMs = sw.ElapsedMilliseconds,
                OverlayImage = hasDefect
                    ? DrawOverlay(preprocessed, detections)
                    : preprocessed
            };
        }

        /// <summary>
        /// Simulate 2×2 binning (halve resolution) + crop to foreground + 
        /// resize/pad to 720×720. This mirrors the Python bin_crop_720().
        /// </summary>
        private Bitmap Preprocess(Bitmap input)
        {
            // 2×2 binning via resize with high-quality interpolation
            int halfW = input.Width / 2;
            int halfH = input.Height / 2;

            var binned = new Bitmap(halfW, halfH, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(binned))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
                g.DrawImage(input, 0, 0, halfW, halfH);
            }

            // For production: implement foreground crop (find o-ring bounding box)
            // Simplified: resize to fit 720×720 maintaining aspect ratio, pad to 720×720
            float scale = Math.Min(720f / binned.Width, 720f / binned.Height);
            int newW = (int)(binned.Width * scale);
            int newH = (int)(binned.Height * scale);

            var canvas = new Bitmap(720, 720, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(canvas))
            {
                // Fill with background color (dark gray ≈ 20)
                g.Clear(System.Drawing.Color.FromArgb(20, 20, 20));
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;

                int offsetX = (720 - newW) / 2;
                int offsetY = (720 - newH) / 2;
                g.DrawImage(binned, offsetX, offsetY, newW, newH);
            }

            binned.Dispose();
            return canvas;
        }

        /// <summary>
        /// Draw red defect mask overlay on the image.
        /// </summary>
        private Bitmap DrawOverlay(Bitmap image, List<Detection> detections)
        {
            var overlay = new Bitmap(image);

            using (var g = Graphics.FromImage(overlay))
            {
                foreach (var det in detections)
                {
                    int maskH = det.Mask.GetLength(0);
                    int maskW = det.Mask.GetLength(1);

                    // Draw semi-transparent red mask
                    for (int y = 0; y < Math.Min(maskH, overlay.Height); y++)
                    {
                        for (int x = 0; x < Math.Min(maskW, overlay.Width); x++)
                        {
                            if (det.Mask[y, x] > 0.5f)
                            {
                                var original = overlay.GetPixel(x, y);
                                // Blend: 60% original + 40% red
                                int r = (int)(original.R * 0.6 + 255 * 0.4);
                                int gr = (int)(original.G * 0.6);
                                int b = (int)(original.B * 0.6);
                                overlay.SetPixel(x, y,
                                    System.Drawing.Color.FromArgb(
                                        Math.Min(r, 255),
                                        Math.Min(gr, 255),
                                        Math.Min(b, 255)));
                            }
                        }
                    }

                    // Draw confidence label
                    string label = $"{det.Score:P0}";
                    float cx = (det.X1 + det.X2) / 2;
                    float cy = (det.Y1 + det.Y2) / 2;
                    g.DrawString(label,
                        new Font("Segoe UI", 14, System.Drawing.FontStyle.Bold),
                        System.Drawing.Brushes.White,
                        cx - 20, cy - 10);
                }
            }

            return overlay;
        }

        /// <summary>
        /// Convert System.Drawing.Bitmap to WPF BitmapSource for display.
        /// </summary>
        public static BitmapSource BitmapToBitmapSource(Bitmap bmp)
        {
            var bmpData = bmp.LockBits(
                new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            var source = BitmapSource.Create(
                bmp.Width, bmp.Height,
                96, 96,
                PixelFormats.Bgr24,
                null,
                bmpData.Scan0,
                bmpData.Stride * bmp.Height,
                bmpData.Stride);

            bmp.UnlockBits(bmpData);
            source.Freeze();  // Make cross-thread accessible
            return source;
        }

        public void Dispose()
        {
            _detector?.Dispose();
        }
    }

    /// <summary>
    /// Result of a single inspection cycle.
    /// </summary>
    public class InspectionResult
    {
        public bool HasDefect { get; set; }
        public List<Detection> Detections { get; set; }
        public string Verdict { get; set; }
        public float TopScore { get; set; }
        public long CycleTimeMs { get; set; }
        public Bitmap OverlayImage { get; set; }
    }
}
