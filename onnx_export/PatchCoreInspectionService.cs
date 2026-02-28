using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OringInspection
{
    /// <summary>
    /// PatchCore inspection service for O-Ring anomaly detection.
    ///
    /// Full pipeline:
    ///   1. Load/capture image (720×720 BMP from binning)
    ///   2. Preprocess: resize 660 → center-crop 640×640
    ///   3. Run PatchCore ONNX inference → anomaly score + heatmap
    ///   4. Draw heatmap overlay on the image
    ///   5. Display verdict (PASS / REJECT)
    /// </summary>
    public class PatchCoreInspectionService : IDisposable
    {
        private readonly PatchCoreDetector _detector;

        /// <summary>Resize dimension before center crop.</summary>
        public int ResizeSize { get; set; } = 660;

        /// <summary>Final crop size fed to the model.</summary>
        public int CropSize { get; set; } = 640;

        /// <summary>
        /// Create the inspection service.
        /// </summary>
        /// <param name="onnxModelPath">Path to the PatchCore ONNX model</param>
        /// <param name="threshold">Anomaly score threshold</param>
        /// <param name="useGpu">Use GPU for ONNX Runtime inference</param>
        public PatchCoreInspectionService(string onnxModelPath,
                                          float threshold = 15.0f,
                                          bool useGpu = false)
        {
            _detector = new PatchCoreDetector(onnxModelPath, threshold, useGpu);
        }

        /// <summary>
        /// Load settings from the metadata JSON produced by the export script.
        /// Sets ResizeSize, CropSize, and recommended threshold automatically.
        /// </summary>
        /// <returns>Parsed metadata dictionary, or null if file not found.</returns>
        public static PatchCoreMetadata LoadMetadata(string jsonPath)
        {
            if (!File.Exists(jsonPath)) return null;
            string json = File.ReadAllText(jsonPath);
            return JsonSerializer.Deserialize<PatchCoreMetadata>(json);
        }

        /// <summary>
        /// Full inspection pipeline: preprocess → detect → verdict.
        /// </summary>
        public PatchCoreInspectionResult Inspect(Bitmap rawImage)
        {
            var sw = Stopwatch.StartNew();

            // 1. Preprocess: resize → center crop to 640×640
            var preprocessed = Preprocess(rawImage);

            // 2. Run PatchCore inference
            var detection = _detector.Detect(preprocessed);

            sw.Stop();

            return new PatchCoreInspectionResult
            {
                AnomalyScore = detection.AnomalyScore,
                IsAnomaly = detection.IsAnomaly,
                Verdict = detection.IsAnomaly ? "REJECT" : "PASS",
                AnomalyMap = detection.AnomalyMap,
                CycleTimeMs = sw.ElapsedMilliseconds,
                PreprocessedImage = preprocessed,
                OverlayImage = detection.IsAnomaly
                    ? DrawHeatmapOverlay(preprocessed, detection.AnomalyMap, 0.5f)
                    : preprocessed
            };
        }

        /// <summary>
        /// Preprocess: resize to 660×660, center-crop to 640×640.
        /// Mirrors the Python torchvision.transforms pipeline.
        /// </summary>
        public Bitmap Preprocess(Bitmap input)
        {
            // Resize to ResizeSize × ResizeSize (bicubic interpolation)
            var resized = new Bitmap(ResizeSize, ResizeSize, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(input, 0, 0, ResizeSize, ResizeSize);
            }

            // Center-crop to CropSize × CropSize
            int margin = (ResizeSize - CropSize) / 2;
            var cropped = new Bitmap(CropSize, CropSize, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(cropped))
            {
                g.DrawImage(resized,
                    new Rectangle(0, 0, CropSize, CropSize),
                    new Rectangle(margin, margin, CropSize, CropSize),
                    GraphicsUnit.Pixel);
            }

            resized.Dispose();
            return cropped;
        }

        /// <summary>
        /// Draw a semi-transparent heatmap overlay on the image.
        /// Uses a blue → green → yellow → red colormap.
        /// </summary>
        /// <param name="image">Base image (640×640)</param>
        /// <param name="anomalyMap">Anomaly map [640×640]</param>
        /// <param name="alpha">Overlay opacity (0–1)</param>
        public static Bitmap DrawHeatmapOverlay(Bitmap image, float[,] anomalyMap, float alpha = 0.5f)
        {
            int h = image.Height;
            int w = image.Width;
            int mapH = anomalyMap.GetLength(0);
            int mapW = anomalyMap.GetLength(1);

            // Find min/max for normalization
            float min = float.MaxValue, max = float.MinValue;
            for (int y = 0; y < mapH; y++)
            {
                for (int x = 0; x < mapW; x++)
                {
                    float v = anomalyMap[y, x];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
            float range = max - min;
            if (range < 1e-6f) range = 1f;

            var overlay = new Bitmap(image);
            var rect = new Rectangle(0, 0, w, h);
            var bmpData = overlay.LockBits(rect, ImageLockMode.ReadWrite,
                                            System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            int stride = bmpData.Stride;
            byte[] pixels = new byte[stride * h];
            Marshal.Copy(bmpData.Scan0, pixels, 0, pixels.Length);

            for (int y = 0; y < h; y++)
            {
                // Map pixel y to anomaly map y
                int my = (int)((float)y / h * mapH);
                if (my >= mapH) my = mapH - 1;

                int rowOff = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int mx = (int)((float)x / w * mapW);
                    if (mx >= mapW) mx = mapW - 1;

                    float norm = (anomalyMap[my, mx] - min) / range;
                    norm = Math.Max(0, Math.Min(1, norm));

                    // Jet-like colormap: blue → cyan → green → yellow → red
                    HeatmapColor(norm, out byte hr, out byte hg, out byte hb);

                    int idx = rowOff + x * 3;
                    // Blend: (1-alpha)*original + alpha*heatmap
                    pixels[idx + 0] = (byte)(pixels[idx + 0] * (1 - alpha) + hb * alpha);  // B
                    pixels[idx + 1] = (byte)(pixels[idx + 1] * (1 - alpha) + hg * alpha);  // G
                    pixels[idx + 2] = (byte)(pixels[idx + 2] * (1 - alpha) + hr * alpha);  // R
                }
            }

            Marshal.Copy(pixels, 0, bmpData.Scan0, pixels.Length);
            overlay.UnlockBits(bmpData);
            return overlay;
        }

        /// <summary>Jet-like heatmap: 0=blue → 0.25=cyan → 0.5=green → 0.75=yellow → 1=red</summary>
        private static void HeatmapColor(float t, out byte r, out byte g, out byte b)
        {
            if (t < 0.25f)
            {
                float s = t / 0.25f;
                r = 0; g = (byte)(255 * s); b = 255;
            }
            else if (t < 0.5f)
            {
                float s = (t - 0.25f) / 0.25f;
                r = 0; g = 255; b = (byte)(255 * (1 - s));
            }
            else if (t < 0.75f)
            {
                float s = (t - 0.5f) / 0.25f;
                r = (byte)(255 * s); g = 255; b = 0;
            }
            else
            {
                float s = (t - 0.75f) / 0.25f;
                r = 255; g = (byte)(255 * (1 - s)); b = 0;
            }
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

    // ─── Data classes ───────────────────────────────────────────────────

    /// <summary>
    /// Result of a single PatchCore inspection cycle.
    /// </summary>
    public class PatchCoreInspectionResult
    {
        public float AnomalyScore { get; set; }
        public bool IsAnomaly { get; set; }
        public string Verdict { get; set; }
        public float[,] AnomalyMap { get; set; }
        public long CycleTimeMs { get; set; }
        public Bitmap PreprocessedImage { get; set; }
        public Bitmap OverlayImage { get; set; }
    }

    /// <summary>
    /// Metadata from the JSON file produced by the Python export script.
    /// </summary>
    public class PatchCoreMetadata
    {
        public string model_name { get; set; }
        public string backbone { get; set; }
        public int[] input_shape { get; set; }
        public int resize { get; set; }
        public int center_crop { get; set; }
        public int original_image_size { get; set; }
        public float? good_score_max { get; set; }
        public float? defect_score_min { get; set; }
        public float? recommended_threshold { get; set; }
        public string onnx_file { get; set; }
        public float onnx_size_mb { get; set; }
    }
}
