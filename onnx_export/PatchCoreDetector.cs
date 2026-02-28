using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace OringInspection
{
    /// <summary>
    /// Result from a single PatchCore anomaly detection inference pass.
    /// </summary>
    public class PatchCoreResult
    {
        /// <summary>Image-level anomaly score (higher = more anomalous).</summary>
        public float AnomalyScore { get; set; }

        /// <summary>Spatial anomaly heatmap [640×640], values ≥ 0.</summary>
        public float[,] AnomalyMap { get; set; }

        /// <summary>True if AnomalyScore exceeds the threshold.</summary>
        public bool IsAnomaly { get; set; }
    }

    /// <summary>
    /// PatchCore ONNX inference wrapper for O-Ring anomaly detection.
    ///
    /// The ONNX model is self-contained: ResNet backbone + memory bank + k-NN scoring.
    /// Input:  [1, 3, 640, 640] float32, RGB, [0-1] scaled
    /// Output: anomaly_score [1], anomaly_map [1, 1, 640, 640]
    ///
    /// Usage:
    ///   var detector = new PatchCoreDetector("patchcore_model1_resnet50.onnx", threshold: 15.0f);
    ///   var result = detector.Detect(preprocessedBitmap);
    ///   detector.Dispose();
    ///
    /// NuGet packages required:
    ///   - Microsoft.ML.OnnxRuntime        (CPU)
    ///   - Microsoft.ML.OnnxRuntime.Gpu     (GPU — optional, for CUDA)
    /// </summary>
    public class PatchCoreDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly float _threshold;

        /// <summary>Expected input spatial size (width and height).</summary>
        public int InputSize { get; } = 640;

        /// <summary>
        /// Initialize the PatchCore detector with an ONNX model.
        /// </summary>
        /// <param name="modelPath">Path to patchcore_*.onnx</param>
        /// <param name="threshold">Anomaly score threshold (scores above → anomaly)</param>
        /// <param name="useGpu">Use CUDA GPU if available</param>
        public PatchCoreDetector(string modelPath, float threshold = 15.0f, bool useGpu = false)
        {
            _threshold = threshold;

            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads = 4;
            options.IntraOpNumThreads = 4;

            if (useGpu)
            {
                // Requires Microsoft.ML.OnnxRuntime.Gpu NuGet package
                // and CUDA 12.x / cuDNN 9.x installed
                options.AppendExecutionProvider_CUDA(0);
            }

            _session = new InferenceSession(modelPath, options);
            _inputName = _session.InputMetadata.Keys.First();  // "image"
        }

        /// <summary>
        /// Run anomaly detection on a preprocessed image.
        /// The image must be 640×640 (already resized and center-cropped).
        /// </summary>
        /// <param name="image">Preprocessed 640×640 RGB image</param>
        /// <returns>Anomaly score, heatmap, and verdict</returns>
        public PatchCoreResult Detect(Bitmap image)
        {
            // 1. Convert Bitmap → float32 tensor [1, 3, 640, 640] (RGB, 0–1)
            var tensor = BitmapToTensor(image);

            // 2. Run inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);

            // 3. Parse outputs
            //    anomaly_score: [1] float32
            //    anomaly_map:   [1, 1, 640, 640] float32
            var scoreTensor = results.ElementAt(0).AsTensor<float>();
            var mapTensor = results.ElementAt(1).AsTensor<float>();

            float score = scoreTensor[0];
            int mapH = mapTensor.Dimensions[2];
            int mapW = mapTensor.Dimensions[3];

            var map = new float[mapH, mapW];
            for (int y = 0; y < mapH; y++)
                for (int x = 0; x < mapW; x++)
                    map[y, x] = mapTensor[0, 0, y, x];

            return new PatchCoreResult
            {
                AnomalyScore = score,
                AnomalyMap = map,
                IsAnomaly = score > _threshold
            };
        }

        /// <summary>
        /// Convert a System.Drawing.Bitmap to a float32 tensor [1, 3, H, W].
        /// RGB channel order, values in [0, 1].
        /// ImageNet normalization is handled inside the ONNX model.
        /// </summary>
        private static DenseTensor<float> BitmapToTensor(Bitmap bmp)
        {
            int w = bmp.Width;
            int h = bmp.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

            // Lock bitmap bits for fast pixel access
            var rect = new Rectangle(0, 0, w, h);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly,
                                        PixelFormat.Format24bppRgb);
            try
            {
                int stride = bmpData.Stride;
                byte[] pixels = new byte[stride * h];
                Marshal.Copy(bmpData.Scan0, pixels, 0, pixels.Length);

                for (int y = 0; y < h; y++)
                {
                    int rowOffset = y * stride;
                    for (int x = 0; x < w; x++)
                    {
                        int idx = rowOffset + x * 3;
                        // BMP is BGR → convert to RGB, scale to [0, 1]
                        float b = pixels[idx] / 255f;
                        float g = pixels[idx + 1] / 255f;
                        float r = pixels[idx + 2] / 255f;

                        tensor[0, 0, y, x] = r;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = b;
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }

            return tensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
