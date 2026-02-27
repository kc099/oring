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
    /// Result from a single Mask R-CNN inference pass.
    /// </summary>
    public class Detection
    {
        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Score { get; set; }
        public int Label { get; set; }
        public float[,] Mask { get; set; }  // soft mask [H, W], threshold at 0.5
    }

    /// <summary>
    /// Mask R-CNN ONNX inference wrapper for O-Ring defect detection.
    /// 
    /// Usage:
    ///   var detector = new MaskRCNNDetector("maskrcnn_oring.onnx");
    ///   var detections = detector.Detect(bitmap, scoreThreshold: 0.5f);
    ///   detector.Dispose();
    /// 
    /// NuGet packages required:
    ///   - Microsoft.ML.OnnxRuntime        (CPU)
    ///   - Microsoft.ML.OnnxRuntime.Gpu     (GPU — optional, for CUDA)
    /// </summary>
    public class MaskRCNNDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        /// <summary>
        /// Initialize the detector with an ONNX model file.
        /// </summary>
        /// <param name="modelPath">Path to maskrcnn_oring.onnx</param>
        /// <param name="useGpu">Use CUDA GPU if available</param>
        public MaskRCNNDetector(string modelPath, bool useGpu = false)
        {
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            // Use 4 threads for CPU inference
            options.InterOpNumThreads = 4;
            options.IntraOpNumThreads = 4;

            if (useGpu)
            {
                // Requires Microsoft.ML.OnnxRuntime.Gpu NuGet package
                // and CUDA 11.x / cuDNN installed
                options.AppendExecutionProvider_CUDA(0);
            }

            _session = new InferenceSession(modelPath, options);
            _inputName = _session.InputMetadata.Keys.First();  // "image"
        }

        /// <summary>
        /// Run inference on a Bitmap image.
        /// The image should be the preprocessed 720×720 o-ring crop.
        /// </summary>
        /// <param name="image">Input image (BGR or RGB Bitmap)</param>
        /// <param name="scoreThreshold">Minimum confidence score (0.0-1.0)</param>
        /// <returns>List of detections above the score threshold</returns>
        public List<Detection> Detect(Bitmap image, float scoreThreshold = 0.5f)
        {
            // 1. Convert Bitmap to float32 tensor [1, 3, H, W] (RGB, 0-1)
            var tensor = BitmapToTensor(image);

            // 2. Run inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);

            // 3. Parse outputs
            //    boxes:  [N, 4]     float32
            //    labels: [N]        int64
            //    scores: [N]        float32
            //    masks:  [N, 1, H, W] float32
            var boxes = results.ElementAt(0).AsTensor<float>();
            var labels = results.ElementAt(1).AsTensor<long>();
            var scores = results.ElementAt(2).AsTensor<float>();
            var masks = results.ElementAt(3).AsTensor<float>();

            int numDetections = (int)scores.Length;
            int maskH = masks.Dimensions.Length >= 3 ? masks.Dimensions[2] : image.Height;
            int maskW = masks.Dimensions.Length >= 4 ? masks.Dimensions[3] : image.Width;

            var detections = new List<Detection>();
            for (int i = 0; i < numDetections; i++)
            {
                float score = scores[i];
                if (score < scoreThreshold)
                    continue;

                var det = new Detection
                {
                    X1 = boxes[i, 0],
                    Y1 = boxes[i, 1],
                    X2 = boxes[i, 2],
                    Y2 = boxes[i, 3],
                    Score = score,
                    Label = (int)labels[i],
                    Mask = new float[maskH, maskW]
                };

                // Copy mask data
                for (int y = 0; y < maskH; y++)
                    for (int x = 0; x < maskW; x++)
                        det.Mask[y, x] = masks[i, 0, y, x];

                detections.Add(det);
            }

            return detections;
        }

        /// <summary>
        /// Convert a System.Drawing.Bitmap to a float32 tensor [1, 3, H, W].
        /// RGB channel order, values in [0, 1].
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
                        // BMP is BGR, convert to RGB
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
