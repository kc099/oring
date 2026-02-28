using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;

namespace OringInspection
{
    /// <summary>
    /// WPF MainWindow code-behind for PatchCore O-Ring Anomaly Detection.
    /// Demonstrates loading an image, running PatchCore ONNX inference,
    /// and displaying the anomaly heatmap + verdict (PASS/REJECT).
    /// </summary>
    public partial class PatchCoreMainWindow : Window
    {
        private PatchCoreInspectionService _service;
        private Bitmap _currentImage;
        private PatchCoreInspectionResult _lastResult;
        private float _currentThreshold = 15.0f;

        // Available ONNX models discovered at startup
        private readonly List<ModelEntry> _models = new();

        private struct ModelEntry
        {
            public string DisplayName;
            public string OnnxPath;
            public string JsonPath;
            public float Threshold;
        }

        public PatchCoreMainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // Discover available PatchCore ONNX models
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string[] searchDirs = new[]
            {
                exeDir,
                Path.Combine(exeDir, ".."),
                @"F:\standard elastomers\onnx_export"
            };

            foreach (string dir in searchDirs)
            {
                if (!Directory.Exists(dir)) continue;
                foreach (string onnx in Directory.GetFiles(dir, "patchcore_*.onnx"))
                {
                    string jsonPath = Path.ChangeExtension(onnx, ".json");
                    float threshold = 15.0f;
                    string displayName = Path.GetFileNameWithoutExtension(onnx);

                    // Try loading metadata for threshold
                    if (File.Exists(jsonPath))
                    {
                        var meta = PatchCoreInspectionService.LoadMetadata(jsonPath);
                        if (meta?.recommended_threshold.HasValue == true)
                            threshold = meta.recommended_threshold.Value;
                        if (!string.IsNullOrEmpty(meta?.model_name))
                            displayName = meta.model_name;
                    }

                    // Avoid duplicates
                    if (_models.Any(m => m.OnnxPath == onnx)) continue;

                    _models.Add(new ModelEntry
                    {
                        DisplayName = displayName,
                        OnnxPath = onnx,
                        JsonPath = jsonPath,
                        Threshold = threshold
                    });
                }
            }

            if (_models.Count == 0)
            {
                StatusText.Text = "No PatchCore ONNX models found!";
                StatusText.Foreground = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(255, 100, 100));
                return;
            }

            // Populate combo box
            foreach (var m in _models)
                CmbModel.Items.Add(m.DisplayName);
            CmbModel.SelectedIndex = 0;
        }

        private async void CmbModel_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (CmbModel.SelectedIndex < 0 || CmbModel.SelectedIndex >= _models.Count)
                return;

            var entry = _models[CmbModel.SelectedIndex];

            // Dispose previous service
            _service?.Dispose();
            _service = null;

            StatusText.Text = $"Loading {entry.DisplayName}...";
            StatusText.Foreground = new SolidColorBrush(
                System.Windows.Media.Color.FromRgb(200, 200, 205));
            BtnAnalyze.IsEnabled = false;

            await Task.Run(() =>
            {
                _service = new PatchCoreInspectionService(
                    entry.OnnxPath,
                    threshold: entry.Threshold,
                    useGpu: false  // Change to true for GPU inference
                );
            });

            _currentThreshold = entry.Threshold;
            SliderThreshold.Value = _currentThreshold;
            TxtThreshold.Text = _currentThreshold.ToString("F1");

            StatusText.Text = $"Model: {entry.DisplayName} — ready";
            StatusText.Foreground = new SolidColorBrush(
                System.Windows.Media.Color.FromRgb(100, 220, 100));
            BtnAnalyze.IsEnabled = _currentImage != null;
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Images|*.bmp;*.jpg;*.jpeg;*.png;*.tiff|All|*.*",
                Title = "Select O-Ring Image",
                InitialDirectory = @"F:\standard elastomers\binned"
            };

            if (dlg.ShowDialog() == true)
            {
                _currentImage?.Dispose();
                _currentImage = new Bitmap(dlg.FileName);
                _lastResult = null;

                // Show original image
                ImageOriginal.Source = PatchCoreInspectionService.BitmapToBitmapSource(_currentImage);
                ImageHeatmap.Source = null;

                BtnAnalyze.IsEnabled = _service != null;
                VerdictText.Text = "AWAITING";
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(55, 55, 64));
                ScoreText.Text = "";
                CycleTimeText.Text = "";
                DetailsText.Text = $"Loaded: {Path.GetFileName(dlg.FileName)}  " +
                                   $"({_currentImage.Width}×{_currentImage.Height})";
            }
        }

        private async void Analyze_Click(object sender, RoutedEventArgs e)
        {
            if (_currentImage == null || _service == null) return;

            BtnAnalyze.IsEnabled = false;
            StatusText.Text = "Analyzing...";

            PatchCoreInspectionResult result = null;
            await Task.Run(() =>
            {
                result = _service.Inspect(_currentImage);
            });

            _lastResult = result;

            // Update score display
            ScoreText.Text = $"Score: {result.AnomalyScore:F2}";

            // Apply current threshold from slider
            bool isAnomaly = result.AnomalyScore > _currentThreshold;
            string verdict = isAnomaly ? "REJECT" : "PASS";

            VerdictText.Text = verdict;

            if (isAnomaly)
            {
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(211, 47, 47));
                DetailsText.Text = $"Anomaly detected (score {result.AnomalyScore:F2} > threshold {_currentThreshold:F1})";
            }
            else
            {
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(27, 94, 32));
                DetailsText.Text = $"No anomaly (score {result.AnomalyScore:F2} ≤ threshold {_currentThreshold:F1})";
            }

            // Show heatmap overlay
            float alpha = (float)SliderAlpha.Value;
            var overlay = PatchCoreInspectionService.DrawHeatmapOverlay(
                result.PreprocessedImage, result.AnomalyMap, alpha);
            ImageHeatmap.Source = PatchCoreInspectionService.BitmapToBitmapSource(overlay);
            overlay.Dispose();

            CycleTimeText.Text = $"Cycle Time: {result.CycleTimeMs} ms";
            StatusText.Text = $"Model: {_models[CmbModel.SelectedIndex].DisplayName} — ready";
            BtnAnalyze.IsEnabled = true;
        }

        private void SliderThreshold_ValueChanged(object sender,
            RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtThreshold == null) return;
            _currentThreshold = (float)SliderThreshold.Value;
            TxtThreshold.Text = _currentThreshold.ToString("F1");

            // Re-evaluate verdict if we have a result
            if (_lastResult != null)
            {
                bool isAnomaly = _lastResult.AnomalyScore > _currentThreshold;
                VerdictText.Text = isAnomaly ? "REJECT" : "PASS";
                VerdictBorder.Background = new SolidColorBrush(
                    isAnomaly
                        ? System.Windows.Media.Color.FromRgb(211, 47, 47)
                        : System.Windows.Media.Color.FromRgb(27, 94, 32));
                DetailsText.Text = isAnomaly
                    ? $"Anomaly detected (score {_lastResult.AnomalyScore:F2} > threshold {_currentThreshold:F1})"
                    : $"No anomaly (score {_lastResult.AnomalyScore:F2} ≤ threshold {_currentThreshold:F1})";
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _service?.Dispose();
            _currentImage?.Dispose();
            _lastResult?.PreprocessedImage?.Dispose();
            _lastResult?.OverlayImage?.Dispose();
            base.OnClosed(e);
        }
    }
}
