using System;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;

namespace OringInspection
{
    /// <summary>
    /// WPF MainWindow code-behind for O-Ring Inspection.
    /// Demonstrates loading an image, running ONNX Mask R-CNN inference,
    /// and displaying the verdict with defect overlay.
    /// </summary>
    public partial class MainWindow : Window
    {
        private InspectionService _service;
        private Bitmap _currentImage;

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // Load the ONNX model asynchronously to avoid blocking UI
            StatusText.Text = "Model: loading...";

            await Task.Run(() =>
            {
                string modelPath = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory,
                    "maskrcnn_oring.onnx");

                if (!File.Exists(modelPath))
                {
                    // Try relative path from workspace
                    modelPath = @"F:\standard elastomers\onnx_export\maskrcnn_oring.onnx";
                }

                _service = new InspectionService(modelPath, scoreThreshold: 0.5f);
            });

            StatusText.Text = "Model: ready";
            StatusText.Foreground = new SolidColorBrush(
                System.Windows.Media.Color.FromRgb(100, 220, 100));
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Images|*.bmp;*.jpg;*.jpeg;*.png;*.tiff|All|*.*",
                Title = "Select O-Ring Image"
            };

            if (dlg.ShowDialog() == true)
            {
                _currentImage?.Dispose();
                _currentImage = new Bitmap(dlg.FileName);

                // Display the raw image
                ImageDisplay.Source = InspectionService.BitmapToBitmapSource(_currentImage);

                BtnAnalyze.IsEnabled = true;
                VerdictText.Text = "AWAITING";
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(55, 55, 64));
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

            InspectionResult result = null;
            await Task.Run(() =>
            {
                result = _service.Inspect(_currentImage);
            });

            // Update UI with results
            VerdictText.Text = result.Verdict;

            if (result.HasDefect)
            {
                // REJECT — red
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(211, 47, 47));
                DetailsText.Text = $"{result.Detections.Count} defect(s) detected  " +
                                   $"(top score: {result.TopScore:P0})";
            }
            else
            {
                // PASS — green
                VerdictBorder.Background = new SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(27, 94, 32));
                DetailsText.Text = "No defects detected";
            }

            // Show overlay image
            ImageDisplay.Source = InspectionService.BitmapToBitmapSource(result.OverlayImage);

            CycleTimeText.Text = $"Cycle Time: {result.CycleTimeMs} ms";
            StatusText.Text = "Model: ready";
            BtnAnalyze.IsEnabled = true;
        }

        protected override void OnClosed(EventArgs e)
        {
            _service?.Dispose();
            _currentImage?.Dispose();
            base.OnClosed(e);
        }
    }
}
