"""
YOLO Segmentation Model Training Script.
Trains a YOLOv8 segmentation model on the defect detection dataset.
"""

import os
import random
import shutil
import torch
import gc
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOSegmentationTrainer:
    """YOLO v8 Segmentation model trainer."""
    
    def __init__(self, data_yaml: str, output_dir: str = 'runs/detect'):
        """
        Initialize the trainer.
        
        Args:
            data_yaml: Path to data.yaml file
            output_dir: Output directory for results
        """
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.device = self._get_device()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Data config: {data_yaml}")
    
    def _get_device(self) -> str:
        """Get the device to use for training."""
        if torch.cuda.is_available():
            device = 0  # GPU
            logger.info(f"GPU available: {torch.cuda.get_device_name(device)}")
        else:
            device = 'cpu'
            logger.warning("GPU not available, using CPU (slow training)")
        return device
    
    def train(
        self,
        model_name: str = 'yolov8m-seg',
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        patience: int = 20,
        save: bool = True,
        save_period: int = -1,
        resume: bool = False,
        device: str = None,
        workers: int = 4,
        **kwargs
    ) -> dict:
        """
        Train the YOLO segmentation model.
        
        Args:
            model_name: Model size ('yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg')
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size
            patience: Early stopping patience
            save: Whether to save checkpoints
            save_period: Save checkpoint every X epochs (-1 to disable, 1 to save every epoch)
            resume: Resume training from last checkpoint (last.pt)
            device: Device to use ('cpu' or GPU index)
            workers: Number of data loader workers
            **kwargs: Additional arguments for YOLO.train()
        
        Returns:
            Training results
        
        Note:
            YOLO saves checkpoints automatically:
            - last.pt: Saved after EVERY epoch (can resume training)
            - best.pt: Saved when validation metric improves
            You can stop training anytime (Ctrl+C) and resume with resume=True
        """
        if device is None:
            device = self.device
        
        logger.info("=" * 80)
        logger.info("YOLO v8 Segmentation Model Training")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Image size: {imgsz}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Resume: {resume}")
        logger.info(f"Save period: {save_period if save_period > 0 else 'best only'}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Checkpoint Info:")
        logger.info("  - 'last.pt' saved after EVERY epoch (for resuming)")
        logger.info("  - 'best.pt' saved when validation improves")
        logger.info("  - Press Ctrl+C to stop training safely")
        logger.info("  - Resume with: python train_yolo.py --resume")
        logger.info("=" * 80)
        
        # Load model
        if resume:
            logger.info("Resuming training from last checkpoint...")
            model = YOLO(f'{model_name}.pt')
        else:
            logger.info(f"Loading model: {model_name}")
            model = YOLO(f'{model_name}.pt')

        def save_ground_truth(sampled_files: list, val_images_dir: str, val_labels_dir: str, output_dir: str):
            os.makedirs(output_dir, exist_ok=True)

            for image_file in sampled_files:
                image_path = os.path.join(val_images_dir, image_file)
                label_path = os.path.join(val_labels_dir, os.path.splitext(image_file)[0] + '.txt')

                image = cv2.imread(image_path)
                if image is None:
                    continue

                h, w = image.shape[:2]

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]

                    for line in lines:
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        coords = parts[1:]
                        if len(coords) % 2 != 0:
                            continue

                        points = []
                        for i in range(0, len(coords), 2):
                            x = float(coords[i]) * w
                            y = float(coords[i + 1]) * h
                            points.append([int(round(x)), int(round(y))])

                        if len(points) >= 3:
                            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                out_name = f"{Path(image_file).stem}.png"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, image)

        def save_side_by_side(sampled_files: list, gt_dir: str, pred_dir: str, output_dir: str):
            os.makedirs(output_dir, exist_ok=True)
            pred_map = {}
            for pred_file in os.listdir(pred_dir):
                if pred_file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    pred_map[Path(pred_file).stem] = os.path.join(pred_dir, pred_file)

            for image_file in sampled_files:
                stem = Path(image_file).stem
                gt_path = os.path.join(gt_dir, f"{stem}.png")
                pred_path = pred_map.get(stem)

                if not os.path.exists(gt_path) or not pred_path or not os.path.exists(pred_path):
                    continue

                gt_img = cv2.imread(gt_path)
                pred_img = cv2.imread(pred_path)
                if gt_img is None or pred_img is None:
                    continue

                # Resize to same height
                h = min(gt_img.shape[0], pred_img.shape[0])
                if gt_img.shape[0] != h:
                    w = int(gt_img.shape[1] * (h / gt_img.shape[0]))
                    gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)
                if pred_img.shape[0] != h:
                    w = int(pred_img.shape[1] * (h / pred_img.shape[0]))
                    pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)

                combined = cv2.hconcat([gt_img, pred_img])
                out_path = os.path.join(output_dir, f"{stem}.png")
                cv2.imwrite(out_path, combined)

        def save_val_predictions(trainer):
            try:
                val_path = None
                if isinstance(trainer.data, dict):
                    val_path = trainer.data.get('val')
                if not val_path:
                    return

                data_root = None
                if isinstance(trainer.data, dict):
                    data_root = trainer.data.get('path')
                if data_root and not os.path.isabs(val_path):
                    val_path = os.path.join(data_root, val_path)

                epoch_num = trainer.epoch + 1
                output_dir = os.path.join(trainer.save_dir, 'val_predictions', f'epoch_{epoch_num:03d}')
                os.makedirs(output_dir, exist_ok=True)

                val_images_dir = val_path
                if os.path.basename(val_images_dir) != 'val':
                    pass
                val_labels_dir = val_images_dir.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
                gt_dir = os.path.join(output_dir, 'ground_truth')

                image_files = [
                    f for f in os.listdir(val_images_dir)
                    if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))
                ]
                if not image_files:
                    return

                sample_size = min(20, len(image_files))
                random.seed(42 + epoch_num)
                sampled_files = random.sample(image_files, sample_size)

                if os.path.isdir(val_images_dir) and os.path.isdir(val_labels_dir):
                    save_ground_truth(sampled_files, val_images_dir, val_labels_dir, gt_dir)

                weights_path = os.path.join(trainer.save_dir, 'weights', 'last.pt')
                if not os.path.exists(weights_path):
                    return

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                yolo_model = YOLO(weights_path)
                sampled_paths = [os.path.join(val_images_dir, f) for f in sampled_files]
                yolo_model.predict(
                    sampled_paths,
                    save=True,
                    project=output_dir,
                    name='images',
                    exist_ok=True,
                    verbose=False,
                    device='cpu',
                    batch=1
                )

                pred_dir = os.path.join(output_dir, 'images')
                if os.path.isdir(gt_dir) and os.path.isdir(pred_dir):
                    save_side_by_side(sampled_files, gt_dir, pred_dir, output_dir)

                if os.path.isdir(gt_dir):
                    shutil.rmtree(gt_dir, ignore_errors=True)
                if os.path.isdir(pred_dir):
                    shutil.rmtree(pred_dir, ignore_errors=True)

                del yolo_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Saved val predictions for epoch {epoch_num} to: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to save val predictions: {e}")

        model.add_callback('on_fit_epoch_end', save_val_predictions)
        
        # Training parameters
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': patience,
            'device': device,
            'workers': workers,
            'save': save,
            'save_period': save_period,
            'resume': resume,
            'project': self.output_dir,
            'name': f'{model_name}_training',
            'exist_ok': True if resume else False,
            'verbose': True,
            'plots': True,
            'amp': True,  # Automatic Mixed Precision
            'hsv_h': 0.015,  # Image HSV-Hue augmentation
            'hsv_s': 0.7,    # Image HSV-Saturation augmentation
            'hsv_v': 0.4,    # Image HSV-Value augmentation
            'degrees': 10.0,  # Image rotation (+/- deg)
            'translate': 0.1,  # Image translation (+/- fraction)
            'scale': 0.5,     # Image scale (+/- gain)
            'flipud': 0.0,    # Image flip up-down (probability)
            'fliplr': 0.5,    # Image flip left-right (probability)
            'mosaic': 1.0,    # Image mosaic (probability)
            'mixup': 0.0,     # Image mixup (probability)
            'copy_paste': 0.0,  # Segment copy-paste (probability)
        }
        
        # Add any additional arguments
        train_args.update(kwargs)
        
        # Train
        try:
            results = model.train(**train_args)
            logger.info("=" * 80)
            logger.info("Training Completed Successfully!")
            logger.info("=" * 80)
            logger.info(f"Results saved to: {self.output_dir}")
            return results
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def validate(self, model_path: str) -> dict:
        """
        Validate a trained model.
        
        Args:
            model_path: Path to the trained model
        
        Returns:
            Validation results
        """
        logger.info("Running validation...")
        model = YOLO(model_path)
        results = model.val(data=self.data_yaml, device=self.device)
        return results
    
    def predict(self, model_path: str, image_path: str, conf: float = 0.5) -> list:
        """
        Run inference on an image.
        
        Args:
            model_path: Path to the trained model
            image_path: Path to the image
            conf: Confidence threshold
        
        Returns:
            Prediction results
        """
        logger.info(f"Running inference on: {image_path}")
        model = YOLO(model_path)
        results = model.predict(source=image_path, conf=conf, device=self.device)
        return results


def main():
    """Main training function."""
    # Paths
    dataset_split_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset_split')
    data_yaml = os.path.join(dataset_split_dir, 'data.yaml')
    output_dir = os.path.join(os.path.dirname(__file__), 'runs')
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml not found at {data_yaml}")
        logger.info("Please run 'python prepare_dataset.py' first to prepare the dataset.")
        return
    
    # Create trainer
    trainer = YOLOSegmentationTrainer(
        data_yaml=data_yaml,
        output_dir=output_dir
    )
    
    # Training parameters
    model_name = 'yolov8m-seg'  # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
    epochs = 100
    batch_size = 8
    imgsz = 640
    
    # Train
    results = trainer.train(
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        patience=20,
        save=True,
        workers=4
    )
    
    logger.info(f"\nTraining complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
