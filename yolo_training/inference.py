"""
Inference and evaluation script for trained YOLO segmentation models.
Provides tools for model evaluation and predictions on new images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import Tuple, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOSegmentationInference:
    """YOLO Segmentation inference and evaluation."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained YOLO model
            device: Device to use ('cpu' or GPU index)
        """
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        logger.info(f"Loaded model from: {model_path}")
    
    def predict_image(
        self,
        image_path: str,
        conf: float = 0.5,
        iou: float = 0.45
    ) -> dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image
            conf: Confidence threshold
            iou: IOU threshold for NMS
        
        Returns:
            Detection results
        """
        logger.info(f"Predicting on: {image_path}")
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        return results[0] if results else None
    
    def predict_batch(
        self,
        image_dir: str,
        conf: float = 0.5,
        iou: float = 0.45,
        output_dir: str = None
    ) -> List[dict]:
        """
        Run inference on a directory of images.
        
        Args:
            image_dir: Directory containing images
            conf: Confidence threshold
            iou: IOU threshold
            output_dir: Optional output directory for annotated images
        
        Returns:
            List of results for each image
        """
        logger.info(f"Predicting on batch from: {image_dir}")
        
        results = self.model.predict(
            source=image_dir,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=True,
            save=output_dir is not None,
            project=output_dir,
            name='predictions'
        )
        
        logger.info(f"Processed {len(results)} images")
        return results
    
    def evaluate(
        self,
        data_yaml: str,
        imgsz: int = 640,
        batch_size: int = 16
    ) -> dict:
        """
        Evaluate model on validation/test set.
        
        Args:
            data_yaml: Path to data.yaml
            imgsz: Input image size
            batch_size: Batch size
        
        Returns:
            Evaluation metrics
        """
        logger.info("Running evaluation...")
        
        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            verbose=True
        )
        
        logger.info("Evaluation complete!")
        return metrics
    
    def visualize_prediction(
        self,
        image_path: str,
        output_path: str = None,
        conf: float = 0.5,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize predictions on an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            conf: Confidence threshold
            thickness: Line thickness for visualization
        
        Returns:
            Annotated image array
        """
        logger.info(f"Visualizing predictions for: {image_path}")
        
        # Run inference
        results = self.predict_image(image_path, conf=conf)
        
        if results is None:
            logger.error("No results returned")
            return None
        
        # Draw on image
        annotated_image = results.plot()
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"Saved annotated image to: {output_path}")
        
        return annotated_image
    
    def get_statistics(self, results: List) -> dict:
        """
        Get statistics from batch predictions.
        
        Args:
            results: List of prediction results
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_images': len(results),
            'images_with_detections': 0,
            'total_detections': 0,
            'avg_confidence': 0,
            'avg_detections_per_image': 0
        }
        
        confidences = []
        
        for result in results:
            if result is None:
                continue
            
            n_detections = len(result.boxes) if result.boxes is not None else 0
            
            if n_detections > 0:
                stats['images_with_detections'] += 1
                stats['total_detections'] += n_detections
                
                if hasattr(result.boxes, 'conf'):
                    confidences.extend(result.boxes.conf.tolist())
        
        if confidences:
            stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        if stats['total_images'] > 0:
            stats['avg_detections_per_image'] = stats['total_detections'] / stats['total_images']
        
        return stats


def main():
    """Main inference function."""
    import glob
    import random
    
    # Paths
    training_dir = os.path.join(os.path.dirname(__file__), 'runs')
    dataset_split_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset_split')
    
    # Find the latest trained model
    yolo_dirs = glob.glob(os.path.join(training_dir, '*/weights/best.pt'))
    
    if not yolo_dirs:
        logger.error("No trained models found. Please run training first.")
        logger.info("Run: python train_yolo.py")
        return
    
    model_path = yolo_dirs[-1]  # Get latest
    data_yaml = os.path.join(dataset_split_dir, 'data.yaml')
    
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml not found at {data_yaml}")
        return
    
    logger.info("=" * 80)
    logger.info("YOLO Segmentation Inference & Evaluation")
    logger.info("=" * 80)
    
    # Initialize inference engine
    inference = YOLOSegmentationInference(model_path, device='cpu')
    
    # Predict on random sample from train/val/test combined
    sample_size_env = os.environ.get('INFERENCE_SAMPLE_SIZE', '30')
    try:
        sample_size = max(1, int(sample_size_env))
    except ValueError:
        sample_size = 30

    logger.info(f"\nRunning inference on {sample_size} random images from dataset split...")

    split_dirs = {
        'train': os.path.join(dataset_split_dir, 'images', 'train'),
        'val': os.path.join(dataset_split_dir, 'images', 'val'),
        'test': os.path.join(dataset_split_dir, 'images', 'test')
    }

    all_images = []
    for split_name, split_dir in split_dirs.items():
        if not os.path.exists(split_dir):
            continue
        for image_file in os.listdir(split_dir):
            if image_file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                all_images.append((split_name, split_dir, image_file))

    if not all_images:
        logger.warning("No images found in dataset split directories.")
        return

    sample_size = min(sample_size, len(all_images))
    sampled = random.sample(all_images, sample_size)

    output_dir = os.path.join(os.path.dirname(__file__), 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for split_name, split_dir, image_name in sampled:
        image_path = os.path.join(split_dir, image_name)
        result = inference.predict_image(image_path, conf=0.5)
        if result is not None:
            annotated_name = f"{split_name}_{image_name}"
            annotated_path = os.path.join(output_dir, annotated_name)
            annotated_image = result.plot()
            cv2.imwrite(annotated_path, annotated_image)
        results.append(result)

    stats = inference.get_statistics(results)

    logger.info("\n" + "=" * 80)
    logger.info("Inference Statistics (Random Dataset Images)")
    logger.info("=" * 80)
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"Images with detections: {stats['images_with_detections']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Average detections per image: {stats['avg_detections_per_image']:.2f}")
    logger.info(f"Average confidence: {stats['avg_confidence']:.4f}")
    logger.info("=" * 80)
    
    logger.info("\nInference complete!")


if __name__ == '__main__':
    main()
