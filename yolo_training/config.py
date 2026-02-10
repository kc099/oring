"""
Configuration and hyperparameter settings for YOLO training.
Easily modify these settings to experiment with different configurations.
"""

# Model Configuration
MODEL_CONFIG = {
    # Model size options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
    'model_name': 'yolov8m-seg',
    'pretrained': True,
    'freeze_backbone': False,  # Freeze backbone layers
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 8,  # Adjust based on GPU memory
    'imgsz': 640,  # Input image size
    'patience': 20,  # Early stopping patience
    'device': 0,  # GPU device (0 for first GPU, 'cpu' for CPU)
    'workers': 4,  # Data loader workers
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,      # Image HSV-Hue augmentation
    'hsv_s': 0.7,        # Image HSV-Saturation augmentation
    'hsv_v': 0.4,        # Image HSV-Value augmentation
    'degrees': 10.0,     # Image rotation (+/- deg)
    'translate': 0.1,    # Image translation (+/- fraction)
    'scale': 0.5,        # Image scale (+/- gain)
    'flipud': 0.0,       # Image flip up-down (probability)
    'fliplr': 0.5,       # Image flip left-right (probability)
    'mosaic': 1.0,       # Image mosaic (probability)
    'mixup': 0.0,        # Image mixup (probability)
    'copy_paste': 0.0,   # Segment copy-paste (probability)
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    'optimizer': 'SGD',  # Options: SGD, Adam, AdamW
    'lr0': 0.01,        # Initial learning rate
    'lrf': 0.01,        # Final learning rate
    'momentum': 0.937,   # SGD momentum
    'weight_decay': 0.0005,  # Weight decay
}

# Dataset Split Configuration
DATASET_SPLIT_CONFIG = {
    'train_ratio': 0.95,
    'val_ratio': 0.05,
    'test_ratio': 0.0,
    'random_seed': 42,  # For reproducibility
}

# Inference Configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'device': 'cpu',  # Use 'cpu' or GPU index
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'training.log',
}

# Paths Configuration (relative to script location)
PATHS_CONFIG = {
    'yolo_dataset': '../yolo_dataset',
    'yolo_dataset_split': '../yolo_dataset_split',
    'output_dir': 'runs',
    'inference_output': 'inference_results',
}


def get_train_config() -> dict:
    """Get complete training configuration."""
    config = {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'paths': PATHS_CONFIG,
    }
    return config


def get_inference_config() -> dict:
    """Get complete inference configuration."""
    return INFERENCE_CONFIG


def print_config():
    """Print current configuration."""
    config = get_train_config()
    print("\n" + "=" * 80)
    print("YOLO Training Configuration")
    print("=" * 80)
    
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in settings.items():
            print(f"  {key:20s}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_config()
