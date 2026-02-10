"""
Script to prepare the YOLO dataset with train/val/test split.
Creates a data.yaml file required by YOLO for training.
Ensures stratified splits with both defect and good samples.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple


def create_train_val_test_split(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> dict:
    """
    Split dataset into train, val, and test sets (stratified by defect/good).
    
    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
        test_ratio: Ratio for test set (default 0.1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with split statistics
    """
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    random.seed(random_seed)

    def is_defect(label_path: str) -> bool:
        if not os.path.exists(label_path):
            return False
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
            return len(content) > 0
        except Exception:
            return False

    def split_class(files: list) -> Tuple[list, list, list]:
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        if n_total >= 3:
            if n_val == 0:
                n_val = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_test = max(0, n_test - 1)
            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(0, n_val - 1)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        return train_files, val_files, test_files

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]

    defect_files = []
    good_files = []

    for file in image_files:
        label_name = os.path.splitext(file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if is_defect(label_path):
            defect_files.append(file)
        else:
            good_files.append(file)

    d_train, d_val, d_test = split_class(defect_files)
    g_train, g_val, g_test = split_class(good_files)

    splits = {
        'train': d_train + g_train,
        'val': d_val + g_val,
        'test': d_test + g_test
    }
    
    stats = {}
    
    for split_name, files in splits.items():
        split_images_dir = os.path.join(output_dir, 'images', split_name)
        split_labels_dir = os.path.join(output_dir, 'labels', split_name)
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        # Copy files
        for file in files:
            image_path = os.path.join(images_dir, file)
            label_name = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            # Copy image
            shutil.copy2(image_path, os.path.join(split_images_dir, file))
            
            # Copy label (even if empty)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(split_labels_dir, label_name))
            else:
                # Create empty label if not found
                open(os.path.join(split_labels_dir, label_name), 'w').close()
        
        stats[split_name] = len(files)
        print(f"✓ {split_name.upper():5} : {len(files):4} images")
    
    return stats


def create_data_yaml(output_dir: str, class_names: list) -> str:
    """
    Create data.yaml file for YOLO training.
    
    Args:
        output_dir: Output directory
        class_names: List of class names
    
    Returns:
        Path to created data.yaml file
    """
    yaml_content = f"""# YOLO Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Main dataset preparation function."""
    # Paths
    yolo_dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset')
    images_dir = os.path.join(yolo_dataset_dir, 'images')
    labels_dir = os.path.join(yolo_dataset_dir, 'labels')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset_split')
    
    print("=" * 80)
    print("YOLO Dataset Preparation - Train/Val/Test Split")
    print("=" * 80)
    print(f"\nInput directories:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "=" * 80)
    print("Creating splits (80% train, 10% val, 10% test)...\n")
    
    # Create splits
    stats = create_train_val_test_split(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    # Create data.yaml
    class_names = ['defect']
    yaml_path = create_data_yaml(output_dir, class_names)
    
    print("\n" + "=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Training set  : {stats['train']} images")
    print(f"  Validation set: {stats['val']} images")
    print(f"  Test set      : {stats['test']} images")
    print(f"  Total         : {sum(stats.values())} images")
    print(f"\nClasses: {class_names}")
    print(f"\ndata.yaml created at: {yaml_path}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  │   ├── test/")
    print(f"  └── data.yaml")
    print("=" * 80)


if __name__ == '__main__':
    main()
