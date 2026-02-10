"""
Quick start script - Runs the entire training pipeline in sequence.
Execute this to train YOLO model from start to finish.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(script_name: str, description: str) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
    
    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    logger.info("=" * 80)
    logger.info(description)
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error code {e.returncode}\n")
        return False
    except Exception as e:
        logger.error(f"✗ Error running {script_name}: {e}\n")
        return False


def main():
    """Main execution function."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "YOLO SEGMENTATION MODEL - TRAINING PIPELINE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    steps = [
        ("prepare_dataset.py", "STEP 1: Preparing Dataset (Train/Val/Test Split)"),
        ("train_yolo.py", "STEP 2: Training YOLO Segmentation Model"),
        ("inference.py", "STEP 3: Running Inference & Evaluation"),
    ]
    
    failed_steps = []
    
    for script, description in steps:
        success = run_command(script, description)
        if not success:
            failed_steps.append(description)
    
    # Summary
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "PIPELINE EXECUTION SUMMARY".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    if not failed_steps:
        logger.info("✓ All steps completed successfully!")
        logger.info("\nTraining pipeline finished. Check results in:")
        logger.info("  - Models: yolo_dataset_split/")
        logger.info("  - Training: runs/")
        logger.info("  - Inference: inference_results/")
        return 0
    else:
        logger.error("✗ Some steps failed:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        logger.info("\nPlease check the errors above and fix them.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
