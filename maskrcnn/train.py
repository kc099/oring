"""
Mask R-CNN Training for O-Ring Defect Detection.

Trains a separate Mask R-CNN model for each o-ring type using
torchvision's pre-trained model with a ResNet-50 FPN backbone.

Features:
- Mixed precision training (AMP)
- Learning rate scheduling (StepLR)
- Early stopping with patience
- Best model checkpointing
- TensorBoard logging
- Per-epoch validation with COCO mAP

Usage:
    python train.py                         # train both models
    python train.py --model model1          # train model1 only
    python train.py --model model2          # train model2 only
    python train.py --resume checkpoint.pth # resume from checkpoint

Author: GitHub Copilot
Date: February 8, 2026
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from config import OUTPUT_ROOT, ORING_MODELS, TRAINING_CONFIG, OringModelConfig
from dataset import OringDefectDataset, get_train_transforms, get_val_transforms, collate_fn


# ─── Model Construction ─────────────────────────────────────────────────────

def build_maskrcnn(num_classes: int = 2, cfg=TRAINING_CONFIG):
    """
    Build a Mask R-CNN model with custom head for binary defect detection.

    Args:
        num_classes: 2 (background + defect)
        cfg: training configuration

    Returns:
        model: Mask R-CNN model
    """
    # Load pre-trained Mask R-CNN
    if cfg.pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=cfg.trainable_backbone_layers,
    )

    # Replace the box predictor (classification head)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            loss_items = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            print(f"    Batch {batch_idx}/{len(data_loader)} | "
                  f"Loss: {losses.item():.4f} | {loss_items}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate model on validation set.

    Returns:
        avg_loss: average total loss
        metrics: dict with per-loss breakdown
    """
    model.train()  # Keep in train mode to get losses
    total_loss = 0.0
    loss_accum = {}
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Filter out images with no annotations for loss computation
        valid_images = []
        valid_targets = []
        for img, tgt in zip(images, targets):
            if tgt["boxes"].shape[0] > 0:
                valid_images.append(img)
                valid_targets.append(tgt)

        if len(valid_images) == 0:
            continue

        loss_dict = model(valid_images, valid_targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        num_batches += 1

    n = max(num_batches, 1)
    avg_loss = total_loss / n
    metrics = {k: v / n for k, v in loss_accum.items()}
    metrics["total_loss"] = avg_loss

    return avg_loss, metrics


@torch.no_grad()
def evaluate_detection(model, data_loader, device, score_threshold=0.5):
    """
    Evaluate detection performance (precision, recall, F1) on validation set.
    Uses a simple image-level metric: does the model detect defects in defect images
    and correctly predict nothing for good images?
    """
    model.eval()

    tp = 0  # defect image, model predicts defect
    fp = 0  # good image, model predicts defect
    fn = 0  # defect image, model predicts nothing
    tn = 0  # good image, model predicts nothing

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            has_gt = target["boxes"].shape[0] > 0
            scores = output["scores"].cpu().numpy()
            has_pred = (scores >= score_threshold).any() if len(scores) > 0 else False

            if has_gt and has_pred:
                tp += 1
            elif has_gt and not has_pred:
                fn += 1
            elif not has_gt and has_pred:
                fp += 1
            else:
                tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


# ─── Main Training ──────────────────────────────────────────────────────────

def train_model(model_name: str, cfg=TRAINING_CONFIG, resume_path: str = None):
    """Train Mask R-CNN for one o-ring model."""
    model_cfg = ORING_MODELS[model_name]
    model_dir = OUTPUT_ROOT / model_name

    print(f"\n{'='*80}")
    print(f"TRAINING MASK R-CNN: {model_cfg.description}")
    print(f"{'='*80}")

    # Check COCO annotations exist
    ann_dir = model_dir / "annotations"
    for split in ["train", "val"]:
        if not (ann_dir / f"{split}.json").exists():
            print(f"ERROR: {ann_dir / f'{split}.json'} not found.")
            print("Run preprocess_to_coco.py first!")
            return

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets
    print("\nLoading datasets...")
    train_dataset = OringDefectDataset(
        image_dir=model_dir / "images" / "train",
        annotation_file=ann_dir / "train.json",
        transforms=get_train_transforms(cfg)
    )
    val_dataset = OringDefectDataset(
        image_dir=model_dir / "images" / "val",
        annotation_file=ann_dir / "val.json",
        transforms=get_val_transforms()
    )

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Model
    print("\nBuilding Mask R-CNN...")
    model = build_maskrcnn(num_classes=cfg.num_classes, cfg=cfg)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_size,
        gamma=cfg.lr_gamma
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision and device.type == "cuda" else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # TensorBoard
    log_dir = model_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(str(log_dir))
    print(f"\nTensorBoard: {log_dir}")

    # Checkpoint directory
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = []

    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Mixed precision: {cfg.mixed_precision}")
    print(f"  Early stopping patience: {cfg.patience}")
    print()

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()

        # Train
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)

        # Validate
        print("  Validating...")
        val_loss, val_metrics = evaluate(model, val_loader, device)
        det_metrics = evaluate_detection(model, val_loader, device, cfg.score_threshold)

        lr_scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Detection - P: {det_metrics['precision']:.3f}, "
              f"R: {det_metrics['recall']:.3f}, F1: {det_metrics['f1']:.3f}")
        print(f"  Time: {epoch_time:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        for k, v in det_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"Detection/{k}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "detection_metrics": det_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time
        })

        # Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ★ New best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{cfg.patience}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "model_name": model_name,
                "num_classes": cfg.num_classes,
                "backbone": cfg.backbone,
            }
        }

        if is_best or not cfg.save_best_only:
            torch.save(checkpoint, ckpt_dir / f"epoch_{epoch+1:03d}.pth")
        if is_best:
            torch.save(checkpoint, ckpt_dir / "best_model.pth")

        # Always save latest
        torch.save(checkpoint, ckpt_dir / "latest.pth")

        # Early stopping
        if patience_counter >= cfg.patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {cfg.patience} epochs)")
            break

        print()

    writer.close()

    # Save history
    with open(model_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE: {model_cfg.description}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model:    {ckpt_dir / 'best_model.pth'}")
    print(f"  History:       {model_dir / 'training_history.json'}")
    print(f"  TensorBoard:   tensorboard --logdir {log_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for o-ring defect detection")
    parser.add_argument("--model", type=str, default=None,
                        choices=["model1", "model2", "combined"],
                        help="Train only this o-ring model (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    cfg = TRAINING_CONFIG
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    if args.model:
        train_model(args.model, cfg, args.resume)
    else:
        for model_name in ORING_MODELS:
            train_model(model_name, cfg, args.resume)


if __name__ == "__main__":
    main()
