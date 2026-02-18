"""Training script for the AI Image Detection CNN classifier.

Usage:
    python scripts/train.py --data_dir data/ --epochs 20 --batch_size 32

Expected data directory layout:
    data/
      train/
        ai/      ← AI-generated images
        real/    ← Real photographs
      val/
        ai/
        real/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.core.config import get_settings
from app.models.cnn_classifier import CNNClassifier


def get_transforms(image_size: int) -> dict[str, transforms.Compose]:
    """Build train/val transforms with augmentation."""
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels_dev)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).squeeze(1)
        preds = (probs >= 0.5).float()
        correct += (preds == labels_dev.squeeze(1)).sum().item()
        total += labels_dev.size(0)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "probs": all_probs,
        "labels": all_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI Image Detector")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models/weights")
    args = parser.parse_args()

    settings = get_settings()
    device = torch.device(args.device or settings.device)
    image_size = settings.input_image_size
    model_name = settings.model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model:  {model_name}")
    print(f"Image size: {image_size}")

    # ---- Data loaders ---------------------------------------------------
    tfms = get_transforms(image_size)
    data_dir = Path(args.data_dir)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=tfms["train"])
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=tfms["val"])

    print(f"Train samples: {len(train_dataset)}  (classes: {train_dataset.classes})")
    print(f"Val   samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Model ----------------------------------------------------------
    # Re-use the same architecture builder from our classifier
    classifier = CNNClassifier()
    model = classifier.model
    if model is None:
        raise RuntimeError("Failed to create model")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop --------------------------------------------------
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "elapsed": round(elapsed, 1),
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={record['train_loss']:.4f}  train_acc={record['train_acc']:.4f}  "
            f"val_loss={record['val_loss']:.4f}  val_acc={record['val_acc']:.4f}  "
            f"lr={record['lr']:.6f}  ({record['elapsed']:.1f}s)"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            weights_path = output_dir / f"{model_name}_detector.pth"
            torch.save(model.state_dict(), weights_path)
            print(f"  → Saved best weights ({best_val_acc:.4f}) to {weights_path}")

    # ---- Save training history ------------------------------------------
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
