# train_vit_nyc.py

import os
import time
import csv
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning for very large images

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# -------------------------
# Data
# -------------------------
def get_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    """
    Build train/validation dataloaders assuming the following structure:

    data_dir/
        train/
            class_0/
                xxx.jpg
            class_1/
                yyy.jpg
            ...
        valid/
            class_0/
                ...
            class_1/
                ...

    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")

    # Standard ImageNet normalization for ViT
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes


# -------------------------
# Model
# -------------------------
def build_model(num_classes):
    """
    Build a ViT-B/16 model pretrained on ImageNet and replace the final head.
    Requires torchvision >= 0.13.
    """
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


# -------------------------
# Early Stopping
# -------------------------
class EarlyStopping:
    """
    Simple early stopping on validation loss.
    """
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# -------------------------
# Confusion Matrix
# -------------------------
def compute_confusion_matrix(model, loader, class_names, device):
    """
    Compute confusion matrix on the given loader using the current model.
    Returns a NumPy array of shape (num_classes, num_classes).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    return cm


def save_confusion_matrix(cm, class_names,
                          csv_path="confusion_matrix.csv",
                          fig_path="confusion_matrix.png"):
    """
    Save confusion matrix as CSV and PNG.
    """
    num_classes = len(class_names)

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [""] + list(class_names)
        writer.writerow(header)
        for i in range(num_classes):
            row = [class_names[i]] + cm[i].tolist()
            writer.writerow(row)

    # Save figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues")

    plt.colorbar(im)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Validation)")

    # Add counts on each cell
    for i in range(num_classes):
        for j in range(num_classes):
            text = str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {csv_path} and {fig_path}")


# -------------------------
# Main
# -------------------------
def main():
    # ===== Paths & hyperparameters =====
    data_dir = "/home/peng_hao/Desktop/columbia/project/dataset"  # <-- change to your path
    save_path = "nyc_vit_best.pth"
    metrics_path = "training_metrics.csv"

    num_epochs = 40
    batch_size = 32
    lr = 3e-4
    weight_decay = 1e-4
    img_size = 224
    patience = 7  # early stopping patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== Dataloaders =====
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size
    )
    num_classes = len(class_names)
    print("Classes:", class_names)

    # ===== Model / Optimizer / Scheduler =====
    model = build_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Learning rate scheduler (compatible with older PyTorch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)

    best_val_acc = 0.0
    best_state_dict = None

    # History for plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # ===== Training loop =====
    for epoch in range(num_epochs):
        since = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        time_elapsed = time.time() - since
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} "
            f"({time_elapsed:.1f}s)"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "class_names": class_names,
                    "img_size": img_size,
                },
                save_path,
            )
            print(f"  -> New best model saved to {save_path}")

        # Early stopping check
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(patience={patience})."
            )
            break

    # ===== Save training history as CSV =====
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i],
            ])

    print(f"Metrics saved to {metrics_path}")
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

    # ===== Compute confusion matrix using the best model =====
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    cm = compute_confusion_matrix(model, val_loader, class_names, device)
    save_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()