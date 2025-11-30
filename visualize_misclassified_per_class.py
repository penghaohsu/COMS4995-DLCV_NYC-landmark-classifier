# visualize_all_misclassified.py
#
# This script:
# 1. Loads the trained ViT classifier and checkpoint.
# 2. Runs inference on the validation set.
# 3. Collects ALL misclassified samples.
# 4. For each misclassified sample, saves a figure showing the image
#    with a title: "True: <class>, Pred: <class>".
# 5. Saves a CSV listing all misclassified samples.
#
# You can then pick representative examples from the output folder
# to include in your report.

import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from PIL import Image
import matplotlib.pyplot as plt


# ==========================
# Paths (EDIT THESE)
# ==========================
DATA_DIR = "/home/peng_hao/Desktop/columbia/project/dataset"  # same root as in training
CKPT_PATH = "nyc_vit_best.pth"                                # checkpoint from training

OUTPUT_DIR = "misclassified_images"                           # folder for images
OUTPUT_CSV = "all_misclassified.csv"                          # CSV summary


# ==========================
# Model definition
# ==========================
def build_model(num_classes):
    """
    Build a ViT-B/16 model with ImageNet pretrained weights and replace the head.
    Must match the architecture used during training.
    """
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def main():
    # ---------- Load checkpoint ----------
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Num classes:", num_classes)

    # ---------- Build model and load weights ----------
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", device)

    # ---------- Validation dataset & dataloader ----------
    val_dir = os.path.join(DATA_DIR, "valid")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Validation samples:", len(val_ds))

    # List of all misclassified samples
    # each element: {"path": str, "true": int, "pred": int}
    misclassified = []

    # ---------- Run inference over validation set ----------
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            batch_size = images.size(0)
            for i in range(batch_size):
                idx_in_ds = batch_idx * val_loader.batch_size + i
                if idx_in_ds >= len(val_ds):
                    break  # safety check

                true_c = labels[i].item()
                pred_c = preds[i].item()

                if true_c != pred_c:
                    img_path, _ = val_ds.samples[idx_in_ds]
                    misclassified.append(
                        {
                            "path": img_path,
                            "true": true_c,
                            "pred": pred_c,
                        }
                    )

    print(f"Total misclassified samples: {len(misclassified)}")

    # ---------- Save CSV summary ----------
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class_idx", "true_class_name",
                         "pred_class_idx", "pred_class_name", "image_path"])
        for info in misclassified:
            writer.writerow([
                info["true"],
                class_names[info["true"]],
                info["pred"],
                class_names[info["pred"]],
                info["path"],
            ])

    print(f"CSV saved to {OUTPUT_CSV}")

    # ---------- Save visualization images ----------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k, info in enumerate(misclassified):
        img = Image.open(info["path"]).convert("RGB")

        true_name = class_names[info["true"]]
        pred_name = class_names[info["pred"]]

        # Create a clean figure for each misclassified sample
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {true_name} | Pred: {pred_name}", fontsize=9)
        plt.tight_layout()

        base_name = os.path.basename(info["path"])
        # sanitize class names a bit for filenames
        true_tag = true_name.replace(" ", "")
        pred_tag = pred_name.replace(" ", "")
        out_name = f"true_{true_tag}__pred_{pred_tag}__{base_name}"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        plt.savefig(out_path, dpi=200)
        plt.close()

        if (k + 1) % 50 == 0:
            print(f"Saved {k+1}/{len(misclassified)} misclassified images...")

    print(f"All misclassified images saved to folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()