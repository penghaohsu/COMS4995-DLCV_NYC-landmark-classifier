# visualize_saliency_vit.py
#
# This script:
# 1. Loads the trained ViT model and checkpoint.
# 2. Iterates over the validation set.
# 3. For images whose TRUE class is a specific target class (e.g. StatueOfLiberty),
#    computes a gradient-based saliency map with respect to the predicted class.
# 4. Saves side-by-side visualizations (original + heatmap overlay)
#    into a folder "saliency_examples_<target_class>/".

import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# ====== EDIT THESE ======
DATA_DIR = "/home/peng_hao/Desktop/columbia/project/dataset"
CKPT_PATH = "nyc_vit_best.pth"

# Must exactly match the folder name under dataset/valid/
# e.g., if your structure is dataset/valid/StatueOfLiberty/xxx.jpg
TARGET_CLASS_NAME = "StatueOfLiberty"

OUTPUT_DIR_ROOT = "saliency_examples"
MAX_IMAGES = 12  # how many images from this class to visualize
# ========================


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
    # ----- Load checkpoint -----
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)
    num_classes = len(class_names)

    # ----- Build model -----
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", device)

    # ----- Validation dataset -----
    val_dir = os.path.join(DATA_DIR, "valid")

    # We use ImageFolder only to get (path, label_idx)
    val_ds = datasets.ImageFolder(val_dir)  # no transform here
    print("Validation samples:", len(val_ds))
    print("Validation classes:", val_ds.classes)

    if TARGET_CLASS_NAME not in val_ds.class_to_idx:
        raise ValueError(
            f"TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' not found in "
            f"val_ds.classes: {val_ds.classes}"
        )

    target_true_idx = val_ds.class_to_idx[TARGET_CLASS_NAME]
    print(f"Target class '{TARGET_CLASS_NAME}' has index {target_true_idx}")

    # Preprocessing (same as training/validation)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Output directory specific to this class
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, TARGET_CLASS_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Generate saliency visualizations for the target class -----
    num_done = 0
    for idx, (img_path, true_idx) in enumerate(val_ds.samples):
        # Only process samples whose TRUE label is the target class
        if true_idx != target_true_idx:
            continue

        if num_done >= MAX_IMAGES:
            break

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((img_size, img_size))

        input_tensor = preprocess(img_resized).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Forward pass (with grad) to get prediction and gradients
        model.zero_grad()
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()

        score = outputs[0, pred_idx]
        score.backward()

        grad = input_tensor.grad[0]  # (C, H, W)
        grad_abs = grad.abs()
        saliency, _ = torch.max(grad_abs, dim=0)  # (H, W)
        saliency = saliency.cpu().numpy()
        saliency = saliency - saliency.min()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        # Visualization: original + overlay
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(img_resized)
        axes[0].axis("off")
        axes[0].set_title(
            f"Original\nTrue: {TARGET_CLASS_NAME}",
            fontsize=9
        )

        axes[1].imshow(img_resized)
        axes[1].imshow(saliency, cmap="jet", alpha=0.5)
        axes[1].axis("off")
        # class_names is in the same order as during training
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        axes[1].set_title(
            f"Saliency (Pred: {pred_name})",
            fontsize=9
        )

        plt.tight_layout()

        base_name = os.path.basename(img_path)
        out_name = (
            f"saliency_{num_done:03d}_true_{TARGET_CLASS_NAME}_"
            f"pred_{pred_name.replace(' ', '')}_{base_name}"
        )
        out_path = os.path.join(OUTPUT_DIR, out_name)

        plt.savefig(out_path, dpi=300)
        plt.close()

        num_done += 1
        print(f"Saved {num_done}/{MAX_IMAGES} -> {out_path}")

    print(
        f"Done. Saliency examples for class '{TARGET_CLASS_NAME}' "
        f"saved in folder: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()