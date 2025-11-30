# plot_misclassified_grid.py
#
# This script:
# 1. Loads the CSV of all misclassified samples (all_misclassified.csv).
# 2. For the first N misclassified samples, loads the original image.
# 3. Creates a grid figure with each cell showing:
#       - the image
#       - title "True: <class> | Pred: <class>"
# 4. Saves the grid image as a single PNG for including in reports.

import os
import csv

from PIL import Image
import matplotlib.pyplot as plt


# ==========================
# Config (EDIT IF NEEDED)
# ==========================
CSV_PATH = "all_misclassified.csv"           # from visualize_all_misclassified.py
OUTPUT_PNG = "misclassified_grid.png"        # output figure

MAX_IMAGES = 40                               # max number of misclassified samples to show
NUM_COLS = 5                                  # number of columns in the grid


def load_misclassified(csv_path, max_images=None):
    """
    Load misclassified samples from CSV.
    Returns a list of dicts:
        {"path": str, "true_name": str, "pred_name": str}
    """
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # If row has empty pred_class_name, skip (means no misclassification)
            if row["pred_class_name"] == "":
                continue
            samples.append({
                "path": row["image_path"],
                "true_name": row["true_class_name"],
                "pred_name": row["pred_class_name"],
            })
            if max_images is not None and len(samples) >= max_images:
                break
    return samples


def main():
    # ---- Load list of misclassified samples ----
    samples = load_misclassified(CSV_PATH, max_images=MAX_IMAGES)
    num_samples = len(samples)
    if num_samples == 0:
        print("No misclassified samples found in CSV.")
        return

    print(f"Loaded {num_samples} misclassified samples from CSV.")

    # ---- Grid size ----
    cols = NUM_COLS
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # ---- Plot each misclassified image ----
    for ax, sample in zip(axes, samples):
        img_path = sample["path"]
        img = Image.open(img_path).convert("RGB")

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"True: {sample['true_name']}\nPred: {sample['pred_name']}",
            fontsize=8
        )

    # Turn off remaining axes (if any)
    for ax in axes[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print(f"Grid image saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()