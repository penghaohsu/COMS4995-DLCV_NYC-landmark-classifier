# dataset_overview.py
#
# This script:
# 1. Counts how many images each class has in train/valid.
# 2. Saves a CSV with counts.
# 3. Saves two bar charts: one for train, one for valid.

import os
import csv
from collections import Counter
import matplotlib.pyplot as plt


# ====== EDIT THIS ======
DATA_DIR = "/home/peng_hao/Desktop/columbia/project/dataset"
OUTPUT_CSV = "dataset_class_counts.csv"
TRAIN_PNG = "class_counts_train.png"
VALID_PNG = "class_counts_valid.png"
# =======================


def count_images_in_split(split_dir):
    """
    Count images in a split directory with structure:
        split_dir/class_name/*.jpg
    Returns:
        class_names (sorted list)
        counts (list of counts in the same order)
    """
    class_counts = Counter()

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Count all files (you can filter by extension if needed)
        num_files = sum(
            1 for fname in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, fname))
        )
        class_counts[class_name] = num_files

    class_names = list(class_counts.keys())
    counts = [class_counts[c] for c in class_names]
    return class_names, counts


def plot_bar(class_names, counts, title, output_path):
    plt.figure(figsize=(max(8, len(class_names) * 0.6), 6))
    plt.bar(range(len(class_names)), counts)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "valid")

    train_classes, train_counts = count_images_in_split(train_dir)
    valid_classes, valid_counts = count_images_in_split(valid_dir)

    # Check that class order is consistent
    if train_classes != valid_classes:
        print("Warning: train and valid classes differ in ordering or names.")
    class_names = train_classes

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "train_count", "valid_count"])
        for i, cname in enumerate(class_names):
            t_cnt = train_counts[i]
            v_cnt = valid_counts[i] if i < len(valid_counts) else 0
            writer.writerow([cname, t_cnt, v_cnt])

    print(f"Saved class counts CSV to {OUTPUT_CSV}")

    # Plot bar charts
    plot_bar(class_names, train_counts, "Train Class Distribution", TRAIN_PNG)
    plot_bar(class_names, valid_counts, "Validation Class Distribution", VALID_PNG)


if __name__ == "__main__":
    main()