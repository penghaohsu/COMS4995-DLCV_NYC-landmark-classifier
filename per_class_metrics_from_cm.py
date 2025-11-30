# per_class_metrics_from_cm.py
#
# This script:
# 1. Loads confusion_matrix.csv (produced by train_vit_nyc.py).
# 2. Computes per-class precision, recall, F1-score and support.
# 3. Saves results as per_class_metrics.csv

import csv
import numpy as np


# ====== EDIT THIS ======
CM_CSV = "confusion_matrix.csv"
OUTPUT_CSV = "per_class_metrics.csv"
# =======================


def load_confusion_matrix(csv_path):
    """
    Load confusion matrix from CSV.
    Expected format from train_vit_nyc.py:
        first row: ["", pred_class_0, pred_class_1, ...]
        following rows: [true_class_name, c00, c01, ...]
    Returns:
        cm: NumPy array of shape (num_classes, num_classes)
        class_names: list of class names (true labels order)
    """
    class_names = []
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # first row
        # header[1:] are predicted class names (not strictly needed here)

        for row in reader:
            true_name = row[0]
            counts = list(map(int, row[1:]))
            class_names.append(true_name)
            rows.append(counts)

    cm = np.array(rows, dtype=int)
    return cm, class_names


def compute_metrics_from_cm(cm):
    """
    Given confusion matrix cm (shape [C, C]):
        cm[i, j] = number of samples with true class i predicted as j
    Returns:
        metrics: dict with keys:
            "precision", "recall", "f1", "support"
        Each is a NumPy array of shape [C].
    """
    num_classes = cm.shape[0]
    metrics = {}

    # True positives per class
    tp = np.diag(cm)

    # For each class:
    #   support (true samples) = sum over row i
    support = cm.sum(axis=1)

    #   predicted positives = sum over column j
    pred_pos = cm.sum(axis=0)

    #   false negatives = support - tp
    fn = support - tp

    #   false positives = pred_pos - tp
    fp = pred_pos - tp

    precision = np.zeros(num_classes, dtype=float)
    recall = np.zeros(num_classes, dtype=float)
    f1 = np.zeros(num_classes, dtype=float)

    for i in range(num_classes):
        # Precision: TP / (TP + FP)
        denom_p = tp[i] + fp[i]
        if denom_p > 0:
            precision[i] = tp[i] / denom_p
        else:
            precision[i] = 0.0

        # Recall: TP / (TP + FN)
        denom_r = tp[i] + fn[i]
        if denom_r > 0:
            recall[i] = tp[i] / denom_r
        else:
            recall[i] = 0.0

        # F1: 2 * P * R / (P + R)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["support"] = support

    return metrics


def main():
    cm, class_names = load_confusion_matrix(CM_CSV)
    metrics = compute_metrics_from_cm(cm)

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = metrics["support"]

    # Save per-class metrics
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "support", "precision", "recall", "f1"])
        for i, cname in enumerate(class_names):
            writer.writerow([
                cname,
                int(support[i]),
                float(precision[i]),
                float(recall[i]),
                float(f1[i]),
            ])

    print(f"Saved per-class metrics to {OUTPUT_CSV}")

    # Also print macro averages for convenience
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")


if __name__ == "__main__":
    main()