# plot_metrics.py

import csv
import matplotlib.pyplot as plt


def load_metrics(csv_path="training_metrics.csv"):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    lrs = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
            if "lr" in row and row["lr"] != "":
                lrs.append(float(row["lr"]))

    return epochs, train_loss, val_loss, train_acc, val_acc, lrs


def main():
    epochs, train_loss, val_loss, train_acc, val_acc, lrs = load_metrics()

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=300)

    # Learning rate curve (optional)
    if lrs:
        plt.figure()
        plt.plot(epochs[:len(lrs)], lrs, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("lr_curve.png", dpi=300)

    print("Saved loss_curve.png, acc_curve.png and (optionally) lr_curve.png")


if __name__ == "__main__":
    main()