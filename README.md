# 🗽 NYC Landmark Image Classifier (ViT-B/16 + PyTorch)

This repository contains an end-to-end **NYC Landmark Image Classification** system built with **Vision Transformer (ViT-B/16)** and **PyTorch**.  
It includes training, evaluation, visualization tools, and a Gradio-based demo for real-time inference.

The repository contains only **code and analysis tools**.  
**Dataset and trained model weights are intentionally excluded** due to size and licensing constraints.

---

## 🚀 Features

- ✔ **Vision Transformer (ViT-B/16)** fine-tuned on NYC landmark dataset  
- ✔ **Full training pipeline** with:
  - Data augmentation
  - AdamW optimizer
  - ReduceLROnPlateau LR scheduler
  - Early stopping
  - Training metrics logging
- ✔ **Evaluation utilities**
  - Confusion matrix (CSV)
  - Per-class precision / recall / F1
- ✔ **Visualization tools**
  - Misclassified sample extraction & grid visualization
  - Gradient-based saliency maps (class-specific, e.g., StatueOfLiberty)
  - Dataset class distribution plots
- ✔ **Gradio Web UI** for uploading images and viewing predictions

---

## 📁 Project Structure

```text
project/
  train_vit_nyc.py                 # Main training script
  app_gradio.py                    # Gradio demo UI for inference

  dataset_overview.py              # Dataset distribution visualizer
  per_class_metrics_from_cm.py     # Per-class precision/recall/F1
  visualize_all_misclassified.py   # Save all misclassified samples
  plot_misclassified_grid.py       # Grid visualization of misclassified samples
  visualize_saliency_vit.py        # ViT gradient-based saliency maps

  results/                         # (Optional) results and figures
  misclassified_images/            # auto-generated folder
  saliency_examples/               # auto-generated folder

  README.md
```

## 📦 Dataset Structure
Your dataset should follow:

```text
dataset/
  train/
    ClassA/
    ClassB/
    ...
  valid/
    ClassA/
    ClassB/
    ...
```

## 🧰 Installation
The minimal set is:

```text
torch
torchvision
matplotlib
numpy
pillow
gradio
scikit-learn
```

## 🏋️ Training the ViT Model
Run the training script:
```bash
python train_vit_nyc.py
```
The script will:
* Load the dataset
* Train ViT-B/16 (ImageNet-pretrained)
* Save the best checkpoint as:
    ```bash
    nyc_vit_best.pth
    ```
* Log training metrics to:
    ```bash
    training_metrics.csv
    confusion_matrix.csv
    ```
You can modify dataset path & hyperparameters inside the script.

## 📊 Evaluation & Analysis Tools

### ✔ 1. Dataset Class Distribution

Generate per-class image counts:
```bash
python dataset_overview.py
```
Outputs:
* dataset_class_counts.csv
* class_counts_train.png
* class_counts_valid.png

### 2. Per-class Precision / Recall / F1
```bash
python per_class_metrics_from_cm.py
```
Outputs:
* per_class_metrics.csv

### ✔ 3. Misclassified Samples

Extract all misclassified images:
```bash
python visualize_all_misclassified.py
```
Create a grid for report usage:
```bash
python plot_misclassified_grid.py
```
Outputs:
* misclassified_images/ (individual images)
* misclassified_grid.png (summary image)

### ✔ 4. Saliency Maps (ViT Attention)

Generate ViT saliency maps (you can target a specific class, e.g., StatueOfLiberty):
```bash
python visualize_saliency_vit.py
```
Outputs:
```bash
saliency_examples/<ClassName>/
```

## 🌐 Gradio Web Demo
Launch the demo:
```bash
python app_gradio.py
```
You will see a local URL.
Upload any NYC landmark image to get predictions