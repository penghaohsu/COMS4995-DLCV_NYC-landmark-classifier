# app_gradio.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


CKPT_PATH = "nyc_vit_best.pth"  # Must match the checkpoint name in train_vit_nyc.py


def build_model(num_classes):
    """
    Build a ViT-B/16 model with ImageNet pretrained weights and replace the head.
    """
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ===== Load model checkpoint =====
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
class_names = checkpoint["class_names"]
img_size = checkpoint.get("img_size", 224)

num_classes = len(class_names)
model = build_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing (should match validation transforms except for augmentation)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def predict(image: Image.Image):
    """
    Gradio callback: take a PIL image, return class probabilities.
    """
    model.eval()
    with torch.no_grad():
        img = preprocess(image).unsqueeze(0).to(device)
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

        topk = torch.topk(probs, k=min(3, len(class_names)))
        scores = topk.values.cpu().numpy()
        indices = topk.indices.cpu().numpy()

        # Gradio Label expects a dict: {class_name: probability}
        return {
            class_names[i]: float(scores[j])
            for j, i in enumerate(indices)
        }


title = "NYC Landmark Classifier (ViT-B/16)"
description = (
    "Upload a photo of a New York City landmark and the model will "
    "predict which landmark it is."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
    examples=None,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)