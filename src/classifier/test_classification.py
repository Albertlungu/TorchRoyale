import json
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import datasets, transforms

# Device setup
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Transform (must match your training transform)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load model
train_dataset = datasets.ImageFolder("data/all_data/", transform=transform)
num_classes = len(train_dataset.classes)  # Change this to your actual number of classes
model = torchvision.models.resnet18(
    weights=None
)  # no pretrained, just your trained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("data/models/resnet18_final.pth", map_location=device))
model.to(device)
model.eval()

# Your 6 test images
test_files = [
    "tests/test_indiv/Screenshot 2026-01-21 at 12.47.22 PM.png",
    "tests/test_indiv/Screenshot 2026-01-21 at 12.47.29 PM.png",
    "tests/test_indiv/Screenshot 2026-01-21 at 12.47.53 PM.png",
]

# Load class mapping
with open("data/json/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}


# Predict function
def predict(image_path):
    preds = []
    # 1. Load the image
    img = Image.open(image_path).convert("RGB")

    # 2. Transform and explicitly tell the type-checker it's a Tensor
    image_tensor = cast(torch.Tensor, transform(img))

    # 3. Use stack now (the type-checker will no longer complain)
    image_batch = torch.stack([image_tensor]).to(device)

    with torch.no_grad():
        output = model(image_batch)
        probs = F.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
    class_probs = {
        idx_to_class[i]: float(probs[0, i]) for i in range(len(idx_to_class))
    }

    return idx_to_class[pred_idx], class_probs


# Run predictions
for file in test_files:
    label = predict(file)
    print(f"{file} -> {label}")
    print(label[1])
