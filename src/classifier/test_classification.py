import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform (must match your training transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load model
num_classes = 12  # Change this to your actual number of classes
model = torchvision.models.resnet18(weights=None)  # no pretrained, just your trained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_final.pth", map_location=device))
model.to(device)
model.eval()

# Your 6 test images
test_files = [
    "test_images/img1.jpg",
    "test_images/img2.jpg",
    "test_images/img3.jpg",
    "test_images/img4.jpg",
    "test_images/img5.jpg",
    "test_images/img6.jpg",
]

# Load class mapping
import json
with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(dim=1).item()
    return idx_to_class[pred_idx]

# Run predictions
for file in test_files:
    label = predict(file)
    print(f"{file} -> {label}")