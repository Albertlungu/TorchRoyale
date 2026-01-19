import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


train_dataset = datasets.ImageFolder(
    "../../data/all_data/", transform=transform
)  # PyTorch uses the name of the folder that the image is inside of as the label
use_pin_mem = True if device.type == "cuda" else False  # Pin memory is for CUDA only
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=use_pin_mem
)

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
num_classes = len(train_dataset.classes)  # Number of folders in the training data.
model.fc = nn.Linear(model.fc.in_features, num_classes)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device).float(), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
