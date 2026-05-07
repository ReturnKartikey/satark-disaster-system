import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
from PIL import ImageFile

# ==============================
# FIX CORRUPTED IMAGES ISSUE
# ==============================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# PARAMETERS
# ==============================
BATCH_SIZE = 16   # If GPU memory error → change to 8
EPOCHS = 5
LR = 2e-5

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ==============================
# LOAD AND SPLIT DATASET
# ==============================
# The cyclone binary data is in "cyclone/binary_data/binary_data"
dataset_path = "cyclone/binary_data/binary_data"
full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# 80% Train, 10% Validation, 10% Test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Total images:", len(full_dataset))
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ==============================
# LOAD PRETRAINED VIT
# ==============================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

model.to(device)

# ==============================
# LOSS & OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # TRAINING
    model.train()
    train_correct = 0
    train_total = 0

    train_loop = tqdm(train_loader)

    for images, labels in train_loop:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.logits, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    train_acc = 100 * train_correct / train_total
    print(f"Train Accuracy: {train_acc:.2f}%")

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# ==============================
# TESTING LOOP
# ==============================
print("\n--- Testing Model ---")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)

        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "vit_cyclone_model.pth")
print("\nModel saved successfully as 'vit_cyclone_model.pth'!")
