import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
from PIL import Image

# ==============================
# DEVICE CONFIGURATION
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# HYPERPARAMETERS
# ==============================
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

# ==============================
# DATASET DEFINITION
# ==============================
class Flood2Dataset(Dataset):
    def __init__(self, labels_dir, source_dir, transform=None):
        self.labels_dir = labels_dir
        self.source_dir = source_dir
        self.transform = transform
        self.valid_data = []
        
        print("Loading dataset metadata...")
        for label_folder in os.listdir(labels_dir):
            if not label_folder.startswith('sen12floods_s2_labels_'):
                continue
            
            suffix = label_folder[len('sen12floods_s2_labels_'):]
            source_folder = f"sen12floods_s2_source_{suffix}"
            
            label_path = os.path.join(labels_dir, label_folder, 'labels.geojson')
            source_folder_path = os.path.join(source_dir, source_folder)
            
            if os.path.exists(label_path) and os.path.exists(source_folder_path):
                try:
                    with open(label_path, 'r') as f:
                        data = json.load(f)
                    is_flooded = data.get('properties', {}).get('FLOODING', False)
                except Exception:
                    continue
                
                label = 1 if is_flooded else 0
                
                files = os.listdir(source_folder_path)
                bands = {"B04": None, "B03": None, "B02": None}
                for f in files:
                    if 'B04.tif' in f: bands["B04"] = f
                    elif 'B03.tif' in f: bands["B03"] = f
                    elif 'B02.tif' in f: bands["B02"] = f
                    
                if all(bands.values()):
                    self.valid_data.append({
                        'source_folder': source_folder_path,
                        'bands': bands,
                        'label': label
                    })
        
        print(f"Total valid tiles found: {len(self.valid_data)}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        b4_path = os.path.join(item['source_folder'], item['bands']['B04'])
        b3_path = os.path.join(item['source_folder'], item['bands']['B03'])
        b2_path = os.path.join(item['source_folder'], item['bands']['B02'])
        
        b4 = tiff.imread(b4_path)
        b3 = tiff.imread(b3_path)
        b2 = tiff.imread(b2_path)
        
        img = np.stack([b4, b3, b2], axis=-1)
        
        img_max = img.max()
        if img_max > 0:
            img = (img / img_max * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        pil_img = Image.fromarray(img)
        label = item['label']
        
        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, torch.tensor(label, dtype=torch.long)

# ==============================
# PREPROCESSING TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ==============================
# LOAD DATA & SPLIT
# ==============================
labels_dir = r"flood2\sen12floods_s2_labels\sen12floods_s2_labels"
source_dir = r"flood2\sen12floods_s2_source\sen12floods_s2_source"

dataset = Flood2Dataset(labels_dir=labels_dir, source_dir=source_dir, transform=transform)

# 80% train, 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# ==============================
# MODEL DEFINITION
# ==============================
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "No Flood", 1: "Flood"},
    label2id={"No Flood": 0, "Flood": 1}
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ==============================
# TRAINING LOOP
# ==============================
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

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

    print("Running validation...")
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
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "vit_flood2_model.pth")
print("\nModel saved successfully as vit_flood2_model.pth!")
