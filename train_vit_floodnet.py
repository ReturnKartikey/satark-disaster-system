import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
class FloodNetDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        
        print("Loading dataset metadata...")
        
        # -------------------------------
        # 1. LOAD TRACK 1 DATA
        # -------------------------------
        track1_path = os.path.join("FloodNet Challenge", "FloodNet Challenge - Track 1", "Train", "Labeled")
        flooded_dir = os.path.join(track1_path, 'Flooded', 'image')
        non_flooded_dir = os.path.join(track1_path, 'Non-Flooded', 'image')
        
        if os.path.exists(flooded_dir):
            for filename in os.listdir(flooded_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.data.append({
                        'filepath': os.path.join(flooded_dir, filename),
                        'label': 1
                    })
        
        if os.path.exists(non_flooded_dir):
            for filename in os.listdir(non_flooded_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.data.append({
                        'filepath': os.path.join(non_flooded_dir, filename),
                        'label': 0
                    })

        # -------------------------------
        # 2. LOAD TRACK 2 DATA
        # -------------------------------
        track2_base = os.path.join("FloodNet Challenge", "FloodNet Challenge - Track 2")
        track2_subsets = [
            (
                os.path.join(track2_base, "Images", "Train_Image"),
                os.path.join(track2_base, "Questions", "Training Question.json")
            ),
            (
                os.path.join(track2_base, "Images", "Valid_Image"),
                os.path.join(track2_base, "Questions", "Valid Question.json")
            )
        ]

        for img_dir, json_path in track2_subsets:
            if os.path.exists(json_path) and os.path.exists(img_dir):
                try:
                    with open(json_path, 'r') as f:
                        qa_data = json.load(f)
                    
                    labeled_t2 = set() # To prevent adding duplicate images due to multiple questions
                    for key, item in qa_data.items():
                        if item.get("Question_Type") == "Condition_Recognition":
                            gt = str(item.get("Ground_Truth", "")).strip().lower()
                            img_id = item.get("Image_ID")
                            
                            if img_id and img_id not in labeled_t2:
                                img_path = os.path.join(img_dir, img_id)
                                if os.path.exists(img_path):  # ensure the image file actually exists
                                    if gt == "flooded":
                                        self.data.append({'filepath': img_path, 'label': 1})
                                        labeled_t2.add(img_id)
                                    elif gt == "non flooded":
                                        self.data.append({'filepath': img_path, 'label': 0})
                                        labeled_t2.add(img_id)
                except Exception as e:
                    print(f"Warning: Failed to parse Track 2 data in {json_path}: {e}")

        # Summary logging
        total_flooded = sum(1 for item in self.data if item['label'] == 1)
        total_non_flooded = sum(1 for item in self.data if item['label'] == 0)
        print(f"Total merged images: {len(self.data)}")
        print(f" -> Flooded (1): {total_flooded}")
        print(f" -> Non-Flooded (0): {total_non_flooded}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['filepath']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupt images
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = item['label']
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

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
dataset = FloodNetDataset(transform=transform)

# Evaluate if we have data to split
if len(dataset) == 0:
    print("Error: No images found. Check your dataset paths.")
    exit(1)

# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")

# ==============================
# MODEL DEFINITION
# ==============================
print("Loading model architecture...")
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
model_path = "vit_floodnet_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved successfully as {model_path}!")
