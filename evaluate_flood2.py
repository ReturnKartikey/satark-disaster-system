import os
import json
import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, accuracy_score
from tqdm import tqdm
from PIL import Image

# ==============================
# DEVICE CONFIGURATION
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
# DATA SETUP
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

labels_dir = r"flood2\sen12floods_s2_labels\sen12floods_s2_labels"
source_dir = r"flood2\sen12floods_s2_source\sen12floods_s2_source"

dataset = Flood2Dataset(labels_dir=labels_dir, source_dir=source_dir, transform=transform)

# Evaluate on the validation split
# Using fixed seed to randomly generate the same validation split to test isolated unseen data if needed
torch.manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

eval_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
classes = ["No Flood", "Flood"]

# ==============================
# MODEL CONFIGURATION
# ==============================
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "No Flood", 1: "Flood"},
    label2id={"No Flood": 0, "Flood": 1}
)

model_path = "vit_flood2_model.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: {model_path} not found. Running evaluation with uninitialized random state.")

model.to(device)
model.eval()

# ==============================
# EVALUATION LOOP
# ==============================
all_preds = []
all_labels = []
all_probs = []

print("Running evaluation...")
with torch.no_grad():
    for images, labels in tqdm(eval_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs.logits, dim=1)
        
        _, predicted = torch.max(outputs.logits, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Store probability of the "Flood" class (class 1)
        all_probs.extend(probabilities[:, 1].cpu().numpy())

# ==============================
# METRICS CALCULATION
# ==============================
f1 = f1_score(all_labels, all_preds, zero_division=0)
precision = precision_score(all_labels, all_preds, zero_division=0)
accuracy = accuracy_score(all_labels, all_preds)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = 0.0  # In rare chance split lacks a class

print("\n" + "="*30)
print("EVALUATION METRICS")
print("="*30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("="*30)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Flood2)")
plt.savefig("flood2_confusion_matrix.png")
print("\nSaved confusion matrix plot as 'flood2_confusion_matrix.png'")

# ==============================
# ROC CURVE
# ==============================
try:
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Flood2)')
    plt.legend(loc="lower right")
    plt.savefig("flood2_roc_curve.png")
    print("Saved ROC curve plot as 'flood2_roc_curve.png'")
except Exception as e:
    print("Could not generate ROC curve:", e)
