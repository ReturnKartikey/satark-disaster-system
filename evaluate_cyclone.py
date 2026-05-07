import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, accuracy_score
from tqdm import tqdm
from PIL import ImageFile

# Fix corrupted images issue
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# DEVICE CONFIGURATION
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# DATA LOADING (MATCHING TRAINING SPLIT)
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_path = "cyclone/binary_data/binary_data"

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset directory '{dataset_path}' not found!")
    exit()

full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# Use the EXACT same split logic and seed as train_vit_cyclone.py
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Seed 42 ensures we get the exact same "unseen" images for testing
_, _, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

classes = full_dataset.classes
print(f"\n--- Evaluation Diagnostic ---")
print(f"Classes: {classes}")
print(f"Total Dataset size: {len(full_dataset)}")
print(f"Evaluating on Unseen Test Set: {len(test_dataset)} images")
print(f"---------------------------\n")

# ==============================
# MODEL SETUP
# ==============================
print("Loading model weights...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
model.load_state_dict(torch.load("vit_cyclone_model.pth", map_location=device))
model.to(device)
model.eval()

# ==============================
# EVALUATION LOOP
# ==============================
all_preds = []
all_labels = []
all_probs = []

print("Running evaluation on test set...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(outputs.logits, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities[:, 1].cpu().numpy())

# ==============================
# METRICS & PLOTS
# ==============================
accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

report = classification_report(all_labels, all_preds, target_names=classes)
print(f"\nFinal Evaluation Results (Unseen Data):\n{report}")
print(f"ROC AUC Score: {auc:.4f}")

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Cyclone Test Set Confusion Matrix (Total: {len(all_labels)})")
plt.savefig("cyclone_final_confusion_matrix.png")
print("Saved confusion matrix as 'cyclone_final_confusion_matrix.png'")

# Save Metrics Text
with open("cyclone_final_metrics.txt", "w") as f:
    f.write("CYCLONE EVALUATION (UNSEEN TEST SET)\n")
    f.write("====================================\n")
    f.write(f"Total Test Images: {len(all_labels)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {auc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
print("Saved metrics text as 'cyclone_final_metrics.txt'")

# ROC CURVE
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cyclone ROC Curve (Unseen Test Set)')
plt.legend(loc="lower right")
plt.savefig("cyclone_final_roc_curve.png")
print("Saved ROC curve as 'cyclone_final_roc_curve.png'")

