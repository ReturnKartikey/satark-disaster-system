import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
# DATA LOADING
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Assuming validation folder is used for testing
val_dir = "wildfire2/the_wildfire_dataset_2n_version/val"
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

classes = val_dataset.classes
print("Classes:", classes)

# ==============================
# MODEL SETUP
# ==============================
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
model.load_state_dict(torch.load("vit_wildfire2_model.pth", map_location=device))
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
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs.logits, dim=1)
        
        _, predicted = torch.max(outputs.logits, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Store probability of the positive class (class 1)
        all_probs.extend(probabilities[:, 1].cpu().numpy())

# ==============================
# METRICS CALCULATION
# ==============================
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print("\n" + "="*30)
print(f"EVALUATION METRICS")
print("="*30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("="*30)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("wildfire2_confusion_matrix.png")
print("\nSaved confusion matrix plot as 'wildfire2_confusion_matrix.png'")

# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("wildfire2_roc_curve.png")
print("Saved ROC curve plot as 'wildfire2_roc_curve.png'")
