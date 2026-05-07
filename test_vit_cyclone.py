import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from transformers import ViTForImageClassification

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD MODEL
# ======================
print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

# Load the trained cyclone model
model.load_state_dict(torch.load("vit_cyclone_model.pth", map_location=device))
model.to(device)
model.eval()

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# FOLDER PATHS
# ======================
# If you pass a folder as argument, it uses that. Otherwise defaults to cycloneTEST.
if len(sys.argv) < 2:
    input_folder = "cycloneTEST"
else:
    input_folder = sys.argv[1]

output_folder = "cyclone test results"
classes = ["No Cyclone", "Cyclone"]

os.makedirs(output_folder, exist_ok=True)

# ======================
# LOOP THROUGH IMAGES
# ======================
print(f"\nScanning folder: {input_folder}")
print("-" * 50)

# We use os.walk to handle subfolders (like 'Yes' and 'No')
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".webp")):

            image_path = os.path.join(root, filename)
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item() * 100

            # Print result to terminal
            print(f"{filename[:30]:<30} -> {predicted_class} ({confidence_score:.2f}%)")

            # Save labeled image using Matplotlib
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis("off")

            # Red for Cyclone, Green for No Cyclone
            color = "red" if predicted_class == "Cyclone" else "green"
            plt.title(f"{predicted_class} ({confidence_score:.2f}%)", color=color, fontsize=14, fontweight='bold')
            
            # Save to output folder (flattened naming to avoid subfolder issues)
            # e.g., Yes_image1.jpg
            parent_name = os.path.basename(root)
            save_name = f"{parent_name}_{filename}"
            output_path = os.path.join(output_folder, save_name)
            
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

print("-" * 50)
print(f"Done! Labeled images saved in: {output_folder}")
