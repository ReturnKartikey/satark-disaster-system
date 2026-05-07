import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import ViTForImageClassification

# ======================
# DEVICE CONFIGURATION
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD MODEL
# ======================
print("Loading model architecture...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "No Flood", 1: "Flood"},
    label2id={"No Flood": 0, "Flood": 1}
)

# Load the trained flood2 model weights
model_path = "vit_flood2_model.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: {model_path} not found. Please ensure you have finished training the model.")
    print("Running with uninitialized classification head weights for testing purposes.")

model.to(device)
model.eval()

# ======================
# IMAGE TRANSFORMS
# ======================
# The transform matching what was used in train_vit_flood2.py padding/resizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ======================
# TARGET FOLDER
# ======================
folder_path = "flood test data"
output_folder = "flood test results"

# Output labels: [0: No Flood, 1: Flood]
classes = ["No Flood", "Flood"]

# ======================
# RUN INFERENCE
# ======================
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
else:
    print(f"Scanning '{folder_path}' for images...")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
            
            image_path = os.path.join(folder_path, filename)
            try:
                # Open image and ensure it has 3 RGB channels (ignores alpha/grayscale issues)
                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = classes[predicted.item()]
                confidence_score = confidence.item() * 100
                
                # Print result to console
                print(f"{filename} -> {predicted_class} ({confidence_score:.2f}%)")
                
                # Save image with matplotlib
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.axis("off")
                
                color = "red" if predicted_class == "Flood" else "green"
                plt.title(f"{predicted_class} ({confidence_score:.2f}%)", color=color, fontweight="bold")
                
                output_path = os.path.join(output_folder, filename)
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                print(f"Saved {filename} to {output_folder}")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
