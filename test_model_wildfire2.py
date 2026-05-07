import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import ViTForImageClassification

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD MODEL
# ======================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

# Load the newly trained wildfire2 model
model.load_state_dict(torch.load("vit_wildfire2_model.pth", map_location=device))
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
# FOLDER PATH
# ======================
folder_path = "wildfire test data"
output_folder = "wildfire test results"

# The classes in wildfire2 dataset are "fire" and "nofire"
# ImageFolder sorts classes alphabetically, so 0: fire, 1: nofire
classes = ["fire", "nofire"]

os.makedirs(output_folder, exist_ok=True)

# ======================
# LOOP THROUGH IMAGES
# ======================
for filename in os.listdir(folder_path):

    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):

        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100

        # Print result
        print(f"{filename} -> {predicted_class} ({confidence_score:.2f}%)")

        # Save image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")

        color = "red" if predicted_class == "fire" else "green"
        plt.title(f"{predicted_class} ({confidence_score:.2f}%)", color=color)
        
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename} to {output_folder}")
