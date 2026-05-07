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

# Load the trained floodnet model weights
model_path = "vit_floodnet_model.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: {model_path} not found. Please ensure you run your training script (train_vit_floodnet.py) to generate this file first!")

model.to(device)
model.eval()

# ======================
# IMAGE TRANSFORMS
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ======================
# TARGET FOLDER
# ======================
folder_path = r"flood test data"

classes = ["No Flood", "Flood"]

# ======================
# RUN INFERENCE
# ======================
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
else:
    print(f"Scanning '{folder_path}' for test images...")
    
    # Optional limit so it doesn't open hundreds of images at once
    test_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))]
    MAX_IMAGES_TO_SHOW = 10
    
    for filename in test_files[:MAX_IMAGES_TO_SHOW]:
        image_path = os.path.join(folder_path, filename)
        try:
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
            
            # Show image with matplotlib
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis("off")
            
            color = "red" if predicted_class == "Flood" else "green"
            plt.title(f"{predicted_class} ({confidence_score:.2f}%)", color=color, fontweight="bold")
            plt.show()
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    if len(test_files) > MAX_IMAGES_TO_SHOW:
        print(f"\n... Stopped after opening {MAX_IMAGES_TO_SHOW} images to prevent visual overload.")
        print(f"There are {len(test_files) - MAX_IMAGES_TO_SHOW} more images in the test folder.")
