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

model.load_state_dict(torch.load("vit_wildfire_model.pth", map_location=device))
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
folder_path = "random"

classes = ["nowildfire", "wildfire"]

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
        print(f"{filename} → {predicted_class} ({confidence_score:.2f}%)")

        # Show image
        plt.imshow(image)
        plt.axis("off")

        color = "red" if predicted_class == "wildfire" else "green"
        plt.title(f"{predicted_class} ({confidence_score:.2f}%)", color=color)
        plt.show()