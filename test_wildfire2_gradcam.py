import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from transformers import ViTForImageClassification

# Try importing GradCAM from pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Please install grad-cam: pip install grad-cam")
    exit()

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
model.eval()

# ======================
# WRAPPER & GRAD-CAM SETUP
# ======================
# pytorch-grad-cam expects the model's forward to return a tensor of shape (batch, logits)
class HuggingfaceViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceViTWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x).logits

wrapped_model = HuggingfaceViTWrapper(model)
wrapped_model.to(device)

def reshape_transform(tensor, height=14, width=14):
    # The output from the ViT encoder layer is [batch, seq_len, hidden_dim]
    # seq_len = 1 (CLS) + 14*14 (patches)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Bring the channels to the first dimension, like in CNNs
    # [batch, height, width, hidden_dim] -> [batch, hidden_dim, height, width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# We target the layernorm before the final MLP layer 
target_layers = [model.vit.encoder.layer[-1].layernorm_before]

# Initialize GradCAM
cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

# ======================
# TRANSFORM
# ======================
# For PyTorch models, input must be normalized if it was normalized during training.
# We didn't use ImageNet mean/std during training in train_vit_wildfire2.py, so we just use ToTensor.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# FOLDER PATH
# ======================
folder_path = "random"
classes = ["fire", "nofire"]

# ======================
# LOOP THROUGH IMAGES
# ======================
print(f"Reading images from {folder_path}...")
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        
        # Resize original for visualization
        img_resized = image.resize((224, 224))
        # Convert to float [0, 1] for show_cam_on_image
        rgb_img = np.float32(img_resized) / 255.0  

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100

        # Run Grad-CAM
        # target_category=None will automatically use the highest scoring class
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        # We need the first item in the batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay heatmap on image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Print result
        print(f"{filename} -> {predicted_class} ({confidence_score:.2f}%)")

        # Plot side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original Image
        axs[0].imshow(img_resized)
        axs[0].axis('off')
        axs[0].set_title('Original Image')
        
        # Heatmap Image
        color = "red" if predicted_class == "fire" else "green"
        axs[1].imshow(cam_image)
        axs[1].axis('off')
        axs[1].set_title(f'Grad-CAM: {predicted_class} ({confidence_score:.2f}%)', color=color)
        
        plt.tight_layout()
        plt.show()
