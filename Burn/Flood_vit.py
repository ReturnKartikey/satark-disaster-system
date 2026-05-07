import os
import json
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from torch.optim import AdamW
from PIL import Image

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FloodDataset(Dataset):
    def __init__(self, root_dir, list_ds='S2list.json', transform=None):
        """
        Args:
            root_dir (string): Directory with all the tiles (SEN12FLOOD).
            list_ds (string): JSON list to read labels from.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        json_path = os.path.join(root_dir, list_ds)
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.tile_ids = list(self.metadata.keys())
        self.valid_data = []
        
        # Pre-filter to only include tiles with all 3 RGB bands present
        for t_id in self.tile_ids:
            tile_folder = os.path.join(self.root_dir, t_id)
            if not os.path.exists(tile_folder):
                continue
            
            # Look for B04 (Red), B03 (Green), B02 (Blue)
            files = os.listdir(tile_folder)
            bands = {"B04": None, "B03": None, "B02": None}
            
            for f in files:
                if 'B04.tif' in f: bands["B04"] = f
                elif 'B03.tif' in f: bands["B03"] = f
                elif 'B02.tif' in f: bands["B02"] = f
                
            if all(bands.values()):
                # Check for label
                label = self.metadata[t_id].get('FLOODING', False)
                self.valid_data.append({
                    'id': t_id,
                    'bands': bands,
                    'label': 1 if label else 0
                })

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        tile_folder = os.path.join(self.root_dir, item['id'])
        
        # Read the TIFF imagery
        b4_path = os.path.join(tile_folder, item['bands']['B04'])
        b3_path = os.path.join(tile_folder, item['bands']['B03'])
        b2_path = os.path.join(tile_folder, item['bands']['B02'])
        
        b4 = tiff.imread(b4_path)
        b3 = tiff.imread(b3_path)
        b2 = tiff.imread(b2_path)
        
        # Stack as RGB (H, W, C)
        img = np.stack([b4, b3, b2], axis=-1)
        
        # Normalize to 0-255 uint8 for PIL
        img_max = img.max()
        if img_max > 0:
            img = (img / img_max * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert to PIL Image for torchvision transforms
        pil_img = Image.fromarray(img)
        
        # Label
        label = item['label']
        
        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, torch.tensor(label, dtype=torch.long)

def main():
    print("Loading valid tiles...")
    # Define torchvision transforms appropriate for ViT-base-patch16-224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset_path = 'flood/SEN12FLOOD'
    dataset = FloodDataset(root_dir=dataset_path, transform=transform)
    
    print(f"Total valid tiles found: {len(dataset)}")
    if len(dataset) == 0:
        print("No valid tiles found. Exiting.")
        return

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing Model...")
    # Output labels: [0: No Flood, 1: Flood]
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=2,
        id2label={0: "No Flood", 1: "Flood"},
        label2id={"No Flood": 0, "Flood": 1}
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Huggingface models return a SequenceClassifierOutput
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%\n")

    # Save Model
    save_path = "vit_flood_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print("Training Complete!")

if __name__ == "__main__":
    main()
