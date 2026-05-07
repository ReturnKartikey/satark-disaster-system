# SATARK: Disaster Management System

**SATARK** (Sensing and Tracking for Advanced Relief and Knowledge) is an AI-powered disaster management platform designed to detect and respond to natural disasters like **Floods, Cyclones, and Wildfires** using satellite imagery and Vision Transformers (ViT).

## 🚀 Features
- **Multi-Disaster Detection**: Real-time classification for Floods, Wildfires, and Cyclones.
- **Deep Learning Core**: Built using state-of-the-art Vision Transformers (ViT) from Hugging Face.
- **Interactive Web Interface**: A sleek, glassmorphism-inspired dashboard for uploading images and viewing predictions.
- **Resource Allocation**: Automatically finds the nearest emergency resources (Hospitals, NDRF units, Fire Stations) based on the disaster location.
- **Visual Analytics**: Performance metrics including Confusion Matrices and ROC Curves for model transparency.

## 🛠️ Tech Stack
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (Fetch API).
- **Backend**: Flask (Python), Flask-CORS.
- **AI/ML**: PyTorch, Torchvision, Transformers (ViT Base).
- **Data Handling**: Pillow, NumPy, Matplotlib.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/satark-disaster-system.git
   cd satark-disaster-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights**:
   The model weights (`.pth` files) are large and not included in this repository. Ensure you have the following files in the root directory:
   - `vit_flood2_model.pth`
   - `vit_wildfire2_model.pth`
   - `vit_cyclone_model.pth`

## 🖥️ Usage

Run the main project script:
```bash
python run_project.py
```
This will:
1. Check for required libraries.
2. Load the AI models.
3. Start the Flask server.
4. Automatically open your browser to `http://127.0.0.1:5000`.

## 📊 Model Performance
Model evaluations can be found in the root directory:
- `flood2_confusion_matrix.png`
- `cyclone_final_metrics.txt`
- `wildfire2_roc_curve.png`

## ⚖️ License
This project is part of a Minor/Major project submission. All rights reserved.
