import os
import io
import math
import random
import torch
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

# ============================================================
# App Setup — templates are in FRONTEND folder
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "FRONTEND")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=None)
CORS(app)

# ============================================================
# Model Configuration
# ============================================================
MODEL_CONFIG = {
    "flood": {
        "path": os.path.join(PROJECT_ROOT, "vit_flood2_model.pth"),
        "classes": ["No Flood", "Flood"],
    },
    "wildfire": {
        "path": os.path.join(PROJECT_ROOT, "vit_wildfire2_model.pth"),
        "classes": ["fire", "nofire"],
    },
    "cyclone": {
        "path": os.path.join(PROJECT_ROOT, "vit_cyclone_model.pth"),
        "classes": ["No Cyclone", "Cyclone"],
    },
}

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SATARK] Using device: {device}")

# ============================================================
# Image transform (same as training pipeline)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# Load all models at startup
# ============================================================
models = {}

for name, cfg in MODEL_CONFIG.items():
    print(f"[SATARK] Loading {name} model from {cfg['path']} ...")
    if not os.path.exists(cfg["path"]):
        print(f"[SATARK] WARNING: Model file not found: {cfg['path']}. Skipping.")
        continue

    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
        )
        model.load_state_dict(torch.load(cfg["path"], map_location=device))
        model.to(device)
        model.eval()
        models[name] = model
        print(f"[SATARK] [OK] {name} model loaded successfully.")
    except Exception as e:
        print(f"[SATARK] [ERROR] Failed to load {name} model: {e}")

print(f"\n[SATARK] Models ready: {list(models.keys())}")


# ============================================================
# Prediction helper
# ============================================================
def predict_image(model, image_bytes, classes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)
    return predicted_class, confidence_score


# ============================================================
# PAGE ROUTES — Flask serves the HTML pages
# ============================================================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/flood")
def flood_page():
    return render_template("flood.html")

@app.route("/wildfire")
def wildfire_page():
    return render_template("wildfire.html")

@app.route("/cyclone")
def cyclone_page():
    return render_template("cyclone.html")

@app.route("/resources")
def resources_page():
    return render_template("resources.html")


# ============================================================
# API ROUTES — Prediction endpoints
# ============================================================
@app.route("/api/predict/<disaster_type>", methods=["POST"])
def predict(disaster_type):
    if disaster_type not in models:
        return jsonify({"error": f"Model '{disaster_type}' not loaded."}), 404

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        classes = MODEL_CONFIG[disaster_type]["classes"]
        predicted_class, confidence = predict_image(models[disaster_type], image_bytes, classes)

        return jsonify({
            "model": disaster_type,
            "prediction": predicted_class,
            "confidence": confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "models_loaded": list(models.keys()),
    })


# ============================================================
# Resource Allocation API
# ============================================================
RESOURCE_DB = [
    {"name": "AIIMS Trauma Centre", "type": "hospital", "lat": 28.5672, "lng": 77.2100},
    {"name": "Safdarjung Hospital", "type": "hospital", "lat": 28.5685, "lng": 77.2065},
    {"name": "Apollo Hospital Delhi", "type": "hospital", "lat": 28.5535, "lng": 77.2588},
    {"name": "Max Hospital Noida", "type": "hospital", "lat": 28.5740, "lng": 77.3562},
    {"name": "Fortis Hospital Gurgaon", "type": "hospital", "lat": 28.4400, "lng": 77.0423},
    {"name": "KEM Hospital Mumbai", "type": "hospital", "lat": 19.0003, "lng": 72.8417},
    {"name": "Lilavati Hospital Mumbai", "type": "hospital", "lat": 19.0509, "lng": 72.8289},
    {"name": "PGIMER Chandigarh", "type": "hospital", "lat": 30.7640, "lng": 76.7756},
    {"name": "CMC Vellore", "type": "hospital", "lat": 12.9249, "lng": 79.1338},
    {"name": "NIMHANS Bangalore", "type": "hospital", "lat": 12.9429, "lng": 77.5963},
    {"name": "NDRF Battalion 1 - Ghaziabad", "type": "ndrf", "lat": 28.6692, "lng": 77.4538},
    {"name": "NDRF Battalion 2 - Kolkata", "type": "ndrf", "lat": 22.5726, "lng": 88.3639},
    {"name": "NDRF Battalion 3 - Mundali", "type": "ndrf", "lat": 20.4050, "lng": 85.8830},
    {"name": "NDRF Battalion 4 - Arakkonam", "type": "ndrf", "lat": 13.0769, "lng": 79.6715},
    {"name": "NDRF Battalion 5 - Pune", "type": "ndrf", "lat": 18.5204, "lng": 73.8567},
    {"name": "NDRF Battalion 8 - Guwahati", "type": "ndrf", "lat": 26.1445, "lng": 91.7362},
    {"name": "Delhi Ambulance Station A1", "type": "ambulance", "lat": 28.6280, "lng": 77.2190},
    {"name": "Delhi Ambulance Station A2", "type": "ambulance", "lat": 28.6508, "lng": 77.2334},
    {"name": "Noida Ambulance Unit", "type": "ambulance", "lat": 28.5355, "lng": 77.3910},
    {"name": "Mumbai Ambulance Central", "type": "ambulance", "lat": 19.0760, "lng": 72.8777},
    {"name": "Kolkata Ambulance HQ", "type": "ambulance", "lat": 22.5465, "lng": 88.3510},
    {"name": "Chennai Ambulance Unit", "type": "ambulance", "lat": 13.0827, "lng": 80.2707},
    {"name": "Bangalore Ambulance Unit", "type": "ambulance", "lat": 12.9716, "lng": 77.5946},
    {"name": "Hyderabad Ambulance Unit", "type": "ambulance", "lat": 17.3850, "lng": 78.4867},
    {"name": "Delhi Fire Station - Connaught Place", "type": "fire_station", "lat": 28.6315, "lng": 77.2167},
    {"name": "Delhi Fire Station - Rohini", "type": "fire_station", "lat": 28.7325, "lng": 77.1107},
    {"name": "Mumbai Fire Brigade HQ", "type": "fire_station", "lat": 18.9432, "lng": 72.8313},
    {"name": "Chennai Fire Station", "type": "fire_station", "lat": 13.0700, "lng": 80.2800},
    {"name": "Kolkata Fire Brigade", "type": "fire_station", "lat": 22.5640, "lng": 88.3530},
    {"name": "SDRF Delhi Rescue Team", "type": "rescue_team", "lat": 28.6100, "lng": 77.2300},
    {"name": "SDRF UP Rescue Unit", "type": "rescue_team", "lat": 28.5800, "lng": 77.3300},
    {"name": "SDRF Maharashtra", "type": "rescue_team", "lat": 19.0560, "lng": 72.8500},
    {"name": "SDRF West Bengal", "type": "rescue_team", "lat": 22.5800, "lng": 88.3700},
    {"name": "SDRF Tamil Nadu", "type": "rescue_team", "lat": 13.0600, "lng": 80.2500},
    {"name": "SDRF Karnataka", "type": "rescue_team", "lat": 12.9800, "lng": 77.5800},
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

@app.route("/api/allocate-resources", methods=["POST"])
def allocate_resources():
    data = request.get_json()
    if not data or "lat" not in data or "lng" not in data:
        return jsonify({"error": "lat and lng are required."}), 400

    lat = float(data["lat"])
    lng = float(data["lng"])
    severity = data.get("severity", "high")
    disaster_type = data.get("disaster_type", "flood")

    max_resources = {"low": 4, "medium": 6, "high": 9, "critical": 14}.get(severity, 8)
    search_radius = {"low": 80, "medium": 150, "high": 300, "critical": 500}.get(severity, 200)

    scored = []
    for r in RESOURCE_DB:
        dist = haversine(lat, lng, r["lat"], r["lng"])
        if dist <= search_radius:
            speed = 45
            eta = round((dist / speed) * 60)
            scored.append({
                "name": r["name"],
                "type": r["type"],
                "lat": r["lat"],
                "lng": r["lng"],
                "distance_km": round(dist, 1),
                "eta_minutes": max(eta, 3),
            })

    scored.sort(key=lambda x: x["distance_km"])
    selected = scored[:max_resources]

    return jsonify({
        "disaster_type": disaster_type,
        "severity": severity,
        "location": {"lat": lat, "lng": lng},
        "resources": selected,
        "total_available": len(scored),
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n[SATARK] ===========================================")
    print("[SATARK]  Starting SATARK Web Application")
    print("[SATARK]  Open http://localhost:5000 in your browser")
    print("[SATARK] ===========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
