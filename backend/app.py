import os
import io
import math
import time
import base64
import random
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "ReturnKartikey/satark-models")

# ============================================================
# App Setup — templates are in FRONTEND folder
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "FRONTEND")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=ASSETS_DIR, static_url_path='/assets')
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit to prevent OOM DOS

# ============================================================
# Image Transforms
# ============================================================
TRANSFORM_STANDARD = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

TRANSFORM_NORMALIZED = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ============================================================
# Model Configuration
# ============================================================
MODEL_CONFIG = {
    "flood": {
        "path": os.path.join(PROJECT_ROOT, "vit_flood2_model.pth"),
        "classes": ["No Flood", "Flood"],
        "transform": TRANSFORM_NORMALIZED,
    },
    "wildfire": {
        "path": os.path.join(PROJECT_ROOT, "Burn", "vit_wildfire_model.pth"),
        "classes": ["nofire", "fire"],
        "transform": TRANSFORM_STANDARD,
    },
    "cyclone": {
        "path": os.path.join(PROJECT_ROOT, "vit_cyclone_model.pth"),
        "classes": ["No Cyclone", "Cyclone"],
        "transform": TRANSFORM_STANDARD,
    },
}

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SATARK] Using device: {device}")

# ============================================================
# Grad-CAM Attention Heatmap Setup
# ============================================================
class ViTWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x).logits

def reshape_transform(tensor, height=14, width=14):
    res = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return res.transpose(2, 3).transpose(1, 2)

models = {}
cam_objects = {}
startup_errors = {}

def load_model_for(name):
    if name in models:
        return models[name]
    cfg = MODEL_CONFIG.get(name)
    if not cfg:
        return None

    model_path = cfg["path"]
    if not os.path.exists(model_path):
        alt_candidates = [
            os.path.join(PROJECT_ROOT, os.path.basename(model_path)),
            os.path.join(PROJECT_ROOT, "Burn", os.path.basename(model_path)),
        ]
        if name == "wildfire":
            alt_candidates.extend([
                os.path.join(PROJECT_ROOT, "vit_wildfire2_model.pth"),
                os.path.join(PROJECT_ROOT, "vit_wildfire_model.pth"),
            ])
        for alt in alt_candidates:
            if os.path.exists(alt):
                model_path = alt
                break

    if not os.path.exists(model_path) and HF_HUB_AVAILABLE:
        remote_filename = os.path.basename(cfg["path"])
        if name == "wildfire":
            remote_filename = "vit_wildfire_model.pth"
        print(f"[SATARK] Local weights not found. Downloading {remote_filename} from Hugging Face Model Hub ({HF_MODEL_REPO})...")
        hf_token = os.environ.get("HF_TOKEN") or None
        if hf_token and not hf_token.strip():
            hf_token = None
        downloaded = None
        try:
            downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=remote_filename, token=hf_token)
        except Exception as e1:
            try:
                downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=remote_filename, token=None)
            except Exception as e2:
                startup_errors[name] = f"Download failed: {e1} | {e2}"
                print(f"[SATARK] WARNING: Could not download {remote_filename} from HF Hub: {e1}")

        if downloaded and os.path.exists(downloaded):
            model_path = downloaded
            print(f"[SATARK] [OK] Downloaded {name} model weights to {model_path}")

    print(f"[SATARK] Loading {name} model from {model_path} ...")
    if not os.path.exists(model_path):
        err_msg = f"Model file not found: {model_path}"
        startup_errors[name] = err_msg
        print(f"[SATARK] WARNING: {err_msg}")
        return None

    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[name] = model
        print(f"[SATARK] [OK] {name} model loaded successfully.")

        if GRADCAM_AVAILABLE:
            try:
                wrapped = ViTWrapper(model).to(device)
                target_layers = [model.vit.encoder.layer[-1].layernorm_before]
                cam_objects[name] = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=reshape_transform)
                print(f"[SATARK] [OK] GradCAM initialized for {name}")
            except Exception as cam_e:
                print(f"[SATARK] GradCAM init failed for {name}: {cam_e}")
        return model
    except Exception as e:
        startup_errors[name] = f"Load error: {e}"
        print(f"[SATARK] [ERROR] Failed to load {name} model: {e}")
        return None

# Load models at startup
for disaster_key in MODEL_CONFIG:
    try:
        load_model_for(disaster_key)
    except Exception as e:
        startup_errors[disaster_key] = str(e)

print(f"\n[SATARK] Models ready: {list(models.keys())}")

def generate_heatmap(model_name, image, input_tensor):
    if not GRADCAM_AVAILABLE or model_name not in models:
        return None
    try:
        current_model = models[model_name]
        model_device = next(current_model.parameters()).device
        cam = cam_objects.get(model_name)
        # Re-wrap if device changed (e.g. during CPU fallback) or not yet created
        if cam is None or getattr(cam, "_target_device", None) != model_device:
            wrapped = ViTWrapper(current_model)
            target_layers = [current_model.vit.encoder.layer[-1].layernorm_before]
            cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=reshape_transform)
            cam._target_device = model_device
            cam_objects[model_name] = cam

        img_resized = image.resize((224, 224))
        rgb_float = np.float32(img_resized) / 255.0
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        viz = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        out_pil = Image.fromarray(viz)
        if hasattr(image, 'size') and out_pil.size != image.size:
            out_pil = out_pil.resize(image.size, Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[SATARK] GradCAM generation error for {model_name}: {e}")
        return None


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
# API ROUTES — Prediction & Telemetry
# ============================================================
@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({
        "flood": "/assets/samples/flood_1.jpg",
        "wildfire": "/assets/samples/fire_1.jpg",
        "cyclone": "/assets/samples/cyclone_1.jpg",
        "catalog": {
            "flood": {
                "positive": ["/assets/samples/flood_1.jpg", "/assets/samples/flood_2.jpg"],
                "negative": ["/assets/samples/non_flood_1.jpg", "/assets/samples/non_flood_2.jpg"]
            },
            "wildfire": {
                "positive": ["/assets/samples/fire_1.jpg", "/assets/samples/fire_2.jpg"],
                "negative": ["/assets/samples/non_fire_1.jpg", "/assets/samples/non_fire_2.jpg"]
            },
            "cyclone": {
                "positive": ["/assets/samples/cyclone_1.jpg", "/assets/samples/cyclone_2.jpg"],
                "negative": ["/assets/samples/non_cyclone_1.jpg", "/assets/samples/non_cyclone_2.jpg"]
            }
        }
    })

@app.route("/api/predict/<disaster_type>", methods=["POST"])
def predict(disaster_type):
    if disaster_type not in models:
        load_model_for(disaster_type)
    if disaster_type not in models:
        detail = startup_errors.get(disaster_type, "Model could not be initialized.")
        return jsonify({"error": f"Model '{disaster_type}' not loaded. Details: {detail}"}), 404

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image file is empty."}), 400

    try:
        raw_image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "Invalid or corrupted image format. Please upload a standard JPEG or PNG image."}), 400

    try:
        t0 = time.perf_counter()
        cfg = MODEL_CONFIG[disaster_type]
        classes = cfg["classes"]
        img_transform = cfg.get("transform", TRANSFORM_STANDARD)
        try:
            input_tensor = img_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = models[disaster_type](input_tensor)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        except RuntimeError as rt_err:
            if "CUDA" in str(rt_err) or "cuda" in str(rt_err):
                print(f"[SATARK] CUDA error during inference, falling back to CPU: {rt_err}")
                cpu_device = torch.device("cpu")
                models[disaster_type] = models[disaster_type].to(cpu_device)
                input_tensor = img_transform(image).unsqueeze(0).to(cpu_device)
                with torch.no_grad():
                    outputs = models[disaster_type](input_tensor)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
            else:
                raise rt_err

        predicted_class = classes[predicted.item()]
        confidence_score = round(confidence.item() * 100, 2)

        # Dual class probabilities breakdown
        probs_dict = {classes[i]: round(probabilities[0][i].item() * 100, 2) for i in range(len(classes))}

        # Generate Explainable AI Heatmap (Grad-CAM)
        heatmap_data = generate_heatmap(disaster_type, image, input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_elapsed = round((time.perf_counter() - t0) * 1000, 1)
        active_device = "CPU" if input_tensor.device.type == "cpu" else str(device).upper()

        return jsonify({
            "model": disaster_type,
            "prediction": predicted_class,
            "confidence": confidence_score,
            "probabilities": probs_dict,
            "heatmap": heatmap_data,
            "latency_ms": t_elapsed,
            "device": active_device,
            "backbone": "ViT-B/16 (86.4M Params)"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "models_loaded": list(models.keys()),
        "cam_ready": list(cam_objects.keys()),
        "startup_errors": startup_errors,
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
    {"name": "AIIMS Rishikesh Trauma Centre", "type": "hospital", "lat": 30.0760, "lng": 78.2880},
    {"name": "SDRF Uttarakhand - Jolly Grant", "type": "rescue_team", "lat": 30.1890, "lng": 78.1800},
    {"name": "Dehradun City Fire Station", "type": "fire_station", "lat": 30.3165, "lng": 78.0322},
    {"name": "SMS Hospital Jaipur", "type": "hospital", "lat": 26.8918, "lng": 75.8164},
    {"name": "Jaipur Central Fire Station", "type": "fire_station", "lat": 26.9220, "lng": 75.7788},
    {"name": "NDRF Battalion 6 - Vadodara", "type": "ndrf", "lat": 22.3072, "lng": 73.1812},
    {"name": "Civil Hospital Ahmedabad", "type": "hospital", "lat": 23.0525, "lng": 72.5934},
    {"name": "Ahmedabad Fire HQ", "type": "fire_station", "lat": 23.0225, "lng": 72.5714},
    {"name": "AIIMS Bhubaneswar", "type": "hospital", "lat": 20.2312, "lng": 85.7758},
    {"name": "Bhubaneswar Fire Station", "type": "fire_station", "lat": 20.2961, "lng": 85.8245},
    {"name": "KGMU Trauma Centre Lucknow", "type": "hospital", "lat": 26.8690, "lng": 80.9160},
    {"name": "Lucknow Fire Station", "type": "fire_station", "lat": 26.8467, "lng": 80.9462},
    {"name": "AIIMS Bhopal Trauma Centre", "type": "hospital", "lat": 23.2065, "lng": 77.4601},
    {"name": "Bhopal Central Fire Brigade", "type": "fire_station", "lat": 23.2599, "lng": 77.4126},
    {"name": "Guwahati Medical College Hospital", "type": "hospital", "lat": 26.1584, "lng": 91.7725},
    {"name": "Assam State Fire Service HQ", "type": "fire_station", "lat": 26.1850, "lng": 91.7539},
    {"name": "Aster Medcity Kochi", "type": "hospital", "lat": 10.0520, "lng": 76.2730},
    {"name": "Ernakulam Fire Station", "type": "fire_station", "lat": 9.9816, "lng": 76.2999},
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
        return jsonify({"error": "Latitude ('lat') and longitude ('lng') are required."}), 400

    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (ValueError, TypeError):
        return jsonify({"error": "Latitude and longitude must be valid numeric floating-point values."}), 400

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return jsonify({"error": "Coordinates out of bounds. Latitude must be between -90 and 90, Longitude between -180 and 180."}), 400

    severity = str(data.get("severity", "high")).lower()
    disaster_type = str(data.get("disaster_type", "flood")).lower()

    max_resources = {"low": 4, "medium": 6, "high": 9, "critical": 14}.get(severity, 8)
    search_radius = {"low": 80, "medium": 150, "high": 300, "critical": 500}.get(severity, 200)

    # Dynamic type prioritization weights (lower weight ranks higher for that disaster)
    TYPE_WEIGHTS = {
        "wildfire": {"fire_station": 0.5, "ndrf": 0.75, "ambulance": 0.95, "hospital": 1.0, "rescue_team": 1.1},
        "flood": {"rescue_team": 0.5, "ndrf": 0.65, "ambulance": 0.9, "hospital": 1.0, "fire_station": 1.3},
        "cyclone": {"ndrf": 0.55, "rescue_team": 0.7, "hospital": 0.8, "ambulance": 0.9, "fire_station": 1.2}
    }
    active_weights = TYPE_WEIGHTS.get(disaster_type, {})

    scored = []
    all_scored = []
    for r in RESOURCE_DB:
        dist = haversine(lat, lng, r["lat"], r["lng"])
        weight = active_weights.get(r["type"], 1.0)
        effective_score = dist * weight
        speed = 45
        eta = round((dist / speed) * 60)
        entry = {
            "name": r["name"],
            "type": r["type"],
            "lat": r["lat"],
            "lng": r["lng"],
            "distance_km": round(dist, 1),
            "effective_score": effective_score,
            "eta_minutes": max(eta, 3),
        }
        all_scored.append(entry)
        if dist <= search_radius:
            scored.append(entry)

    # Sort prioritizing relevant hazard unit types
    scored.sort(key=lambda x: x["effective_score"])
    if len(scored) < 4:
        all_scored.sort(key=lambda x: x["effective_score"])
        scored = all_scored[:max(4, max_resources)]

    selected = scored[:max_resources]
    # Remove internal effective_score from public payload
    for item in selected:
        item.pop("effective_score", None)

    closest_dist = selected[0]["distance_km"] if selected else 0
    is_domestic = closest_dist <= 1500

    return jsonify({
        "disaster_type": disaster_type,
        "severity": severity,
        "location": {"lat": lat, "lng": lng},
        "resources": selected,
        "total_available": len(scored),
        "is_domestic": is_domestic,
        "notice": None if is_domestic else "Incident coordinates are outside the national emergency corridor. Local regional units must be mobilized."
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n[SATARK] ===========================================")
    print("[SATARK]  Starting SATARK Web Application")
    port = int(os.environ.get("PORT", 5000))
    print(f"[SATARK]  Open http://localhost:{port} in your browser")
    print("[SATARK] ===========================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
