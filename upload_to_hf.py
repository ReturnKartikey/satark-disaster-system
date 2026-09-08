import os
from huggingface_hub import HfApi

# ============================================================
# CONFIGURATION
# ============================================================
# Replace with your Hugging Face username and Space name
REPO_ID = "username/space-name" 
# Go to https://huggingface.co/settings/tokens to create a "Write" token
HF_TOKEN = "your_hf_write_token_here" 

# ============================================================
# UPLOAD
# ============================================================
api = HfApi()

files_to_upload = [
    "vit_flood2_model.pth",
    "vit_wildfire2_model.pth",
    "vit_cyclone_model.pth"
]

print(f"Starting upload to Hugging Face Space: {REPO_ID}...")

for filename in files_to_upload:
    if os.path.exists(filename):
        print(f"Uploading {filename} (this might take a few minutes)...")
        try:
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="space",
                token=HF_TOKEN
            )
            print(f"Successfully uploaded {filename}!")
        except Exception as e:
            print(f"Error uploading {filename}: {e}")
    else:
        print(f"Warning: Local file '{filename}' not found. Skipping.")

print("Upload process completed!")
