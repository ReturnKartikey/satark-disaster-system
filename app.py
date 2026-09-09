"""
SATARK Web Application Entry Point.
Re-exports the core Flask application from backend.app to prevent duplicate logic and code drift.
"""
import os
import sys

# Ensure current directory is on sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from backend.app import app

if __name__ == "__main__":
    print("\n[SATARK] ===========================================")
    print("[SATARK]  Starting SATARK Web Application")
    port = int(os.environ.get("PORT", 5000))
    print(f"[SATARK]  Open http://localhost:{port} in your browser")
    print("[SATARK] ===========================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
