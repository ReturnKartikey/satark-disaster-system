import os
import subprocess
import sys
import webbrowser
import time

def run_project():
    print("="*50)
    print("      SATARK: DISASTER MANAGEMENT SYSTEM")
    print("="*50)
    
    # 1. Check/Install dependencies
    print("\n[1/3] Checking dependencies...")
    try:
        import flask
        import torch
        import transformers
        print("Done.")
    except ImportError:
        print("Missing libraries. Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "torch", "transformers", "pillow", "numpy"])

    # 2. Get the path to app.py
    # Assuming the root is C:\Users\Karti\Desktop\MINORMID\minor_project
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "backend", "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        return

    # 3. Start the server
    print("\n[2/3] Starting SATARK Server...")
    print("Please wait for models to load (this may take a minute)...")
    
    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(5) # Wait for server to initialize
        print("\n[3/3] Opening SATARK in your browser...")
        webbrowser.open("http://127.0.0.1:5000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run the Flask app
    try:
        os.chdir(os.path.join(current_dir, "backend"))
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\nProject stopped by user.")

if __name__ == "__main__":
    run_project()
