FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's layer cache
COPY requirements.txt /app/

# Install pip upgrade, CPU-only PyTorch/Torchvision (reduces image size from ~5GB to ~1.8GB),
# and the rest of the requirements.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose port (default for Hugging Face Spaces is 7860)
EXPOSE 7860

# Run with Gunicorn using shell form to dynamically respect the PORT environment variable
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 240 backend.app:app
