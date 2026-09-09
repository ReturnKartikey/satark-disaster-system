FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HF_HOME=/tmp/huggingface

WORKDIR /code

# Install OpenCV runtime dependencies and CA certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch and Torchvision directly from official PyTorch CPU wheel repo
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Create non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user && \
    mkdir -p /tmp/huggingface && \
    chown -R user:user /tmp/huggingface

# Copy application source code and grant full access to user
COPY . /code
RUN chown -R user:user /code

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "backend.app:app"]
