# PINN Thermal Analysis Framework Docker Image
# Multi-stage build for production deployment

# ===== Stage 1: Builder =====
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Stage 2: Production =====
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/

# Create directories for data and outputs
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embeddings \
    /app/outputs/checkpoints /app/outputs/logs /app/outputs/reports \
    /app/models

# Environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Default command (Streamlit app)
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Labels
LABEL maintainer="your-email@example.com"
LABEL version="2.0"
LABEL description="PINN Thermal Analysis Framework - Physics-Informed Neural Networks for Heat Transfer Simulation"
LABEL features="geometry-lifetime,melting-analysis,thermal-limits,hotspot-detection"
