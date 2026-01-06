# Multi-stage build for WhisperX-FastAPI
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=120

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Build argument to control GPU installation
ARG INSTALL_GPU=false

COPY pyproject.toml uv.lock ./

# Install dependencies based on GPU flag
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        echo "Installing with GPU support..." && \
        uv sync --no-dev --extra gpu; \
    else \
        echo "Installing CPU-only version..." && \
        uv sync --no-dev; \
    fi

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create cache directories
RUN mkdir -p /root/.cache/huggingface /root/.cache/torch

# Expose the port
EXPOSE 8000

# Create uploads directory
RUN mkdir -p /tmp/uploads

# Default command (can be overridden)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "app/uvicorn_log_conf.yaml", "--log-level", "info"]