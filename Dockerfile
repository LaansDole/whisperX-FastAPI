# Multi-stage build for WhisperX-FastAPI
FROM python:3.11.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned version for supply-chain security)
RUN pip install uv==0.9.21

WORKDIR /app

# Build argument to control GPU installation
ARG INSTALL_GPU=false

# Install dependencies using bind mounts for better layer caching
# This layer is cached separately and only rebuilt when pyproject.toml or uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    if [ "$INSTALL_GPU" = "true" ]; then \
        echo "Installing dependencies with GPU support..." && \
        uv sync --frozen --no-install-project --no-dev --extra gpu; \
    else \
        echo "Installing dependencies (CPU-only)..." && \
        uv sync --frozen --no-install-project --no-dev; \
    fi

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY pyproject.toml uv.lock ./

# Install the project itself (fast since dependencies are already installed)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_GPU" = "true" ]; then \
        uv sync --frozen --no-dev --extra gpu; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Create cache directories
RUN mkdir -p /root/.cache/huggingface /root/.cache/torch

# Expose the port
EXPOSE 8000

# Create uploads directory
RUN mkdir -p /tmp/uploads

# Default command (can be overridden)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "app/uvicorn_log_conf.yaml", "--log-level", "info"]