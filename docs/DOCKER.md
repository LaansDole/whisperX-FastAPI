# Docker Setup Guide for WhisperX-FastAPI

This guide provides instructions for running WhisperX-FastAPI using Docker Compose for faster and more consistent setup.

## Prerequisites

1. **Docker & Docker Compose**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine with Compose
2. **NVIDIA Container Toolkit** (for GPU support): Follow [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. **Environment Configuration**: Copy and configure your environment file

## Quick Start

### 1. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your HF_TOKEN
nano .env
```

**Required environment variables:**
- `HF_TOKEN`: Your Hugging Face token for pyannote speaker diarization models

**Runtime defaults** are in `config.yaml` (model, language, device, etc.)

### 2. Choose Your Setup

#### CPU-Only Setup (Recommended for development/testing)

```bash
# Start all services (API, Worker, and Temporal server)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### GPU-Accelerated Setup (For production workloads)

```bash
# Start with GPU support
docker-compose -f docker-compose.gpu.yaml up -d

# View logs
docker-compose -f docker-compose.gpu.yaml logs -f

# Stop services
docker-compose -f docker-compose.gpu.yaml down
```

## Service Architecture

The Docker Compose setup includes these main services:

### 1. `temporal` - Workflow Orchestration Server
- **Purpose**: Manages workflow execution and state
- **Ports**: 
  - `7233`: gRPC API for client connections
  - `8233`: Web UI for monitoring workflows
- **Web UI**: Access at `http://localhost:8233`

### 2. `whisperx-api` - FastAPI Application
- **Purpose**: REST API endpoints for audio processing
- **Port**: `8000`
- **API Docs**: Access at `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 3. `whisperx-worker` - Temporal Worker
- **Purpose**: Executes audio processing workflows
- **Scaling**: Increase `replicas` in compose file for more workers

### 4. `streamlit-ui` - Demo Web Interface
- **Purpose**: Upload audio and view transcriptions
- **Port**: `8501`
- **Access**: `http://localhost:8501`

## Persistent Storage

The setup includes persistent volumes for:

- **`whisperx-huggingface-cache`**: Stores downloaded Whisper and other models
- **`whisperx-torch-cache`**: Stores PyTorch model cache
- **`temporal-data`**: Temporal server data and workflow state

This ensures models and data aren't re-downloaded/lost between container restarts.

## Configuration

### Environment Variables (.env)

Secrets and environment-specific settings only:

```bash
# Required
HF_TOKEN=your_huggingface_token_here

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Runtime Defaults (config.yaml)

All other settings are in `config.yaml` and can be overridden per-request:

```yaml
whisperx:
  model: base      # tiny, base, small, medium, large-v3
  language: en     # en, vi, zh, yue
  device: cpu      # cpu, cuda
  compute_type: int8  # int8, float16, float32
```

### GPU-Specific Overrides

For GPU setup, override defaults via environment:

```bash
DEVICE=cuda
COMPUTE_TYPE=float16
```

### Scaling Workers

To handle more concurrent requests, increase worker replicas:

```yaml
# In docker-compose.yaml under whisperx-worker service
deploy:
  replicas: 3  # Increase this number
```

## Common Commands

### Development Workflow

```bash
# Build images without cache
docker-compose build --no-cache

# Start in foreground (see logs directly)
docker-compose up

# Restart specific service
docker-compose restart whisperx-api

# View specific service logs
docker-compose logs -f whisperx-worker

# Execute commands inside running container
docker-compose exec whisperx-api bash
```

### Production Deployment

```bash
# GPU production setup
docker-compose -f docker-compose.gpu.yaml up -d

# Check service health
docker-compose ps

# Monitor resource usage
docker stats

# Update and restart services
docker-compose pull
docker-compose up -d
```

### Maintenance Commands

```bash
# Clean up stopped containers and unused images
docker-compose down --rmi local
docker system prune -f

# Backup model cache (optional)
docker run --rm -v whisperx-huggingface-cache:/data -v $(pwd):/backup alpine tar czf /backup/model-cache-backup.tar.gz -C /data .

# View volume usage
docker system df -v
```

## Monitoring and Debugging

### Health Checks

Each service includes health checks:

- **API**: `curl http://localhost:8000/health`
- **Temporal**: Available via Docker health status
- **Worker**: Monitored via Temporal UI

### Log Monitoring

```bash
# All services
docker-compose logs -f

# Specific service with timestamps
docker-compose logs -f -t whisperx-api

# Last N lines
docker-compose logs --tail=100 whisperx-worker
```

### Temporal Web UI

Access workflow monitoring at `http://localhost:8233`:
- View active and completed workflows
- Monitor worker status
- Debug failed executions
- Performance metrics

### Common Issues and Solutions

#### 1. Model Download Issues

**Problem**: Models fail to download
```bash
# Check HF_TOKEN is set correctly
docker-compose exec whisperx-api printenv HF_TOKEN

# Pre-download models
docker-compose exec whisperx-api python scripts/download_diarization_model.py
```

#### 2. GPU Not Detected

**Problem**: CUDA not available in GPU mode
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker-compose -f docker-compose.gpu.yaml exec whisperx-api python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Out of Memory Errors

**Solutions**:
- Use smaller model (`tiny`, `base` instead of `large`)
- Increase Docker memory limits
- Use `int8` compute type for CPU
- For GPU, try `float16` instead of `float32`

#### 4. Temporal Connection Issues

```bash
# Check temporal service status
docker-compose ps temporal

# Restart temporal
docker-compose restart temporal

# Check temporal logs
docker-compose logs temporal
```

## File Upload Handling

The setup includes an optional `./uploads` directory mount:

```bash
# Create local uploads directory
mkdir -p ./uploads

# Files uploaded via API are accessible in both API and worker containers
# at /tmp/uploads
```

## Performance Optimization

### CPU Optimization
- Use `int8` compute type
- Choose appropriate model size (`small` is good balance)
- Increase worker replicas for concurrency

### GPU Optimization
- Use `float16` compute type for balance of speed/accuracy
- Use `float32` for maximum accuracy
- Ensure sufficient GPU memory for model + batch processing
- Consider larger models (`medium`, `large`) for better accuracy

## Security Considerations

1. **Environment Variables**: Never commit `.env` files with real tokens
2. **Network Security**: Use reverse proxy (nginx) for production
3. **Container Security**: Run with non-root user in production
4. **API Security**: Implement authentication/authorization as needed