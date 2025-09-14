# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Core Architecture

WhisperX-FastAPI is a production-ready speech processing API built with FastAPI and Temporal workflow orchestration. The system processes audio/video files through multiple stages: transcription, alignment, diarization, and speaker assignment.

### Key Components

- **FastAPI Application**: RESTful API with automatic OpenAPI documentation
- **Temporal Workflows**: Orchestrates complex audio processing pipelines with retry policies
- **WhisperX Processing Engine**: Core audio processing using faster-whisper, pyannote, and wav2vec2
- **Model Manager**: Handles automatic model downloads and caching from Hugging Face Hub
- **Optional RAG Chatbot**: Advanced agentic RAG system (planned with LangGraph + Graphiti)

### Architecture Layers

1. **API Layer**: FastAPI routers for STT, services, tasks, and health endpoints
2. **Workflow Layer**: Temporal client, activities, and workflow management
3. **Processing Layer**: WhisperX services for transcription, alignment, and diarization
4. **Storage Layer**: Model cache (~/.cache/huggingface, ~/.cache/torch), temporary file storage

## Development Commands

### Environment Setup
```bash
# Install production dependencies
make install-prod

# Install development dependencies (includes ruff, pytest, pre-commit)
make install-dev

# Create environment file
cp .env.example .env
```

### Running the Application
```bash
# Start Temporal server locally (requires temporal CLI installed)
make run-temporal-local

# Start FastAPI server with hot reload
make run-local

# Start Temporal worker (in separate terminal)
make run-worker-local

# Stop Temporal server
make stop-temporal-local
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_all.py

# Run tests with coverage
pytest --cov=app
```

### Code Quality
```bash
# Format and lint code
ruff check . --fix
ruff format .

# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### Environment Variables
Key environment variables in `.env`:
- `HF_TOKEN`: Hugging Face token for model downloads
- `WHISPER_MODEL`: Default model (tiny, base, small, medium, large variants)
- `DEVICE`: Processing device (cuda/cpu)
- `COMPUTE_TYPE`: Precision (float16/float32/int8)
- `TEMPORAL_*`: Temporal server configuration
- `*_TIMEOUT`: Activity timeout configurations

### Model Management
- Models are automatically downloaded and cached on first use
- Cache locations: `~/.cache/huggingface/hub` and `~/.cache/torch`
- Diarization model can be pre-downloaded: `python scripts/download_diarization_model.py`
- Supported formats: Audio (.mp3, .wav, .aac, .ogg, .m4a, .wma, .amr, .awb, .oga) and Video (.mp4, .mov, .avi, .wmv, .mkv)

### Temporal Workflow System
- Task queue: `whisperx-task-queue`
- Activities: Transcription, Alignment, Diarization, Speaker Assignment
- Retry policies: Different policies for model loading, GPU memory, and default operations
- Monitoring: Web UI at `http://localhost:8233` when running locally

### Testing Strategy
- Test files located in `tests/test_files/`
- Integration tests cover full workflow pipelines
- Health check endpoints for liveness and readiness probes
- Database tests use in-memory SQLite

## Key File Structure

```
app/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration and environment variables
├── temporal_manager.py     # Temporal client management
├── temporal_workflows.py   # Workflow definitions
├── temporal_activities.py  # Activity implementations
├── whisperx_services.py    # Core WhisperX processing logic
├── model_manager.py        # Model loading and caching
├── routers/                # API route definitions
│   ├── stt.py             # Main speech-to-text endpoints
│   ├── stt_services.py    # Individual service endpoints
│   └── temporal_tasks.py  # Task management endpoints
└── schemas.py             # Pydantic models and response schemas

tests/
├── test_all.py            # Comprehensive integration tests
└── test_files/            # Audio files for testing

docs/
├── advanced_agentic_rag_plan.md    # Future RAG implementation
└── temporal_retry_policies.md      # Retry policy documentation
```

## Debugging and Monitoring

### Health Endpoints
- `/health` - Basic service status
- `/health/live` - Liveness probe with timestamp
- `/health/ready` - Readiness probe checking Temporal connection

### Logging Configuration
- Structured logging via `app/log_conf.yaml` and `app/uvicorn_log_conf.yaml`
- Log level controlled by `LOG_LEVEL` environment variable
- Trace middleware for request tracking

### Common Issues
- **Model download failures**: Check `HF_TOKEN` and internet connectivity
- **GPU memory issues**: Adjust `COMPUTE_TYPE` or use CPU fallback
- **Temporal connection**: Ensure Temporal server is running on correct port
- **Audio format errors**: Verify file extensions match supported formats

### Monitoring Tools
- Temporal Web UI: http://localhost:8233
- FastAPI docs: http://localhost:8000/docs
- Task status API: GET `/task/{identifier}`

## Production Considerations

### Docker Support
- GPU support via `docker-compose.gpu.yml`
- Model cache persistence through Docker volumes
- Multi-stage builds for optimization

### Performance
- Model caching reduces cold start times
- GPU acceleration for faster processing
- Temporal retry policies handle transient failures
- Activity timeouts prevent resource waste

### Security
- HF_TOKEN stored as environment variable
- Pre-commit hooks include secret detection (gitleaks)
- CORS and trace middleware for API security
