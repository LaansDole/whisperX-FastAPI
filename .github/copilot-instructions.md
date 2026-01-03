# GitHub Copilot Code Review Instructions

**Project**: whisperX-FastAPI - Audio Processing API
**Stack**: Python 3.10+, FastAPI, Temporal, WhisperX, SQLite, PyTorch
**Last Updated**: 2025-12-28

---

## Review Philosophy

- Only comment when you have HIGH CONFIDENCE (>80%) that an issue exists
- Be concise: one sentence per comment when possible
- Focus on actionable feedback, not observations
- When reviewing text, only comment on clarity issues if the text is genuinely confusing or could lead to errors
- Prioritize correctness, performance, and maintainability

---

## Priority Areas (Review These)

### Security (CRITICAL)
- Hardcoded secrets, API keys, tokens, or credentials
- SQL injection vulnerabilities in database queries
- Command injection risks in shell commands or user input
- Missing input validation on external data (file uploads, API parameters)
- Improper error handling that could leak sensitive information
- CORS misconfigurations that could expose data
- Missing authentication/authorization on protected endpoints

### GPU Memory Management (CRITICAL)
- Missing GPU memory cleanup after model operations (`gc.collect()`, `torch.cuda.empty_cache()`, `del model`)
- Memory leaks in model loading/unloading cycles
- Explicitly check that all model operations in `whisperx_services.py` follow the cleanup pattern

### Temporal Workflow Patterns
- Activities missing `@activity.defn` decorator
- Incorrect error classification (retryable vs non-retryable)
- Missing or inappropriate retry policies
- Activities that could timeout but don't have proper timeout configuration
- Workflow IDs that don't follow consistent naming format

### Async/Await Patterns
- Blocking calls in async functions (e.g., synchronous file I/O, HTTP requests)
- Missing `await` on async calls
- Mixed sync/async patterns that could block the event loop
- Use of `sync_to_async` or `async_to_sync` when proper async alternatives exist

### Error Handling
- Swallowed exceptions (bare `except:` or `pass` without logging)
- Missing error context in error messages
- Incorrect use of `unwrap()` or similar patterns that could panic
- Generic exceptions instead of specific exception types
- Missing error propagation in async functions

### Database Operations
- Missing async session management (`async with async_session()`)
- SQL injection vulnerabilities in raw SQL queries
- Missing indexes on frequently queried columns (`patient_hash`, `workflow_id`)
- N+1 query patterns in database operations
- Missing transaction management for multi-step operations

### Type Safety & Validation
- Missing type hints on function arguments and return values
- Missing Pydantic validation on API endpoints
- Optional types that don't need to be optional
- Booleans that should default to `False` but are set as optional
- Missing `response_model` on FastAPI endpoints

### API Design
- Inconsistent response formats across endpoints
- Missing appropriate HTTP status codes
- Missing or incorrect OpenAPI documentation
- Endpoints missing rate limiting on public APIs
- File upload endpoints missing file type/size validation

### Code Quality
- Unnecessary comments that just restate what the code already shows (remove them)
- Overly defensive code that adds unnecessary checks
- Functions that are too long (>50 lines) and should be refactored
- Duplicate code that should be extracted into reusable functions
- Missing docstrings on public APIs

### Testing
- Missing test coverage for critical code paths
- Tests that don't properly mock model downloads
- Integration tests that require external services without proper fixtures
- Missing cleanup in test fixtures

---

## Project-Specific Context

This is a **Python 3.10+ FastAPI application** with the following characteristics:

### Core Architecture
- **FastAPI**: Async web framework with automatic OpenAPI documentation at `/docs`
- **Temporal**: Workflow orchestration for audio processing pipelines
- **WhisperX**: Audio processing (transcription, alignment, diarization) using faster-whisper and pyannote
- **SQLite**: Workflow mapping database with SQLAlchemy 2.0 async
- **PyTorch**: Deep learning models for audio processing (GPU/CPU)

### Key Components
- `app/main.py`: FastAPI application with lifespan management
- `app/temporal/workflows.py`: Workflow orchestration logic
- `app/temporal/activities.py`: Individual processing tasks (transcribe, align, diarize)
- `app/whisperx_services.py`: Core audio processing with model management
- `app/model_manager.py`: Model loading and GPU memory management
- `app/patients/database.py`: SQLite database for workflow mappings
- `app/patients/mapping.py`: Patient-to-workflow mapping utilities
- `app/patients/filename_utils.py`: Patient filename utilities (for future use)

### Critical Patterns

#### 1. GPU Memory Cleanup Pattern
Every model operation MUST follow this pattern:
```python
import gc
import torch

# After model operations
gc.collect()
torch.cuda.empty_cache()
del model
```
Location: `app/whisperx_services.py` lines 119, 197, 274

#### 2. Warning Filter Import Order
`warnings_filter.py` MUST be imported FIRST in `main.py` before any stdlib/third-party imports to prevent library warnings in application output.

#### 3. Patient Hash Generation (for future use)
The codebase includes utilities for patient hashing, but this is not yet in production. **No active review needed for HIPAA-related code.**
```python
import hashlib

patient_hash = hashlib.sha256(f"{patient_name}{HIPAA_SALT}".encode()).hexdigest()[:8]
```
Located in: `app/patients/filename_utils.py` and `app/patients/mapping.py`

#### 4. Temporal Error Handling
- Activities use `@activity.defn` decorator
- `TemporalErrorHandler` in `app/temporal/error_handler.py` classifies errors:
  - 401/auth errors = non-retryable
  - Network/CUDA OOM = retryable
- Three retry policies in `app/temporal/config.py`:
  - `get_model_loading_retry_policy()` for downloads
  - `get_gpu_memory_retry_policy()` for CUDA errors
  - `get_default_retry_policy()` for everything else

### API Endpoint Structure
- `POST /speech-to-text`: Main endpoint for audio processing
- `POST /speech-to-text-url`: Process files from URL
- `GET /tasks/{task_id}`: Check Temporal workflow status
- `GET /admin/*`: Admin endpoints (for future use)

### Supported File Formats
- Audio: `.oga`, `.m4a`, `.aac`, `.wav`, `.amr`, `.wma`, `.awb`, `.mp3`, `.ogg`
- Video: `.wmv`, `.mkv`, `.avi`, `.mov`, `.mp4`

### Package Management
- Uses `uv` for dependency management (NOT pip)
- Install with: `uv sync` (CPU) or `uv sync --extra gpu` (GPU)
- See `pyproject.toml` for dependencies

### Configuration
- Environment variables in `.env`
- `HF_TOKEN`: Required for HuggingFace model downloads
- `DEVICE`: cpu or cuda
- `COMPUTE_TYPE`: int8 (CPU) or float16/float32 (GPU)
- `TEMPORAL_SERVER_URL`: Temporal server endpoint (default: localhost:7233)

---

## CI Pipeline Context

The following are already covered by CI/CD pipelines:

### Already Checked by CI
- **Linting**: Ruff linter checks code style and potential errors
- **Formatting**: Ruff formatter ensures consistent code formatting
- **Type Checking**: Type hints validated where present
- **Unit Tests**: Core logic tests run on every PR
- **Security Scanning**: Basic security vulnerability scans

### Do NOT Comment On These (CI Covers It)
- Code style violations (handled by ruff)
- Missing imports (caught by linter/tests)
- Formatting issues (handled by ruff format)
- Basic type errors (caught by mypy if enabled)
- Simple syntax errors (caught by linter)

### Focus On Things CI Can't Catch
- Logic errors that compile but produce wrong results
- Race conditions in async code
- Memory leaks that only appear at runtime
- Security vulnerabilities in application logic
- Performance issues
- Architectural inconsistencies

---

## Testing Guidelines

### Test Structure
- **Unit Tests**: `tests/unit/` - Test individual functions/classes
- **Integration Tests**: `tests/integration/` - Test API endpoints (require running server)
- **Service Tests**: `tests/services/` - Test service layer logic
- **Scripts**: `tests/scripts/` - Utility scripts (not pytest tests)

### Test Markers
- `@pytest.mark.integration`: Integration tests requiring live server
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.gpu`: GPU tests (skip if CUDA unavailable)

### Running Tests
```bash
# All tests
pytest

# Specific categories
pytest tests/unit/
pytest tests/integration/
pytest tests/services/

# With markers
pytest -m "not gpu"
pytest -m integration
```

### Test Coverage
- Always mock model downloads: patch `load_model`, `load_align_model`, `DiarizationPipeline`
- GPU tests should skip if CUDA unavailable: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`
- Shared fixtures in `tests/conftest.py`
- **Excluded from test coverage**: `streamlit_app/` is a demonstration-only directory and does not require test coverage. Only review logic issues if they exist.

---

## Common Mistakes to Watch For

### 1. Missing GPU Memory Cleanup
**Bad:**
```python
def transcribe(audio):
    model = load_model()
    result = model.transcribe(audio)
    return result  # Model not cleaned up!
```

**Good:**
```python
def transcribe(audio):
    model = load_model()
    try:
        result = model.transcribe(audio)
        return result
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        del model
```

### 2. Blocking in Async Functions
**Bad:**
```python
async def process_audio(file_path: str):
    audio = whisperx.load_audio(file_path)  # Blocking!
    result = await transcribe(audio)
    return result
```

**Good:**
```python
async def process_audio(file_path: str):
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, whisperx.load_audio, file_path)
    result = await transcribe(audio)
    return result
```

### 3. Swallowed Exceptions
**Bad:**
```python
try:
    process_audio(audio)
except:
    pass  # Silent failure!
```

**Good:**
```python
try:
    process_audio(audio)
except AudioProcessingError as e:
    logger.error(f"Failed to process audio: {e}")
    raise
```

---

## Performance Considerations

- **GPU Utilization**: Monitor GPU memory usage during processing
- **Database Queries**: Use indexes on `patient_hash` and `workflow_id`
- **Async Operations**: Never block the event loop with sync I/O
- **Caching**: Leverage HuggingFace model cache
- **Batch Processing**: Process multiple files when possible

---

## Documentation Standards

- Use Google-style docstrings for all public APIs
- Include type hints on all function arguments and return values
- Keep OpenAPI documentation up to date for all endpoints
- Include examples in docstrings for complex functions
