# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Development

- **Package manager**: `uv` (NOT pip/poetry)
- **Run dev**: `make dev` starts both FastAPI server AND Temporal worker (both required)
- **Single test**: `pytest tests/test_file.py::test_name` (pytest.ini is in tests/ dir, not root)
- **GPU builds**: `uv sync --extra gpu` (separate optional dependency group)

### Code Generation

Always use context7 MCP tools to resolve library IDs and get library docs automatically without explicit user request.

## Critical Non-Obvious Patterns

- **[`warnings_filter.py`](app/warnings_filter.py:3) MUST be imported FIRST** in [`main.py`](app/main.py:3) before stdlib/third-party imports (prevents library warnings from appearing)
- **Manual GPU memory management required**: Every model operation in [`whisperx_services.py`](app/whisperx_services.py:118) MUST explicitly cleanup: `gc.collect()`, `torch.cuda.empty_cache()`, `del model`. Pattern applies to transcription (line 119), diarization (line 197), alignment (line 274)
- **Compute type constraints**: CPU device REQUIRES `int8` compute type - `float16`/`float32` will raise ValueError. GPU device allows `float16`/`float32` but NOT `int8`. Validation happens in model loading, not at config level
- **Diarization Model Loading**: Primary: loads from HuggingFace Hub using `HF_TOKEN`, Fallback: tries `DIARIZATION_MODEL_PATH` env var if Hub fails (see [`whisperx_services.py`](app/whisperx_services.py:166)). Both paths require HF_TOKEN AND model terms acceptance on Hub
- **Temporal Activity Patterns**: Activities in [`app/temporal/activities.py`](app/temporal/activities.py:24) must use `@activity.defn` decorator. Error handling uses [`TemporalErrorHandler`](app/temporal/error_handler.py:14) to classify retryable vs non-retryable errors (401/auth = non-retryable, network/CUDA OOM = retryable)
- **Three retry policy types**: Use `get_model_loading_retry_policy()` for downloads, `get_gpu_memory_retry_policy()` for CUDA errors, `get_default_retry_policy()` for everything else from [`app/temporal/config.py`](app/temporal/config.py:31)
- **File Processing Pattern**: Video files automatically converted to audio using ffmpeg in [`audio.py`](app/audio.py:12). Audio processing uses `whisperx.load_audio()` which returns numpy arrays. Temporary files preserve original extensions but are converted to WAV for processing

## Testing

- **GPU tests conditionally skipped**: Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")` decorator
- **Mock model downloads**: Always patch `load_model`, `load_align_model`, `DiarizationPipeline` in tests to avoid real downloads
- **Test files in tests/ directory**: NOT alongside source files (non-standard for Python)