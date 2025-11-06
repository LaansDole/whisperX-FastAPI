# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Development

- **Package manager**: `uv` (NOT pip/poetry) - faster dependency management
- **Run dev**: `make dev` starts both FastAPI server AND Temporal worker (required)
- **Single test**: `pytest tests/test_file.py::test_name` (use `pytest.ini` in tests/ dir)
- **GPU builds**: `uv sync --extra gpu` (separate dependency group, not automatic)

### Code generation

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

## Critical Non-Obvious Patterns

- **warnings_filter.py MUST be imported FIRST** in [`app/main.py`](app/main.py:3) before any other imports to suppress library warnings
- **Manual GPU memory management required**: Every model operation in [`app/whisperx_services.py`](app/whisperx_services.py:118) MUST explicitly cleanup: `gc.collect()`, `torch.cuda.empty_cache()`, `del model`
- **Temporal Activity Patterns**: Activities in [`activities.py`](app/activities.py:24) must use `@activity.defn` decorator
- **Error handling uses [`TemporalErrorHandler`](app/temporal_error_handler.py:14) to classify retryable vs non-retryable errors**
- **401/auth errors are non-retryable, network/CUDA OOM are retryable**
- **Diarization Model Loading**: Primary: loads from HuggingFace Hub using `HF_TOKEN`, Fallback: tries `DIARIZATION_MODEL_PATH` env var if Hub fails (see [`whisperx_services.py`](app/whisperx_services.py:166))
- **File Processing Pattern**: Video files automatically converted to audio using ffmpeg in [`audio.py`](app/audio.py:12)
- **Audio processing uses whisperx.load_audio() which returns numpy arrays**
- **Temporary files preserve original extensions but are converted to WAV for processing**

## Error Handling

- **Temporal errors classified by type**: [`app/temporal_error_handler.py`](app/temporal_error_handler.py:14) determines if errors are retryable based on error message content (401/auth = non-retryable, network/CUDA OOM = retryable)
- **Three retry policy types**: Use `get_model_loading_retry_policy()` for downloads, `get_gpu_memory_retry_policy()` for CUDA errors, `get_default_retry_policy()` for everything else from [`app/temporal_config.py`](app/temporal_config.py:31)

## Code Conventions

- **Import order enforced**: `warnings_filter` must be first, then stdlib, then third-party (see [`app/main.py`](app/main.py:1) pattern)
- **Compute type validation**: CPU device REQUIRES `int8` compute type - `float16`/`float32` will raise ValueError
- **GPU device allows `float16`/`float32` but NOT `int8`**
- **Validation happens in model loading, not at config level**
- **Device config precedence**: `Config.DEVICE` auto-detects CUDA but ENV var `DEVICE=cpu` overrides this

## Testing

- **GPU tests conditionally skipped**: Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")` decorator
- **Mock model downloads**: Always patch `load_model`, `load_align_model`, `DiarizationPipeline` in tests to avoid real downloads
- **Test files in tests/ directory**: NOT alongside source files (non-standard for Python)