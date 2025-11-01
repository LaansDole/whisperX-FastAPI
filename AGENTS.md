# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Development

- **Package manager**: `uv` (NOT pip/poetry) - faster dependency management
- **Run dev**: `make dev` starts both FastAPI server AND Temporal worker (required)
- **Single test**: `pytest tests/test_file.py::test_name` (use `pytest.ini` in tests/ dir)
- **GPU builds**: `uv sync --extra gpu` (separate dependency group, not automatic)

## Critical Non-Obvious Patterns

- **warnings_filter.py MUST be imported FIRST** in [`app/main.py`](app/main.py:3) before any other imports to suppress library warnings
- **Manual GPU memory management required**: Every model load/inference in [`app/whisperx_services.py`](app/whisperx_services.py:118) must explicitly call `gc.collect()`, `torch.cuda.empty_cache()`, and `del model` to prevent OOM errors
- **Temporal worker is mandatory**: API calls fail without running worker via `make worker` - not optional despite appearing so
- **HF_TOKEN requires model acceptance**: Must accept terms on HuggingFace Hub for `pyannote/speaker-diarization-3.1` - token alone is insufficient
- **Local diarization fallback**: Set `DIARIZATION_MODEL_PATH` in `.env` for offline operation - see [`app/whisperx_services.py`](app/whisperx_services.py:166) fallback logic

## Error Handling

- **Temporal errors classified by type**: [`app/temporal_error_handler.py`](app/temporal_error_handler.py:14) determines if errors are retryable based on error message content (401/auth = non-retryable, network/CUDA OOM = retryable)
- **Three retry policy types**: Use `get_model_loading_retry_policy()` for downloads, `get_gpu_memory_retry_policy()` for CUDA errors, `get_default_retry_policy()` for everything else from [`app/temporal_config.py`](app/temporal_config.py:31)

## Code Conventions

- **Import order enforced**: `warnings_filter` must be first, then stdlib, then third-party (see [`app/main.py`](app/main.py:1) pattern)
- **Compute type validation**: CPU MUST use `int8`, GPU can use `float16`/`float32` - violating this raises ValueError in [`app/whisperx_services.py`](app/whisperx_services.py:272)
- **Device config precedence**: `Config.DEVICE` auto-detects CUDA but ENV var `DEVICE=cpu` overrides this

## Testing

- **GPU tests conditionally skipped**: Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")` decorator
- **Mock model downloads**: Always patch `load_model`, `load_align_model`, `DiarizationPipeline` in tests to avoid real downloads
- **Test files in tests/ directory**: NOT alongside source files (non-standard for Python)