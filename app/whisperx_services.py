"""
This module provides services for transcribing, diarizing, and aligning audio using Whisper and other models.

All models (transcription, diarization, alignment) are cached using thread-safe singleton patterns
to eliminate 5-10 second loading overhead on every request. Models are loaded once on first use
and reused for subsequent requests. Use clear_all_model_caches() to free GPU memory if needed.
"""

import gc
import threading
from typing import Optional

import torch
from whisperx.diarize import DiarizationPipeline
from whisperx import (
    align,
    load_align_model,
    load_model,
)

from .config import Config
from .logger import logger  # Import the logger from the new module

LANG = Config.LANG
HF_TOKEN = Config.HF_TOKEN
WHISPER_MODEL = Config.WHISPER_MODEL
DIARIZATION_MODEL_PATH = Config.DIARIZATION_MODEL_PATH
device = Config.DEVICE
compute_type = Config.COMPUTE_TYPE

_diarization_model_cache = {}
_diarization_model_lock = threading.Lock()
_diarization_cache_hits = 0
_diarization_cache_misses = 0


def get_diarization_model(device_param: str = None) -> DiarizationPipeline:
    """
    Get cached diarization model, loading only if necessary.

    Thread-safe singleton pattern for model caching to eliminate 5-10 second
    loading overhead on every request.

    Args:
        device_param: Device to load model on (cuda/cpu). Defaults to Config.DEVICE.

    Returns:
        Cached DiarizationPipeline instance

    Raises:
        RuntimeError: If model loading fails from both HuggingFace Hub and local path
    """
    global _diarization_model_cache, _diarization_model_lock, _diarization_cache_hits, _diarization_cache_misses

    device_param = device_param or Config.DEVICE
    cache_key = f"diarization_{device_param}"

    # Double-checked locking pattern for thread safety
    with _diarization_model_lock:
        if cache_key in _diarization_model_cache:
            _diarization_cache_hits += 1
            logger.debug(
                f"Diarization model cache HIT for {cache_key} (hits: {_diarization_cache_hits}, misses: {_diarization_cache_misses})"
            )
            return _diarization_model_cache[cache_key]

        _diarization_cache_misses += 1
        logger.info(f"Loading diarization model for device {device_param} (cache MISS, first time only)")

        try:
            # Attempt to load from Hugging Face Hub
            model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device_param)
            _diarization_model_cache[cache_key] = model
            logger.info(f"Diarization model cached successfully for {device_param}")

            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                logger.info(f"GPU memory after diarization model load: {allocated:.2f} MB / {total:.2f} MB")

        except Exception as e:
            logger.error(f"Failed to load diarization model from HuggingFace Hub: {e}")

            # Try local path fallback
            if DIARIZATION_MODEL_PATH:
                logger.info(f"Attempting local model load from: {DIARIZATION_MODEL_PATH}")
                try:
                    model = DiarizationPipeline(
                        model_name=DIARIZATION_MODEL_PATH, use_auth_token=HF_TOKEN, device=device_param
                    )
                    _diarization_model_cache[cache_key] = model
                    logger.info(f"Diarization model loaded from local path: {DIARIZATION_MODEL_PATH}")
                except Exception as local_e:
                    logger.error(f"Failed to load diarization model from local path: {local_e}")
                    raise RuntimeError(
                        f"Failed to load diarization model from both HuggingFace Hub and local path. "
                        f"Hub error: {e}, Local error: {local_e}"
                    ) from local_e
            else:
                raise RuntimeError(
                    f"Failed to load diarization model from HuggingFace Hub: {e}. "
                    "No local model path configured (DIARIZATION_MODEL_PATH)."
                ) from e

    return _diarization_model_cache[cache_key]


def clear_diarization_model_cache() -> None:
    """
    Clear all cached diarization models to free GPU memory.

    Useful for:
    - Memory management
    - Model updates
    - Troubleshooting
    """
    global _diarization_model_cache, _diarization_model_lock

    with _diarization_model_lock:
        for cache_key, model in list(_diarization_model_cache.items()):
            logger.info(f"Clearing diarization model cache for {cache_key}")
            # Cleanup model
            del model

        _diarization_model_cache.clear()

        # Force GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared after clearing diarization models")


def get_diarization_cache_status() -> dict:
    """
    Get status of diarization model cache.

    Returns:
        Dictionary with cache status information
    """
    global _diarization_model_cache, _diarization_cache_hits, _diarization_cache_misses

    total_requests = _diarization_cache_hits + _diarization_cache_misses
    hit_rate = (_diarization_cache_hits / total_requests * 100) if total_requests > 0 else 0

    cache_info = {
        "cached_models": list(_diarization_model_cache.keys()),
        "cache_size": len(_diarization_model_cache),
        "cache_hits": _diarization_cache_hits,
        "cache_misses": _diarization_cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
    }

    if torch.cuda.is_available():
        cache_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        cache_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2

    return cache_info


_transcription_model_cache = {}
_transcription_model_lock = threading.Lock()
_transcription_cache_hits = 0
_transcription_cache_misses = 0


def get_transcription_model(
    model_name: str,
    device_param: str,
    device_index: int = 0,
    compute_type_param: str = None,
    asr_options: dict = None,
    vad_options: dict = None,
    language: str = None,
    task: str = "transcribe",
    threads: int = 4,
):
    """
    Get cached transcription model, loading only if necessary.

    Thread-safe singleton pattern for model caching to eliminate 5-10 second
    loading overhead on every request.

    Args:
        model_name: Name of the Whisper model to use
        device_param: Device to load model on (cuda/cpu)
        device_index: Device index for FasterWhisper
        compute_type_param: Compute type for computation
        asr_options: ASR options for the model
        vad_options: VAD options for the model
        language: Language code
        task: Task type (transcribe/translate)
        threads: Number of threads

    Returns:
        Cached Whisper model instance

    Raises:
        RuntimeError: If model loading fails
    """
    global _transcription_model_cache, _transcription_model_lock, _transcription_cache_hits, _transcription_cache_misses

    compute_type_param = compute_type_param or compute_type
    cache_key = f"transcription_{model_name}_{device_param}_{compute_type_param}"

    # Double-checked locking pattern for thread safety
    with _transcription_model_lock:
        if cache_key in _transcription_model_cache:
            _transcription_cache_hits += 1
            logger.debug(
                f"Transcription model cache HIT for {cache_key} (hits: {_transcription_cache_hits}, misses: {_transcription_cache_misses})"
            )
            return _transcription_model_cache[cache_key]

        _transcription_cache_misses += 1
        logger.info(f"Loading transcription model {model_name} for device {device_param} (cache MISS, first time only)")

        try:
            # Check memory pressure before loading
            if torch.cuda.is_available() and check_gpu_memory_pressure(threshold=0.85):
                logger.warning(
                    "GPU memory pressure detected, clearing oldest caches before loading transcription model"
                )
                clear_diarization_model_cache()

            model = load_model(
                model_name,
                device_param,
                device_index=device_index,
                compute_type=compute_type_param,
                asr_options=asr_options,
                vad_options=vad_options,
                language=language,
                task=task,
                threads=threads,
            )
            _transcription_model_cache[cache_key] = model
            logger.info(f"Transcription model cached successfully for {cache_key}")

            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                logger.info(f"GPU memory after transcription model load: {allocated:.2f} MB / {total:.2f} MB")

            return model

        except Exception as e:
            logger.error(f"Failed to load transcription model: {e}")
            if "401" in str(e) or "authorization" in str(e).lower():
                raise RuntimeError(
                    "Hugging Face authentication failed. Please ensure your HF_TOKEN is correct and has permissions for the Whisper model."
                ) from e
            if "Could not find the requested files" in str(e):
                raise RuntimeError(
                    "Could not download the Whisper model. This might be due to a network issue or because you have not accepted the model's terms of service on the Hugging Face Hub."
                ) from e
            raise


def clear_transcription_model_cache() -> None:
    """Clear all cached transcription models to free GPU memory."""
    global _transcription_model_cache, _transcription_model_lock

    with _transcription_model_lock:
        for cache_key, model in list(_transcription_model_cache.items()):
            logger.info(f"Clearing transcription model cache for {cache_key}")
            del model

        _transcription_model_cache.clear()

        # Force GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared after clearing transcription models")


def get_transcription_cache_status() -> dict:
    """Get status of transcription model cache."""
    global _transcription_model_cache, _transcription_cache_hits, _transcription_cache_misses

    total_requests = _transcription_cache_hits + _transcription_cache_misses
    hit_rate = (_transcription_cache_hits / total_requests * 100) if total_requests > 0 else 0

    return {
        "cached_models": list(_transcription_model_cache.keys()),
        "cache_size": len(_transcription_model_cache),
        "cache_hits": _transcription_cache_hits,
        "cache_misses": _transcription_cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
    }


_alignment_model_cache = {}
_alignment_model_lock = threading.Lock()
_alignment_cache_hits = 0
_alignment_cache_misses = 0


def get_alignment_model(language_code: str, device_param: str, model_name: str = None):
    """
    Get cached alignment model, loading only if necessary.

    Thread-safe singleton pattern for model caching.

    Args:
        language_code: Language code for alignment
        device_param: Device to load model on (cuda/cpu)
        model_name: Optional specific model name

    Returns:
        Tuple of (cached alignment model, metadata)

    Raises:
        RuntimeError: If model loading fails
    """
    global _alignment_model_cache, _alignment_model_lock, _alignment_cache_hits, _alignment_cache_misses

    cache_key = f"alignment_{language_code}_{device_param}_{model_name or 'default'}"

    # Double-checked locking pattern
    with _alignment_model_lock:
        if cache_key in _alignment_model_cache:
            _alignment_cache_hits += 1
            logger.debug(
                f"Alignment model cache HIT for {cache_key} (hits: {_alignment_cache_hits}, misses: {_alignment_cache_misses})"
            )
            return _alignment_model_cache[cache_key]

        _alignment_cache_misses += 1
        logger.info(f"Loading alignment model for {language_code} on {device_param} (cache MISS, first time only)")

        try:
            # Check memory pressure before loading
            if torch.cuda.is_available() and check_gpu_memory_pressure(threshold=0.85):
                logger.warning("GPU memory pressure detected, clearing caches before loading alignment model")
                clear_transcription_model_cache()

            align_model, align_metadata = load_align_model(
                language_code=language_code, device=device_param, model_name=model_name
            )

            model_tuple = (align_model, align_metadata)
            _alignment_model_cache[cache_key] = model_tuple
            logger.info(f"Alignment model cached successfully for {cache_key}")

            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                logger.info(f"GPU memory after alignment model load: {allocated:.2f} MB / {total:.2f} MB")

            return model_tuple

        except Exception as e:
            logger.error(f"Failed to load alignment model: {e}")
            raise


def clear_alignment_model_cache() -> None:
    """Clear all cached alignment models to free GPU memory."""
    global _alignment_model_cache, _alignment_model_lock

    with _alignment_model_lock:
        for cache_key, (model, metadata) in list(_alignment_model_cache.items()):
            logger.info(f"Clearing alignment model cache for {cache_key}")
            del model
            del metadata

        _alignment_model_cache.clear()

        # Force GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared after clearing alignment models")


def get_alignment_cache_status() -> dict:
    """Get status of alignment model cache."""
    global _alignment_model_cache, _alignment_cache_hits, _alignment_cache_misses

    total_requests = _alignment_cache_hits + _alignment_cache_misses
    hit_rate = (_alignment_cache_hits / total_requests * 100) if total_requests > 0 else 0

    return {
        "cached_models": list(_alignment_model_cache.keys()),
        "cache_size": len(_alignment_model_cache),
        "cache_hits": _alignment_cache_hits,
        "cache_misses": _alignment_cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
    }


def check_gpu_memory_pressure(threshold: float = 0.9) -> bool:
    """
    Check if GPU memory usage exceeds threshold.

    Args:
        threshold: Memory usage threshold (0.0 to 1.0)

    Returns:
        True if memory pressure detected, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    usage_ratio = allocated / total

    if usage_ratio > threshold:
        logger.warning(f"GPU memory pressure: {usage_ratio * 100:.1f}% used (threshold: {threshold * 100:.1f}%)")
        return True

    return False


def clear_all_model_caches() -> None:
    """Clear all cached models (transcription, alignment, diarization) to free GPU memory."""
    logger.info("Clearing all model caches")

    clear_transcription_model_cache()
    clear_alignment_model_cache()
    clear_diarization_model_cache()

    logger.info("All model caches cleared")


def get_all_cache_status() -> dict:
    """Get combined status of all model caches."""
    status = {
        "transcription": get_transcription_cache_status(),
        "alignment": get_alignment_cache_status(),
        "diarization": get_diarization_cache_status(),
    }

    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

        status["gpu_memory"] = {
            "allocated_mb": round(allocated_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
            "total_mb": round(total_mb, 2),
            "usage_percent": round((allocated_mb / total_mb) * 100, 2),
        }

    return status


def get_total_gpu_memory_usage() -> Optional[dict]:
    """Get total GPU memory usage by all cached models."""
    if not torch.cuda.is_available():
        return None

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }


def transcribe_with_whisper(
    audio,
    task,
    asr_options,
    vad_options,
    language,
    batch_size: int = 16,
    chunk_size: int = 20,
    model: str = WHISPER_MODEL,
    device: str = device,
    device_index: int = 0,
    compute_type: str = compute_type,
    threads: int = 0,
):
    """
    Transcribe an audio file using cached Whisper model.

    Uses thread-safe model caching to eliminate 5-10 second loading overhead
    on every request.

    Args:
       audio (Audio): The audio to transcribe.
       batch_size (int): Batch size for transcription (default 16).
       chunk_size (int): Chunk size for transcription (default 20).
       model (str): Name of the Whisper model to use.
       device (str): Device to use for PyTorch inference.
       device_index (int): Device index to use for FasterWhisper inference.
       compute_type (str): Compute type for computation.

    Returns:
       Transcript: The transcription result.
    """
    logger.debug(
        "Starting transcription with Whisper model: %s on device: %s",
        WHISPER_MODEL,
        device,
    )
    # Log GPU memory before transcription
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before transcription - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    faster_whisper_threads = 4
    if (threads := threads) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    logger.debug(
        "Loading model with config - model: %s, device: %s, compute_type: %s, threads: %d, task: %s, language: %s",
        model.value,
        device,
        compute_type,
        faster_whisper_threads,
        task,
        language,
    )

    try:
        # Get cached model (loads only on first call)
        cached_model = get_transcription_model(
            model_name=model.value,
            device_param=device,
            device_index=device_index,
            compute_type_param=compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            task=task,
            threads=faster_whisper_threads,
        )
        logger.debug("Transcription model loaded from cache")

        result = cached_model.transcribe(audio=audio, batch_size=batch_size, chunk_size=chunk_size, language=language)

        # Log GPU memory after transcription (model kept cached!)
        if torch.cuda.is_available():
            logger.debug(
                f"GPU memory after transcription: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (model kept cached)"
            )

        logger.debug("Completed transcription")
        return result

    except Exception as e:
        logger.error(f"Error during transcription model loading or inference: {e}")
        if "401" in str(e) or "authorization" in str(e).lower():
            raise RuntimeError(
                "Hugging Face authentication failed. Please ensure your HF_TOKEN is correct and has permissions for the Whisper model."
            )
        if "Could not find the requested files" in str(e):
            raise RuntimeError(
                "Could not download the Whisper model. This might be due to a network issue or because you have not accepted the model's terms of service on the Hugging Face Hub."
            )
        raise e


def diarize(audio, device: str = None, min_speakers=None, max_speakers=None):
    """
    Diarize audio using cached model.

    Uses thread-safe model caching to eliminate 5-10 second loading overhead
    on every request.

    Args:
        audio: Audio array to diarize
        device: Device to use (cuda/cpu). Defaults to Config.DEVICE
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers

    Returns:
        Diarization result DataFrame

    Raises:
        ValueError: If HF_TOKEN is not set
        RuntimeError: If model loading fails
    """
    device = device or Config.DEVICE
    logger.debug(f"Starting diarization with device: {device}")

    if not HF_TOKEN:
        raise ValueError(
            "Hugging Face token is not set. Please set the HF_TOKEN environment variable."
            "You can get a token from https://huggingface.co/settings/tokens"
            "Also, you need to agree to the terms of service for the pyannote/speaker-diarization-3.1 model on the Hugging Face Hub."
        )

    # Log GPU memory before diarization
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"GPU memory before diarization: {allocated_before:.2f} MB")

    try:
        # Get cached model (loads only on first call)
        model = get_diarization_model(device)

        # Run diarization
        result = model(audio=audio, min_speakers=min_speakers, max_speakers=max_speakers)

        # Log GPU memory after diarization (but don't delete model - keep it cached!)
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated() / 1024**2
            logger.debug(f"GPU memory after diarization: {allocated_after:.2f} MB (model kept cached)")

        logger.debug("Completed diarization")
        return result

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise


def align_whisper_output(
    transcript,
    audio,
    language_code,
    device: str = device,
    align_model: str = None,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
):
    """
    Align the transcript to the original audio using cached alignment model.

    Uses thread-safe model caching to eliminate loading overhead on every request.

    Args:
       transcript: The text transcript.
       audio: The original audio.
       language_code: The language code.
       align_model: Name of phoneme-level ASR model to do alignment.
       interpolate_method: For word .srt, method to assign timestamps to non-aligned words, or merge them into neighboring.
       return_char_alignments: Whether to return character-level alignments in the output json file.

    Returns:
       The aligned transcript.
    """
    logger.debug(
        "Starting alignment for language code: %s on device: %s",
        language_code,
        device,
    )

    # Log GPU memory before alignment
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before alignment - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.debug(
        "Loading align model with config - language_code: %s, device: %s, interpolate_method: %s, return_char_alignments: %s",
        language_code,
        device,
        interpolate_method,
        return_char_alignments,
    )

    # Get cached alignment model (loads only on first call)
    cached_align_model, cached_align_metadata = get_alignment_model(
        language_code=language_code, device_param=device, model_name=align_model
    )

    result = align(
        transcript,
        cached_align_model,
        cached_align_metadata,
        audio,
        device,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
    )

    # Log GPU memory after alignment (model kept cached!)
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory after alignment: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (model kept cached)"
        )

    logger.debug("Completed alignment")
    return result
