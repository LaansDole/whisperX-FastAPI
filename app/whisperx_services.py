"""This module provides services for transcribing, diarizing, and aligning audio using Whisper and other models."""

import gc
from datetime import datetime

import psutil
import torch
from fastapi import Depends
from sqlalchemy.orm import Session
from whisperx.diarize import DiarizationPipeline
from whisperx import (
    align,
    assign_word_speakers,
    load_align_model,
    load_model,
)

from .config import Config
from .logger import logger  # Import the logger from the new module
from .schemas import AlignedTranscription, SpeechToTextProcessingParams
from .transcript import filter_aligned_transcription
from .language_optimization import (
    get_best_model_for_language,
    get_optimal_compute_type,
    get_diarization_config_for_language
)

LANG = Config.LANG
HF_TOKEN = Config.HF_TOKEN
WHISPER_MODEL = Config.WHISPER_MODEL
DIARIZATION_MODEL_PATH = Config.DIARIZATION_MODEL_PATH
device = Config.DEVICE
compute_type = Config.COMPUTE_TYPE


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
    Transcribe an audio file using the Whisper model.

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
    # Log GPU memory before loading model
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before loading model - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )
    faster_whisper_threads = 4
    if (threads := threads) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    # Handle both WhisperModel enum and HuggingFace model strings
    if isinstance(model, str) and "/" in model:
        # HuggingFace model - use as-is
        model_name = model
        model_display = model
    else:
        # WhisperModel enum - use .value
        model_name = model.value if hasattr(model, 'value') else model
        model_display = model.value if hasattr(model, 'value') else model

    logger.debug(
        "Loading model with config - model: %s, device: %s, compute_type: %s, threads: %d, task: %s, language: %s",
        model_display,
        device,
        compute_type,
        faster_whisper_threads,
        task,
        language,
    )
    try:
        # Check if we're on CPU but trying to use CUDA - fallback to CPU
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
            if compute_type == "float16":
                compute_type = "int8"
        
        model = load_model(
            model_name,
            device,
            device_index=device_index,
            compute_type=compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            task=task,
            threads=faster_whisper_threads,
        )
        logger.debug("Transcription model loaded successfully")
        result = model.transcribe(
            audio=audio, batch_size=batch_size, chunk_size=chunk_size, language=language
        )
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

    # Log GPU memory before cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    # delete model
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # Log GPU memory after cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.debug("Completed transcription")
    return result


def diarize(audio, device: str = device, min_speakers=None, max_speakers=None):
    """
    Diarize an audio file using the PyAnnotate model.

    Args:
       audio (Audio): The audio to diarize.

    Returns:
       Diarizartion: The diarization result.
    """
    logger.debug("Starting diarization with device: %s", device)

    if not HF_TOKEN:
        raise ValueError(
            "Hugging Face token is not set. Please set the HF_TOKEN environment variable."
            "You can get a token from https://huggingface.co/settings/tokens"
            "Also, you need to agree to the terms of service for the pyannote/speaker-diarization-3.1 model on the Hugging Face Hub."
        )

    # Log GPU memory before loading model
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before loading model - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    try:
        # Attempt to load from Hugging Face Hub first
        model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        result = model(audio=audio, min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as e:
        logger.error(f"Error during diarization model loading from Hugging Face Hub: {e}")
        
        # If download fails, try loading from the local path if provided
        if DIARIZATION_MODEL_PATH:
            logger.info(f"Attempting to load diarization model from local path: {DIARIZATION_MODEL_PATH}")
            try:
                model = DiarizationPipeline(model_name=DIARIZATION_MODEL_PATH, use_auth_token=HF_TOKEN, device=device)
                result = model(audio=audio, min_speakers=min_speakers, max_speakers=max_speakers)
            except Exception as local_e:
                logger.error(f"Failed to load diarization model from local path: {local_e}")
                raise RuntimeError(
                    "Failed to load diarization model from both Hugging Face Hub and the local path. "
                    "Please check your internet connection, HF_TOKEN, and the local model path."
                ) from local_e
        else:
            # If no local path is provided, re-raise the original error with a more informative message
            if "401" in str(e) or "authorization" in str(e).lower():
                raise RuntimeError(
                    "Hugging Face authentication failed. Please ensure your HF_TOKEN is correct and has permissions for 'pyannote/speaker-diarization-3.1'. See README for troubleshooting."
                ) from e
            if "Could not find the requested files" in str(e):
                raise RuntimeError(
                    "Could not download the diarization model. This might be due to a network issue or because you have not accepted the model's terms of service on the Hugging Face Hub for 'pyannote/speaker-diarization-3.1'. See README for troubleshooting."
                ) from e
            raise e


    # Log GPU memory before cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    # delete model
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # Log GPU memory after cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.debug("Completed diarization with device: %s", device)
    return result


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
    Align the transcript to the original audio.

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

    # Log GPU memory before loading model
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before loading model - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.debug(
        "Loading align model with config - language_code: %s, device: %s, interpolate_method: %s, return_char_alignments: %s",
        language_code,
        device,
        interpolate_method,
        return_char_alignments,
    )
    align_model, align_metadata = load_align_model(
        language_code=language_code, device=device, model_name=align_model
    )

    result = align(
        transcript,
        align_model,
        align_metadata,
        audio,
        device,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
    )

    # Log GPU memory before cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    # delete model
    gc.collect()
    torch.cuda.empty_cache()
    del align_model
    del align_metadata

    # Log GPU memory after cleanup
    if torch.cuda.is_available():
        logger.debug(
            f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.debug("Completed alignment")
    return result


def transcribe_with_optimized_model(
    audio,
    language: str,
    task: str = "transcribe",
    device: str = "cuda",
    device_index: int = 0,
    asr_options: dict = None,
    vad_options: dict = None,
    batch_size: int = 8,
    threads: int = 0,
    override_model: str = None,
) -> dict:
    """
    Transcribe audio using the optimal model for the specified language.
    
    This function automatically selects the best Whisper model for the given language
    based on AudioBench performance data, along with optimal compute settings.
    
    Args:
        audio: Audio data for transcription
        language: Target language code (e.g., 'en', 'zh', 'ja')
        task: Transcription task ('transcribe' or 'translate')
        device: Device for inference ('cuda' or 'cpu')
        device_index: GPU device index
        asr_options: ASR configuration options
        vad_options: VAD configuration options
        batch_size: Batch size for processing
        threads: Number of CPU threads
        override_model: Force use of specific model instead of optimal one
        
    Returns:
        dict: Transcription results with metadata about model selection
    """
    from .schemas import WhisperModel
    
    # Get optimal model for language (unless overridden)
    if override_model:
        optimal_model = WhisperModel(override_model)
        logger.info(f"Using overridden model: {optimal_model.value}")
    else:
        model_str = get_best_model_for_language(language)
        # Check if it's a WhisperModel enum value or HuggingFace model string
        if model_str in [m.value for m in WhisperModel]:
            optimal_model = WhisperModel(model_str)
            logger.info(f"Auto-selected optimal model {optimal_model.value} for language '{language}'")
        else:
            # It's a HuggingFace model string - use as-is
            optimal_model = model_str
            logger.info(f"Auto-selected optimal HuggingFace model {optimal_model} for language '{language}'")
    
    # Get optimal compute type for language
    optimal_compute_type = get_optimal_compute_type(language, device)
    logger.info(f"Using compute type '{optimal_compute_type}' for language '{language}' on device '{device}'")
    
    # Log model selection reasoning
    model_display = optimal_model.value if hasattr(optimal_model, 'value') else optimal_model
    logger.info(
        f"Language optimization: language='{language}' -> model='{model_display}', "
        f"compute_type='{optimal_compute_type}'"
    )
    
    # Use the existing transcribe function with optimal parameters
    # Ensure model is a string for the transcribe function
    model_for_transcribe = optimal_model.value if hasattr(optimal_model, 'value') else optimal_model
    
    result = transcribe_with_whisper(
        audio=audio,
        model=model_for_transcribe,
        device=device,
        device_index=device_index,
        compute_type=optimal_compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
        batch_size=batch_size,
        threads=threads,
        language=language,
        task=task,
    )
    
    # Add optimization metadata - ensure result is a dict
    if isinstance(result, dict):
        result["optimization_metadata"] = {
            "selected_model": optimal_model.value if hasattr(optimal_model, 'value') else optimal_model,
            "language": language,
            "compute_type": optimal_compute_type,
            "optimization_applied": override_model is None,
            "device": device
        }
    else:
        # If result is not a dict, wrap it with metadata
        result = {
            "transcription": result,
            "optimization_metadata": {
                "selected_model": optimal_model.value if hasattr(optimal_model, 'value') else optimal_model,
                "language": language,
                "compute_type": optimal_compute_type,
                "optimization_applied": override_model is None,
                "device": device
            }
        }
    
    return result


def diarize_with_optimized_config(
    audio,
    language: str,
    diarization_model_path: str = None,
    use_auth_token: bool = False,
    device: str = "cuda",
    hf_token: str = None,
    override_min_speakers: int = None,
    override_max_speakers: int = None,
) -> dict:
    """
    Perform diarization using language-specific optimal configuration.
    
    Args:
        audio: Audio data for diarization
        language: Language code for optimal configuration
        diarization_model_path: Path to diarization model
        use_auth_token: Whether to use authentication token
        device: Device for inference
        hf_token: Hugging Face token
        override_min_speakers: Override minimum speaker count
        override_max_speakers: Override maximum speaker count
        
    Returns:
        dict: Diarization results with configuration metadata
    """
    # Get optimal diarization config for language
    diar_config = get_diarization_config_for_language(language)
    
    # Apply overrides if provided
    min_speakers = override_min_speakers or diar_config["min_speakers"]
    max_speakers = override_max_speakers or diar_config["max_speakers"]
    confidence_threshold = diar_config["confidence_threshold"]
    
    logger.info(
        f"Language-optimized diarization for '{language}': "
        f"min_speakers={min_speakers}, max_speakers={max_speakers}, "
        f"confidence_threshold={confidence_threshold}"
    )
    
    # Use existing diarization function with optimal parameters
    result = diarize(
        audio=audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )
    
    # Convert DataFrame to dict format and add optimization metadata
    import pandas as pd
    
    # Convert to dict format for consistency with other functions
    result_dict = {
        "segments": result.to_dict(orient="records"),
        "metadata": {
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
    }
    
    # Filter segments by confidence threshold if applicable
    filtered_segments = [
        segment for segment in result_dict["segments"]
        if segment.get("confidence", 1.0) >= confidence_threshold
    ]
    
    # Add optimization metadata
    result_dict["optimization_metadata"] = {
        "language": language,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "confidence_threshold": confidence_threshold,
        "original_segment_count": len(result_dict["segments"]),
        "filtered_segment_count": len(filtered_segments),
        "optimization_applied": True
    }
    
    # Use filtered segments if confidence filtering was applied
    if confidence_threshold > 0:
        result_dict["segments"] = filtered_segments
    
    return result_dict



