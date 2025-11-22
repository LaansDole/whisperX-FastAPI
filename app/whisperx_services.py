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
from .meralion_services import transcribe_with_fallback as meralion_transcribe

LANG = Config.LANG
HF_TOKEN = Config.HF_TOKEN
WHISPER_MODEL = Config.WHISPER_MODEL
DIARIZATION_MODEL_PATH = Config.DIARIZATION_MODEL_PATH
device = Config.DEVICE
compute_type = Config.COMPUTE_TYPE

# Fallback model hierarchy for MERaLiON
FALLBACK_MODELS = [
    "whisper-large-v3",
    "faster-whisper",
    "whisperX"
]


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
        
        # Ensure compute type is compatible with device
        if device == "cpu" and compute_type in ["float16", "float32"]:
            logger.info(f"CPU device detected, changing compute_type from {compute_type} to int8")
            compute_type = "int8"
        elif device == "cuda" and compute_type == "int8":
            logger.info(f"CUDA device detected, changing compute_type from int8 to float16")
            compute_type = "float16"
        
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




def transcribe_with_meralion_fallback(
    audio,
    task: str = "transcribe",
    language: str = "auto",
    device: str = None,
    device_index: int = 0,
    asr_options: dict = None,
    vad_options: dict = None,
    batch_size: int = 8,
    threads: int = 0,
    use_meralion: bool = True,
    meralion_fallback_enabled: bool = True,
    meralion_max_new_tokens: int = 256,
    override_model: str = None,
) -> dict:
    """
    Transcribe audio using MERaLiON model with fallback to Whisper models.
    
    This function provides a unified interface for transcription that prioritizes
    MERaLiON-AudioLLM-Whisper-SEA-LION model but gracefully falls back to
    traditional Whisper models when needed.
    
    Args:
        audio: Audio data for transcription
        task: Transcription task ('transcribe' or 'translate')
        language: Target language code (e.g., 'en', 'zh', 'ja')
        device: Device for inference ('cuda', 'cpu', or None for auto-detection)
        device_index: GPU device index
        asr_options: ASR configuration options
        vad_options: VAD configuration options
        batch_size: Batch size for processing
        threads: Number of CPU threads
        use_meralion: Whether to use MERaLiON as primary model
        meralion_fallback_enabled: Whether to enable fallback to Whisper models
        meralion_max_new_tokens: Maximum tokens for MERaLiON generation
        override_model: Force use of specific model instead of auto-selection
        
    Returns:
        dict: Transcription results with metadata about model selection and fallback
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if MERaLiON is enabled in config
    meralion_config_enabled = getattr(Config, 'MERALION_ENABLED', True)
    
    # Determine if we should use MERaLiON
    should_use_meralion = (
        use_meralion and
        meralion_config_enabled and
        override_model is None
    )
    
    logger.info(
        f"Transcription request - task: {task}, language: {language}, device: {device}, "
        f"use_meralion: {should_use_meralion}, fallback_enabled: {meralion_fallback_enabled}"
    )
    
    if should_use_meralion:
        try:
            # Use MERaLiON with fallback
            fallback_models = None
            if meralion_fallback_enabled:
                fallback_models = getattr(Config, 'MERALION_FALLBACK_MODELS', FALLBACK_MODELS)
            
            result = meralion_transcribe(
                audio=audio,
                task=task,
                device=device,
                max_new_tokens=meralion_max_new_tokens,
                fallback_models=fallback_models
            )
            
            # Add comprehensive metadata
            result["transcription_metadata"] = {
                "primary_model": "MERaLiON-AudioLLM-Whisper-SEA-LION",
                "actual_model": result.get("model_used", "unknown"),
                "device": device,
                "task": task,
                "language": language,
                "meralion_used": result.get("model_used") == "MERaLiON-AudioLLM-Whisper-SEA-LION",
                "fallback_used": result.get("fallback_used", False),
                "meralion_fallback_enabled": meralion_fallback_enabled,
                "audio_duration": result.get("audio_duration"),
                "success": result.get("success", False)
            }
            
            if result.get("success", False):
                logger.info(
                    f"MERaLiON transcription successful using model: {result.get('model_used')}"
                )
                return result
            else:
                logger.warning(
                    f"MERaLiON transcription failed: {result.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"MERaLiON transcription failed with exception: {e}")
            result = {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    # Fallback to traditional Whisper models
    logger.info("Falling back to traditional Whisper models")
    
    try:
        # Use the standard transcription function
        whisper_result = transcribe_with_whisper(
            audio=audio,
            task=task,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            batch_size=batch_size,
            chunk_size=20,
            model=override_model or WHISPER_MODEL,
            device=device,
            device_index=device_index,
            compute_type="int8" if device == "cpu" else "float16",
            threads=threads,
        )
        
        # Extract text from whisper result
        if isinstance(whisper_result, dict):
            if "segments" in whisper_result:
                text = " ".join([segment.get("text", "") for segment in whisper_result["segments"]])
            elif "transcription" in whisper_result:
                # Handle wrapped transcription
                transcription = whisper_result["transcription"]
                if hasattr(transcription, 'segments'):
                    text = " ".join([segment.text for segment in transcription.segments])
                elif isinstance(transcription, dict) and "segments" in transcription:
                    text = " ".join([segment.get("text", "") for segment in transcription["segments"]])
                else:
                    text = str(transcription)
            else:
                text = str(whisper_result)
        elif hasattr(whisper_result, 'segments'):
            text = " ".join([segment.text for segment in whisper_result.segments])
        else:
            text = str(whisper_result)
        
        # Create unified result format
        final_result = {
            "text": text,
            "model_used": "whisper",
            "device": device,
            "task": task,
            "success": True,
            "fallback_used": True,
            "whisper_result": whisper_result  # Include full whisper result for compatibility
        }
        
        # Add comprehensive metadata
        final_result["transcription_metadata"] = {
            "primary_model": "MERaLiON-AudioLLM-Whisper-SEA-LION",
            "actual_model": final_result["model_used"],
            "device": device,
            "task": task,
            "language": language,
            "meralion_used": False,
            "fallback_used": True,
            "meralion_fallback_enabled": meralion_fallback_enabled,
            "success": True
        }
        
        logger.info(f"Whisper fallback transcription successful using model: {final_result['model_used']}")
        return final_result
        
    except Exception as e:
        logger.error(f"Whisper fallback transcription failed: {e}")
        
        # Return error result
        error_result = {
            "text": "",
            "model_used": "none",
            "device": device,
            "task": task,
            "success": False,
            "error": f"All transcription methods failed. MERaLiON error: {result.get('error', 'Unknown') if 'result' in locals() else 'Unknown'}, Whisper error: {str(e)}"
        }
        
        # Add error metadata
        error_result["transcription_metadata"] = {
            "primary_model": "MERaLiON-AudioLLM-Whisper-SEA-LION",
            "actual_model": "none",
            "device": device,
            "task": task,
            "language": language,
            "meralion_used": False,
            "fallback_used": False,
            "meralion_fallback_enabled": meralion_fallback_enabled,
            "success": False,
            "meralion_error": result.get("error", "Unknown") if 'result' in locals() else "Unknown",
            "whisper_error": str(e)
        }
        
        return error_result



