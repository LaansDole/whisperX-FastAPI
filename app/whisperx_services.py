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
from .db import get_db_session
from .logger import logger  # Import the logger from the new module
from .schemas import AlignedTranscription, SpeechToTextProcessingParams, TaskStatus
from .tasks import update_task_status_in_db
from .transcript import filter_aligned_transcription

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
        model = load_model(
            model.value,
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


def process_audio_common(
    params: SpeechToTextProcessingParams, session: Session = Depends(get_db_session)
):
    """
    Process an audio clip to generate a transcript with speaker labels.

    Args:
        params (SpeechToTextProcessingParams): The speech-to-text processing parameters
        session (Session): Database session

    Returns:
        None: The result is saved in the transcription requests dict.
    """
    try:
        start_time = datetime.now()
        logger.info(
            "Starting speech-to-text processing for identifier: %s",
            params.identifier,
        )

        # Create a local reference to the session that won't be closed during processing
        local_session = session
        
        # Explicitly free memory before starting any processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Initial GPU memory state: {torch.cuda.memory_allocated() / 1024**2:.2f} MB used")

        logger.debug(
            "Transcription parameters - task: %s, language: %s, batch_size: %d, chunk_size: %d, model: %s, device: %s, device_index: %d, compute_type: %s, threads: %d",
            params.whisper_model_params.task,
            params.whisper_model_params.language,
            params.whisper_model_params.batch_size,
            params.whisper_model_params.chunk_size,
            params.whisper_model_params.model,
            params.whisper_model_params.device,
            params.whisper_model_params.device_index,
            params.whisper_model_params.compute_type,
            params.whisper_model_params.threads,
        )

        # Add more aggressive memory management
        import psutil
        import os
        
        # Check system memory before starting
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 85:
            logger.warning(f"High memory usage detected: {memory_info.percent}%")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Set memory fraction for CUDA to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory

        # Transcription step
        try:
            segments_before_alignment = transcribe_with_whisper(
                audio=params.audio,
                task=params.whisper_model_params.task.value,
                asr_options=params.asr_options,
                vad_options=params.vad_options,
                language=params.whisper_model_params.language,
                batch_size=params.whisper_model_params.batch_size,
                chunk_size=params.whisper_model_params.chunk_size,
                model=params.whisper_model_params.model,
                device=params.whisper_model_params.device,
                device_index=params.whisper_model_params.device_index,
                compute_type=params.whisper_model_params.compute_type,
                threads=params.whisper_model_params.threads,
            )
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
        finally:
            # Force cleanup even if there's an error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete

        # Alignment step
        logger.debug(
            "Alignment parameters - align_model: %s, interpolate_method: %s, return_char_alignments: %s, language_code: %s",
            params.alignment_params.align_model,
            params.alignment_params.interpolate_method,
            params.alignment_params.return_char_alignments,
            segments_before_alignment["language"],
        )
        
        try:
            segments_transcript = align_whisper_output(
                transcript=segments_before_alignment["segments"],
                audio=params.audio,
                language_code=segments_before_alignment["language"],
                align_model=params.alignment_params.align_model,
                interpolate_method=params.alignment_params.interpolate_method,
                return_char_alignments=params.alignment_params.return_char_alignments,
            )
            transcript = AlignedTranscription(**segments_transcript)
            # removing words within each segment that have missing start, end, or score values
            transcript = filter_aligned_transcription(transcript).model_dump()
        except Exception as e:
            logger.error(f"Error during alignment: {str(e)}")
            raise
        finally:
            # Cleanup after alignment
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Diarization step
        logger.debug(
            "Diarization parameters - device: %s, min_speakers: %s, max_speakers: %s",
            params.whisper_model_params.device,
            params.diarization_params.min_speakers,
            params.diarization_params.max_speakers,
        )
        
        try:
            diarization_segments = diarize(
                params.audio,
                device=params.whisper_model_params.device,
                min_speakers=params.diarization_params.min_speakers,
                max_speakers=params.diarization_params.max_speakers,
            )
        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            raise
        finally:
            # Cleanup after diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        logger.debug("Starting to combine transcript with diarization results")
        
        try:
            result = assign_word_speakers(diarization_segments, transcript)

            # Remove unnecessary data to reduce memory usage
            for segment in result["segments"]:
                if "words" in segment:
                    del segment["words"]

            if "word_segments" in result:
                del result["word_segments"]
        except Exception as e:
            logger.error(f"Error during speaker assignment: {str(e)}")
            raise

        logger.debug("Completed combining transcript with diarization results")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Completed speech-to-text processing for identifier: %s. Duration: %ss",
            params.identifier,
            duration,
        )

        # Make sure we've cleared as much memory as possible before updating DB
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Final GPU memory state: {torch.cuda.memory_allocated() / 1024**2:.2f} MB used")

        try:
            update_task_status_in_db(
                identifier=params.identifier,
                update_data={
                    "status": TaskStatus.completed,
                    "result": result,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                session=local_session,
            )
        except Exception as e:
            logger.error(f"Error updating task status in database: {str(e)}")
            # Try one last time with a new session
            try:
                from .db import get_db_session
                with get_db_session() as fallback_session:
                    update_task_status_in_db(
                        identifier=params.identifier,
                        update_data={
                            "status": TaskStatus.completed,
                            "result": result,
                            "duration": duration,
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                        session=fallback_session,
                    )
                    logger.info("Successfully updated task with fallback session")
            except Exception as inner_e:
                logger.critical(f"Critical database error: {str(inner_e)}")
            
    except (RuntimeError, ValueError, KeyError) as e:
        logger.error(
            "Speech-to-text processing failed for identifier: %s. Error: %s",
            params.identifier,
            str(e),
        )
        try:
            update_task_status_in_db(
                identifier=params.identifier,
                update_data={
                    "status": TaskStatus.failed,
                    "error": str(e),
                },
                session=session,
            )
        except Exception as db_error:
            logger.error(f"Failed to update task status in DB: {str(db_error)}")
    except MemoryError as e:
        logger.error(
            f"Task failed for identifier {params.identifier} due to out of memory. Error: {str(e)}"
        )
        try:
            update_task_status_in_db(
                identifier=params.identifier,
                update_data={"status": TaskStatus.failed, "error": f"Memory error: {str(e)}"},
                session=session,
            )
        except Exception as db_error:
            logger.error(f"Failed to update task status in DB: {str(db_error)}")
    except Exception as e:
        logger.error(f"Unexpected error for identifier {params.identifier}: {str(e)}")
        try:
            update_task_status_in_db(
                identifier=params.identifier,
                update_data={"status": TaskStatus.failed, "error": f"Unexpected error: {str(e)}"},
                session=session,
            )
        except Exception as db_error:
            logger.error(f"Failed to update task status in DB: {str(db_error)}")
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear references to potentially large objects
        params.audio = None
        result = None
        diarization_segments = None
        transcript = None
        segments_transcript = None
        segments_before_alignment = None

        # Multiple cleanup passes
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
