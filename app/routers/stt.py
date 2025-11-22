"""
This module contains the FastAPI routes for speech-to-text processing.

It includes endpoints for processing uploaded audio files and audio files from URLs.
"""

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Optional
import uuid

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..compatibility import log_compatibility_warnings
from ..files import ALLOWED_EXTENSIONS, save_temporary_file, validate_extension
from ..logger import logger
from ..schemas import (
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    Response,
    VADOptions,
    WhisperModelParams,
)
from ..temporal_manager import temporal_manager
from ..temporal_workflows import WhisperXWorkflow
from ..temporal_config import config

# Configure logging
logging.basicConfig(level=logging.INFO)

stt_router = APIRouter()

# Log compatibility warnings on module import
log_compatibility_warnings()


@stt_router.post("/speech-to-text", tags=["Speech-2-Text"])
async def speech_to_text(
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    file: UploadFile = File(...),
) -> Response:
    """
    Process an uploaded audio file for speech-to-text conversion.
    """
    logger.info("Received file upload request: %s", file.filename)

    validate_extension(file.filename, ALLOWED_EXTENSIONS)

    temp_file = save_temporary_file(file.file, file.filename)
    logger.info("%s saved as temporary file: %s", file.filename, temp_file)

    params = {
        "whisper_model_params": model_params.model_dump(),
        "alignment_params": align_params.model_dump(),
        "diarization_params": diarize_params.model_dump(),
        "asr_options": asr_options_params.model_dump(),
        "vad_options": vad_options_params.model_dump(),
    }

    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)

    return Response(identifier=handle.id, message="Workflow started")


@stt_router.post("/speech-to-text-url", tags=["Speech-2-Text"])
async def speech_to_text_url(
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    url: str = Form(...),
) -> Response:
    """
    Process an audio file from a URL for speech-to-text conversion.
    """
    logger.info("Received URL for processing: %s", url)

    # Extract filename from HTTP response headers or URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        # Check for filename in Content-Disposition header
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            # Fall back to extracting from the URL path
            filename = os.path.basename(url)

        # Get the file extension
        _, original_extension = os.path.splitext(filename)

        # Use shared uploads directory for Docker environment
        # This ensures files are accessible across containers
        uploads_dir = "/tmp/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Create a unique filename with original extension
        unique_filename = f"{uuid.uuid4()}{original_extension}"
        temp_audio_file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save the file to the shared location
        with open(temp_audio_file_path, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        logger.info("File downloaded and saved temporarily: %s", temp_audio_file_path)
        validate_extension(temp_audio_file_path, ALLOWED_EXTENSIONS)

    params = {
        "whisper_model_params": model_params.model_dump(),
        "alignment_params": align_params.model_dump(),
        "diarization_params": diarize_params.model_dump(),
        "asr_options": asr_options_params.model_dump(),
        "vad_options": vad_options_params.model_dump(),
    }

    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_audio_file_path, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)

    return Response(identifier=handle.id, message="Workflow started")




@stt_router.post("/speech-to-text-meralion", tags=["Speech-2-Text"])
async def speech_to_text_meralion(
    file: UploadFile = File(...),
    language: str = Form(default="sg", description="Target language code (e.g., 'sg', 'en', 'zh', 'vi')"),
    task: str = Form(default="transcribe", description="Task: 'transcribe' or 'translate'"),
    device: str = Form(default="auto", description="Device: 'auto', 'cpu', or 'cuda'"),
    use_meralion: bool = Form(default=True, description="Use MERaLiON model as primary transcription engine"),
    meralion_fallback_enabled: bool = Form(default=True, description="Enable fallback to Whisper models if MERaLiON fails"),
    meralion_max_new_tokens: int = Form(default=256, description="Maximum tokens for MERaLiON generation"),
    return_word_alignments: bool = Form(default=False, description="Return word-level timestamps"),
    return_char_alignments: bool = Form(default=False, description="Return character-level timestamps"),
    min_speakers: Optional[int] = Form(default=None, description="Minimum number of speakers (optional)"),
    max_speakers: Optional[int] = Form(default=None, description="Maximum number of speakers (optional)"),
) -> Response:
    """
    Process audio using MERaLiON-AudioLLM-Whisper-SEA-LION model with fallback to Whisper models.
    
    This endpoint prioritizes the MERaLiON model for transcription and translation tasks,
    with automatic fallback to traditional Whisper models when needed. Optimized for
    Singapore English ('sg') by default.
    """
    logger.info(
        "Received MERaLiON file upload request: %s for language: %s, "
        "use_meralion: %s, device: %s",
        file.filename, language, use_meralion, device
    )

    validate_extension(file.filename, ALLOWED_EXTENSIONS)

    temp_file = save_temporary_file(file.file, file.filename)
    logger.info("%s saved as temporary file: %s", file.filename, temp_file)

    # Build params with MERaLiON configuration
    params = {
        "whisper_model_params": {
            "language": language,
            "task": task,
            "model": None,  # Will be auto-selected by MERaLiON
            "device": device,
            "batch_size": 8,
            "threads": 0,
            "compute_type": None,  # Will be auto-selected based on device
            "device_index": 0,
            "chunk_size": 20,
            "use_meralion": use_meralion,
            "meralion_fallback_enabled": meralion_fallback_enabled,
            "meralion_max_new_tokens": meralion_max_new_tokens,
        },
        "alignment_params": {
            "align_model": None,
            "interpolate_method": "nearest",
            "return_char_alignments": return_char_alignments,
            "return_word_alignments": return_word_alignments,
            "device": device,
        },
        "diarization_params": {
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "device": device,
        },
        "asr_options": {
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "temperatures": 0.0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "initial_prompt": None,
            "suppress_tokens": [-1],
            "suppress_numerals": False,
            "hotwords": None,
        },
        "vad_options": {
            "vad_onset": 0.500,
            "vad_offset": 0.363,
        },
        "meralion_enabled": use_meralion,
    }

    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    workflow_id = f"meralion-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("MERaLiON workflow started: ID %s", handle.id)

    return Response(identifier=handle.id, message="MERaLiON workflow started")
