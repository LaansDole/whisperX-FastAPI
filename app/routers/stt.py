"""
This module contains the FastAPI routes for speech-to-text processing.

It includes endpoints for processing uploaded audio files and audio files from URLs.
"""

import logging
import os
from tempfile import NamedTemporaryFile
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
from ..temporal_workflows import WhisperXWorkflow, WhisperXOptimizedWorkflow
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


@stt_router.post("/speech-to-text-optimized", tags=["Speech-2-Text"])
async def speech_to_text_optimized(
    language: str = Form(..., description="Target language code (e.g., 'en', 'zh', 'ja')"),
    task: str = Form(default="transcribe", description="Task: 'transcribe' or 'translate'"),
    device: str = Form(default="cuda", description="Device: 'cuda' or 'cpu'"),
    batch_size: int = Form(default=8, description="Batch size for processing"),
    threads: int = Form(default=0, description="Number of CPU threads"),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    file: UploadFile = File(...),
    override_model: str = Form(default=None, description="Override auto-selected model"),
) -> Response:
    """
    Process audio with language-optimized model selection.
    
    This endpoint automatically selects the optimal Whisper model for the specified language
    based on AudioBench performance data, along with optimal compute settings and diarization parameters.
    """
    logger.info("Received optimized file upload request: %s for language: %s", file.filename, language)

    validate_extension(file.filename, ALLOWED_EXTENSIONS)

    temp_file = save_temporary_file(file.file, file.filename)
    logger.info("%s saved as temporary file: %s", file.filename, temp_file)

    # Build params with optimized model selection
    params = {
        "whisper_model_params": {
            "language": language,
            "task": task,
            "model": override_model,  # Will be overridden by optimization if None
            "device": device,
            "batch_size": batch_size,
            "threads": threads,
            "compute_type": None,  # Will be auto-selected
            "device_index": 0,
            "chunk_size": 20,
        },
        "alignment_params": align_params.model_dump(),
        "diarization_params": diarize_params.model_dump(),
        "asr_options": asr_options_params.model_dump(),
        "vad_options": vad_options_params.model_dump(),
        "optimization_enabled": True,
        "override_model": override_model,
    }

    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    workflow_id = f"whisperx-optimized-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXOptimizedWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Optimized workflow started: ID %s", handle.id)

    return Response(identifier=handle.id, message="Optimized workflow started")
