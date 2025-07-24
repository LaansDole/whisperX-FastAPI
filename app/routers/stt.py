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

        # Save the file to a temporary location
        temp_audio_file = NamedTemporaryFile(suffix=original_extension, delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp_audio_file.write(chunk)

    logger.info("File downloaded and saved temporarily: %s", temp_audio_file.name)
    validate_extension(temp_audio_file.name, ALLOWED_EXTENSIONS)

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
        args=[temp_audio_file.name, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)

    return Response(identifier=handle.id, message="Workflow started")
