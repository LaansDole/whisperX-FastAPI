"""
This module contains the FastAPI routes for speech-to-text processing.

It includes endpoints for processing uploaded audio files and audio files from URLs.
"""

import logging
import os
import uuid

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query

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
from ..temporal.manager import temporal_manager
from ..temporal.workflows import WhisperXWorkflow
from ..temporal.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)

stt_router = APIRouter()

# Log compatibility warnings on module import
log_compatibility_warnings()


@stt_router.post(
    "/speech-to-text",
    tags=["Speech-2-Text"],
    response_model=Response,
    summary="Process audio/video file for speech-to-text",
    description="""
Upload an audio or video file to transcribe with optional alignment and speaker diarization.

**Supported Formats:**
- Audio: MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM
- Video: MP4, AVI, MOV, MKV, WEBM

**Processing Pipeline:**
1. **Transcription** - Convert speech to text using WhisperX
2. **Alignment** - Add precise word-level timestamps (optional)
3. **Diarization** - Identify different speakers (optional)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/speech-to-text?language=en&model=base&min_speakers=2&max_speakers=3" \\
  -F "file=@interview.mp3"
```

**Response:**
```json
{
  "identifier": "whisperx-workflow-a1b2c3d4-...",
  "message": "Workflow started"
}
```

**Next Steps:**
Use the returned workflow ID to check status:
`GET /temporal/workflow/{workflow_id}`

**Tips:**
- Use `base` model for balanced speed/accuracy
- Specify speaker count for better diarization
- Set `device=cuda` for GPU acceleration (if available)
    """,
)
async def speech_to_text(
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    file: UploadFile = File(...),
    patient_name: str = Query(..., description="Patient full name for HIPAA-compliant identification"),
) -> Response:
    """
    Process an uploaded audio file for speech-to-text conversion.

    Args:
        file: Audio/video file to process
        patient_name: Required patient full name (will be encrypted internally for HIPAA compliance)
    """
    logger.info("Received file upload request: %s", file.filename)

    validate_extension(file.filename, ALLOWED_EXTENSIONS)

    # Generate patient hash for HIPAA-compliant filenames
    from ..patients.filename_utils import generate_patient_file_id

    # Use the same hash function as filename_utils for consistency
    patient_hash = generate_patient_file_id(patient_name)

    logger.info(f"Processing upload for patient (hash: {patient_hash})")

    temp_file = save_temporary_file(file.file, file.filename, patient_name=patient_name)
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

    # Generate HIPAA-compliant workflow ID with patient hash
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workflow_id = f"whisperx-wf-pt_{patient_hash}-{timestamp}"

    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)

    # Store patient-workflow mapping in database
    from ..patients.mapping import store_patient_workflow

    store_patient_workflow(
        patient_name=patient_name,
        patient_hash=patient_hash,
        workflow_id=workflow_id,
        file_path=temp_file,
        department=None,  # TODO: Add department parameter
    )
    logger.info(f"Stored mapping: {patient_name} → {workflow_id}")

    return Response(identifier=handle.id, message="Workflow started")


@stt_router.post(
    "/speech-to-text-url",
    tags=["Speech-2-Text"],
    response_model=Response,
    summary="Process audio/video from URL",
    description="""
Download and process an audio or video file from a publicly accessible URL.

**Use Cases:**
- Process files from cloud storage (S3, Google Cloud, etc.)
- Transcribe YouTube videos or podcasts
- Batch processing from web sources

**Example Request:**
```bash
curl -X POST "http://localhost:8000/speech-to-text-url?language=vi&model=large-v3" \\
  -F "url=https://example.com/meeting-recording.mp3"
```

**Requirements:**
- URL must be publicly accessible (no authentication required)
- File must be in supported format
- Server must have network access to the URL

**Response:**
Returns a workflow ID for tracking processing status.

**Processing Time:**
Depends on file size, network speed, and server load. Check workflow status for progress.
    """,
)
async def speech_to_text_url(
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    url: str = Form(...),
    patient_name: str = Form(None, description="Optional patient name for HIPAA-compliant identification"),
) -> Response:
    """
    Process an audio file from a URL for speech-to-text conversion.
    
    Args:
        url: Public URL to audio/video file
        patient_name: Optional patient name for HIPAA-compliant workflow tracking
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

    # Generate HIPAA-compliant workflow ID if patient_name provided
    if patient_name:
        from ..patients.filename_utils import generate_patient_file_id
        from datetime import datetime

        patient_hash = generate_patient_file_id(patient_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_id = f"whisperx-wf-pt_{patient_hash}-{timestamp}"
        
        logger.info(f"Processing URL upload for patient (hash: {patient_hash})")
    else:
        # Fallback to random UUID for anonymous uploads
        workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
        patient_hash = None
        logger.info("Processing anonymous URL upload")

    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_audio_file_path, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)

    # Store patient-workflow mapping if patient_name provided
    if patient_name:
        from ..patients.mapping import store_patient_workflow

        store_patient_workflow(
            patient_name=patient_name,
            patient_hash=patient_hash,
            workflow_id=workflow_id,
            file_path=temp_audio_file_path,
            department=None,
        )
        logger.info(f"Stored mapping: {patient_name} → {workflow_id}")

    return Response(identifier=handle.id, message="Workflow started")
