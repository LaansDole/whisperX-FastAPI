"""
This module provides API endpoints for speech-to-text services including transcription.

Alignment, diarization, and combining transcripts with diarization results.
"""

import json
import uuid
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import ValidationError

from ..compatibility import log_compatibility_warnings
from ..files import ALLOWED_EXTENSIONS, save_temporary_file, validate_extension
from ..logger import logger
from ..schemas import (
    AlignedTranscription,
    AlignmentParams,
    ASROptions,
    Device,
    DiarizationParams,
    DiarizationSegment,
    Response,
    Transcript,
    VADOptions,
    WhisperModelParams,
)
from ..temporal.manager import temporal_manager
from ..temporal.workflows import WhisperXWorkflow
from ..temporal.config import config
from ..whisperx_services import device

service_router = APIRouter()

# Log compatibility warnings on module import
log_compatibility_warnings()


@service_router.post(
    "/service/transcribe",
    tags=["Speech-2-Text services"],
    name="1. Transcribe",
)
async def transcribe(
    model_params: WhisperModelParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    file: UploadFile = File(..., description="Audio/video file to transcribe"),
) -> Response:
    """
    Transcribe an uploaded audio file.
    """
    logger.info("Received transcription request for file: %s", file.filename)
    validate_extension(file.filename, ALLOWED_EXTENSIONS)
    temp_file = save_temporary_file(file.file, file.filename)

    params = {
        "whisper_model_params": model_params.model_dump(),
        "asr_options": asr_options_params.model_dump(),
        "vad_options": vad_options_params.model_dump(),
    }

    client = await temporal_manager.get_client()
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)
    return Response(identifier=handle.id, message="Workflow started")


@service_router.post(
    "/service/align",
    tags=["Speech-2-Text services"],
    name="2. Align Transcript",
)
async def align(
    transcript: UploadFile = File(
        ..., description="Whisper style transcript json file"
    ),
    file: UploadFile = File(
        ..., description="Audio/video file which has been transcribed"
    ),
    device: Device = Query(
        default=device,
        description="Device to use for PyTorch inference",
    ),
    align_params: AlignmentParams = Depends(),
) -> Response:
    """
    Align a transcript with an audio file.
    """
    logger.info(
        "Received alignment request for file: %s and transcript: %s",
        file.filename,
        transcript.filename,
    )
    validate_extension(transcript.filename, {".json"})
    try:
        transcript_data = Transcript(**json.loads(transcript.file.read()))
        logger.info("Transcript loaded with %d segments", len(transcript_data.segments))
    except ValidationError as e:
        logger.error("Invalid JSON content in transcript file: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON content. {str(e)}")

    validate_extension(file.filename, ALLOWED_EXTENSIONS)
    temp_file = save_temporary_file(file.file, file.filename)

    params = {
        "alignment_params": align_params.model_dump(),
        "device": device,
    }

    client = await temporal_manager.get_client()
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)
    return Response(identifier=handle.id, message="Workflow started")


@service_router.post(
    "/service/diarize", tags=["Speech-2-Text services"], name="3. Diarize"
)
async def diarize(
    file: UploadFile = File(...),
    device: Device = Query(
        default=device,
        description="Device to use for PyTorch inference",
    ),
    diarize_params: DiarizationParams = Depends(),
) -> Response:
    """
    Perform diarization on an uploaded audio file.
    """
    logger.info("Received diarization request for file: %s", file.filename)
    validate_extension(file.filename, ALLOWED_EXTENSIONS)
    temp_file = save_temporary_file(file.file, file.filename)

    params = {
        "diarization_params": diarize_params.model_dump(),
        "device": device,
    }

    client = await temporal_manager.get_client()
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[temp_file, params],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)
    return Response(identifier=handle.id, message="Workflow started")


@service_router.post(
    "/service/combine",
    tags=["Speech-2-Text services"],
    name="4. Combine Transcript and Diarization result",
)
async def combine(
    aligned_transcript: UploadFile = File(...),
    diarization_result: UploadFile = File(...),
) -> Response:
    """
    Combine a transcript with diarization results.
    """
    logger.info(
        "Received combine request for aligned transcript: %s and diarization result: %s",
        aligned_transcript.filename,
        diarization_result.filename,
    )
    validate_extension(aligned_transcript.filename, {".json"})
    validate_extension(diarization_result.filename, {".json"})

    try:
        transcript = AlignedTranscription(**json.loads(aligned_transcript.file.read()))
    except ValidationError as e:
        logger.error("Invalid JSON content in aligned transcript file: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON content. {str(e)}")
    try:
        diarization_segments = []
        for item in json.loads(diarization_result.file.read()):
            diarization_segments.append(DiarizationSegment(**item))
    except ValidationError as e:
        logger.error("Invalid JSON content in diarization result file: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON content. {str(e)}")

    client = await temporal_manager.get_client()
    workflow_id = f"whisperx-workflow-{uuid.uuid4()}"
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[None, {"diarization_segments": [s.model_dump() for s in diarization_segments], "transcript": transcript.model_dump()}],
        id=workflow_id,
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    logger.info("Workflow started: ID %s", handle.id)
    return Response(identifier=handle.id, message="Workflow started")
