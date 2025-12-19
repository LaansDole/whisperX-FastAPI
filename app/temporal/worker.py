"""Temporal worker for processing audio workflows."""

from app.warnings_filter import filter_warnings

filter_warnings()

import asyncio  # noqa: E402
from temporalio.worker import Worker  # noqa: E402
from app.logger import logger  # noqa: E402

from .manager import temporal_manager  # noqa: E402
from .workflows import WhisperXWorkflow, MedicalRAGWorkflow, HybridAudioMedicalWorkflow  # noqa: E402
from .activities import (  # noqa: E402
    transcribe_activity,
    align_activity,
    diarize_activity,
    assign_speakers_activity,
    # Medical RAG activities
    phi_detection_activity,
    medical_entity_extraction_activity,
    soap_generation_activity,
    document_structuring_activity,
    embedding_generation_activity,
    vector_storage_activity,
    comprehensive_medical_processing_activity,
)
from .config import config  # noqa: E402


async def main():
    client = await temporal_manager.get_client()
    if not client:
        logger.error("Failed to connect to Temporal, worker cannot start.")
        return

    worker = Worker(
        client,
        task_queue=config.TEMPORAL_TASK_QUEUE,
        workflows=[
            WhisperXWorkflow,
            MedicalRAGWorkflow,
            HybridAudioMedicalWorkflow,
        ],
        activities=[
            # Core WhisperX activities
            transcribe_activity,
            align_activity,
            diarize_activity,
            assign_speakers_activity,
            # Medical RAG activities
            phi_detection_activity,
            medical_entity_extraction_activity,
            soap_generation_activity,
            document_structuring_activity,
            embedding_generation_activity,
            vector_storage_activity,
            comprehensive_medical_processing_activity,
        ],
    )
    logger.info("Starting Temporal worker with WhisperX and Medical RAG activities")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
