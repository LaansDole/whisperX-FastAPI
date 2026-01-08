"""Temporal worker for processing audio workflows."""

from app.warnings_filter import filter_warnings

filter_warnings()

import asyncio  # noqa: E402
import concurrent.futures  # noqa: E402
import torch  # noqa: E402
from temporalio.worker import Worker  # noqa: E402
from app.logger import logger  # noqa: E402

from .manager import temporal_manager  # noqa: E402
from .workflows import WhisperXWorkflow, MedicalRAGWorkflow, HybridAudioMedicalWorkflow  # noqa: E402
from .stt_to_medical_workflow import STTToMedicalWorkflow  # noqa: E402
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
from .speaker_aware_medical_activities import (  # noqa: E402
    # Speaker-aware medical activities
    transform_to_dialogue_activity,
    detect_phi_in_dialogue_activity,
    extract_entities_with_speaker_activity,
    generate_soap_from_dialogue_activity,
    store_consultation_with_speaker_data_activity,
)
from .config import config  # noqa: E402


async def main():
    client = await temporal_manager.get_client()
    if not client:
        logger.error("Failed to connect to Temporal, worker cannot start.")
        return

    # CRITICAL: Configure concurrent activity execution
    # This enables multiple activities to run in parallel (3-5x throughput improvement)
    #
    # Smart defaults based on GPU availability:
    # - GPU detected: Default to 1 worker to prevent OOM errors
    # - CPU only: Default to 5 workers for high throughput
    max_activity_workers = config.get_max_activity_workers()

    logger.info("=" * 70)
    logger.info("Temporal Worker Configuration")
    logger.info("=" * 70)
    logger.info(f"Task queue: {config.TEMPORAL_TASK_QUEUE}")
    logger.info(f"Max concurrent activity workers: {max_activity_workers}")
    logger.info("Activity executor: ThreadPoolExecutor (concurrent execution enabled)")
    logger.info("Workflows registered: 4")
    logger.info("Activities registered: 18")

    # Log GPU information if available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU devices: {gpu_count}")
        logger.info(f"GPU memory per device: {gpu_memory:.2f} GB")
        logger.info("GPU memory optimization: Model caching enabled")
        logger.info(f"Default worker count for GPU: 1 (current: {max_activity_workers})")

        # Warn only if user explicitly overrode the safe default
        if max_activity_workers > 1:
            logger.warning("=" * 70)
            logger.warning("CUSTOM GPU WORKER CONFIGURATION")
            logger.warning(f"MAX_ACTIVITY_WORKERS set to {max_activity_workers} (recommended: 1 for GPU)")
            logger.warning("High concurrency may cause GPU OOM errors")
            logger.warning("Monitor GPU memory usage carefully during operation")
            logger.warning("=" * 70)
    else:
        logger.info("GPU: Not available (using CPU)")
        logger.info(f"Default worker count for CPU: 5 (current: {max_activity_workers})")

    # CRITICAL: Use ThreadPoolExecutor for concurrent activity execution
    # Do NOT use context manager - executor must stay alive during worker.run()
    activity_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_activity_workers,
        thread_name_prefix="temporal-activity-",
    )

    try:
        worker = Worker(
            client,
            task_queue=config.TEMPORAL_TASK_QUEUE,
            workflows=[
                WhisperXWorkflow,
                MedicalRAGWorkflow,
                HybridAudioMedicalWorkflow,
                STTToMedicalWorkflow,
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
                # Speaker-aware medical activities
                transform_to_dialogue_activity,
                detect_phi_in_dialogue_activity,
                extract_entities_with_speaker_activity,
                generate_soap_from_dialogue_activity,
                store_consultation_with_speaker_data_activity,
            ],
            activity_executor=activity_executor,  # CRITICAL: enables concurrent execution
            max_concurrent_activities=max_activity_workers,  # Limit concurrent activities
        )

        logger.info("=" * 70)
        logger.info("Worker started successfully")
        logger.info(f"Throughput: Up to {max_activity_workers} audio files can process concurrently")
        logger.info("=" * 70)

        # Run worker (blocking call until shutdown)
        await worker.run()
    finally:
        # Ensure executor is properly shut down when worker stops
        logger.info("Shutting down activity executor...")
        activity_executor.shutdown(wait=True)
        logger.info("Activity executor shut down successfully")


if __name__ == "__main__":
    asyncio.run(main())
