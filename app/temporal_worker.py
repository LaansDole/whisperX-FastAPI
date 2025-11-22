
import asyncio
from temporalio.worker import Worker
from .temporal_manager import temporal_manager
from .temporal_workflows import WhisperXWorkflow
from .activities import (
    transcribe_activity,
    transcribe_meralion_activity,
    align_activity,
    diarize_activity,
    assign_speakers_activity,
)
from .temporal_config import config
from .logger import logger


async def main():
    client = await temporal_manager.get_client()
    if not client:
        logger.error("Failed to connect to Temporal, worker cannot start.")
        return

    worker = Worker(
        client,
        task_queue=config.TEMPORAL_TASK_QUEUE,
        workflows=[WhisperXWorkflow],
        activities=[
            transcribe_activity,
            transcribe_meralion_activity,
            align_activity,
            diarize_activity,
            assign_speakers_activity,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
