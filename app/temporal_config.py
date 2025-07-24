
import os
from dotenv import load_dotenv

load_dotenv()

class TemporalConfig:
    """Temporal configuration."""

    TEMPORAL_SERVER_URL = os.getenv("TEMPORAL_SERVER_URL", "localhost:7233")
    TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")
    TEMPORAL_TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE", "whisperx-task-queue")

config = TemporalConfig()
