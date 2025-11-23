import os
from dotenv import load_dotenv
from temporalio.common import RetryPolicy
from datetime import timedelta

load_dotenv()

class TemporalConfig:
    """Temporal configuration with retry policies."""

    TEMPORAL_SERVER_URL = os.getenv("TEMPORAL_SERVER_URL", "localhost:7233")
    TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")
    TEMPORAL_TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE", "whisperx-task-queue")
    
    # Retry policy configuration
    MAX_ATTEMPTS = int(os.getenv("TEMPORAL_MAX_ATTEMPTS", "3"))
    INITIAL_INTERVAL_SECONDS = int(os.getenv("TEMPORAL_INITIAL_INTERVAL", "5"))
    BACKOFF_COEFFICIENT = float(os.getenv("TEMPORAL_BACKOFF_COEFFICIENT", "2.0"))
    MAXIMUM_INTERVAL_SECONDS = int(os.getenv("TEMPORAL_MAX_INTERVAL", "300"))  # 5 minutes
    
    # Activity-specific timeouts
    TRANSCRIPTION_TIMEOUT_MINUTES = int(os.getenv("TRANSCRIPTION_TIMEOUT", "30"))
    ALIGNMENT_TIMEOUT_MINUTES = int(os.getenv("ALIGNMENT_TIMEOUT", "10"))
    DIARIZATION_TIMEOUT_MINUTES = int(os.getenv("DIARIZATION_TIMEOUT", "10"))
    SPEAKER_ASSIGNMENT_TIMEOUT_MINUTES = int(os.getenv("SPEAKER_ASSIGNMENT_TIMEOUT", "5"))

from temporalio.common import RetryPolicy
from datetime import timedelta

def get_default_retry_policy() -> RetryPolicy:
    """Get default retry policy for temporal activities."""
    return RetryPolicy(
        initial_interval=timedelta(seconds=TemporalConfig.INITIAL_INTERVAL_SECONDS),
        backoff_coefficient=TemporalConfig.BACKOFF_COEFFICIENT,
        maximum_interval=timedelta(seconds=TemporalConfig.MAXIMUM_INTERVAL_SECONDS),
        maximum_attempts=TemporalConfig.MAX_ATTEMPTS,
    )

def get_model_loading_retry_policy() -> RetryPolicy:
    """Get retry policy specifically for model loading operations."""
    return RetryPolicy(
        initial_interval=timedelta(seconds=10),  # Longer initial wait for model loading
        backoff_coefficient=1.5,  # Gentler backoff for model loading
        maximum_interval=timedelta(seconds=600),  # 10 minutes max
        maximum_attempts=5,  # More attempts for model loading
    )

def get_gpu_memory_retry_policy() -> RetryPolicy:
    """Get retry policy for GPU memory related failures."""
    return RetryPolicy(
        initial_interval=timedelta(seconds=15),  # Wait for GPU memory cleanup
        backoff_coefficient=1.2,  # Gentle backoff
        maximum_interval=timedelta(seconds=120),  # 2 minutes max
        maximum_attempts=3,
    )

config = TemporalConfig()
