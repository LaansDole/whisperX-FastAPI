from datetime import timedelta
from pathlib import Path

import yaml
from temporalio.common import RetryPolicy


def _load_yaml_config() -> dict:
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# Load YAML config once at module import
_yaml = _load_yaml_config()


def _get_temporal(key: str, default=None):
    """Get a value from temporal section of YAML config."""
    return _yaml.get("temporal", {}).get(key, default)


def _get_temporal_nested(section: str, key: str, default=None):
    """Get a nested value from temporal section of YAML config."""
    return _yaml.get("temporal", {}).get(section, {}).get(key, default)


class TemporalConfig:
    """Temporal configuration with retry policies."""

    # All settings from config.yaml
    TEMPORAL_SERVER_URL = _get_temporal("server_url", "localhost:7233")
    TEMPORAL_NAMESPACE = _get_temporal("namespace", "default")
    TEMPORAL_TASK_QUEUE = _get_temporal("task_queue", "whisperx-task-queue")

    # Retry policy configuration
    MAX_ATTEMPTS = _get_temporal_nested("retry", "max_attempts", 3)
    INITIAL_INTERVAL_SECONDS = _get_temporal_nested("retry", "initial_interval", 5)
    BACKOFF_COEFFICIENT = _get_temporal_nested("retry", "backoff_coefficient", 2.0)
    MAXIMUM_INTERVAL_SECONDS = _get_temporal_nested("retry", "max_interval", 300)

    # Activity-specific timeouts
    TRANSCRIPTION_TIMEOUT_MINUTES = _get_temporal_nested("timeouts", "transcription_minutes", 30)
    ALIGNMENT_TIMEOUT_MINUTES = _get_temporal_nested("timeouts", "alignment_minutes", 10)
    DIARIZATION_TIMEOUT_MINUTES = _get_temporal_nested("timeouts", "diarization_minutes", 10)
    SPEAKER_ASSIGNMENT_TIMEOUT_MINUTES = _get_temporal_nested("timeouts", "speaker_assignment_minutes", 5)

    # Worker concurrency (computed at runtime based on GPU availability)
    # Default: 1 for GPU (prevent OOM), 5 for CPU
    @classmethod
    def get_max_activity_workers(cls) -> int:
        """Get max activity workers based on hardware."""
        import torch
        if torch.cuda.is_available():
            return 1  # Safe default for GPU
        return 5  # Optimal for CPU


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
        initial_interval=timedelta(seconds=10),
        backoff_coefficient=1.5,
        maximum_interval=timedelta(seconds=600),
        maximum_attempts=5,
    )


def get_gpu_memory_retry_policy() -> RetryPolicy:
    """Get retry policy for GPU memory related failures."""
    return RetryPolicy(
        initial_interval=timedelta(seconds=15),
        backoff_coefficient=1.2,
        maximum_interval=timedelta(seconds=120),
        maximum_attempts=3,
    )


config = TemporalConfig()
