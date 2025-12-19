"""Temporal error handling utilities."""

import logging
from temporalio.exceptions import ApplicationError

logger = logging.getLogger(__name__)


class TemporalErrorHandler:
    """Utility class for handling temporal-specific errors."""

    @staticmethod
    def classify_error(error: Exception) -> tuple[bool, str]:
        """
        Classify an error as retryable or non-retryable.

        Args:
            error: The exception to classify

        Returns:
            Tuple of (is_retryable, error_category)
        """
        error_msg = str(error).lower()

        # Non-retryable errors
        if isinstance(error, ValueError):
            return False, "configuration"

        if isinstance(error, RuntimeError):
            if any(keyword in error_msg for keyword in ["authentication", "token", "401", "unauthorized"]):
                return False, "authentication"
            elif any(keyword in error_msg for keyword in ["terms of service", "agreement", "license"]):
                return False, "licensing"
            elif any(keyword in error_msg for keyword in ["model not found", "invalid model"]):
                return False, "model_configuration"

        # Retryable errors
        if isinstance(error, RuntimeError):
            if any(keyword in error_msg for keyword in ["network", "download", "connection", "timeout"]):
                return True, "network"
            elif any(keyword in error_msg for keyword in ["memory", "cuda", "gpu", "out of memory"]):
                return True, "gpu_memory"
            elif any(keyword in error_msg for keyword in ["temporary", "busy", "loading"]):
                return True, "temporary"

        # Default to retryable for unknown errors
        return True, "unknown"

    @staticmethod
    def create_application_error(
        error: Exception, context: str = "", force_non_retryable: bool = False
    ) -> ApplicationError:
        """
        Create an ApplicationError with appropriate retry settings.

        Args:
            error: The original exception
            context: Additional context for the error
            force_non_retryable: Force the error to be non-retryable

        Returns:
            ApplicationError with appropriate retry settings
        """
        is_retryable, category = TemporalErrorHandler.classify_error(error)

        if force_non_retryable:
            is_retryable = False

        error_message = f"{context}: {error}" if context else str(error)

        logger.error(
            f"Creating ApplicationError - Category: {category}, Retryable: {is_retryable}, Message: {error_message}"
        )

        return ApplicationError(error_message, non_retryable=not is_retryable, type=category)


def handle_activity_error(func):
    """
    Decorator for activity functions to handle errors consistently.

    Usage:
        @activity.defn
        @handle_activity_error
        async def my_activity():
            # activity code here
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApplicationError:
            # Re-raise ApplicationErrors as-is
            raise
        except Exception as e:
            # Convert other exceptions to ApplicationError
            activity_name = func.__name__
            logger.error(f"Error in activity {activity_name}: {e}")
            raise TemporalErrorHandler.create_application_error(e, context=f"Activity {activity_name} failed")

    return wrapper
