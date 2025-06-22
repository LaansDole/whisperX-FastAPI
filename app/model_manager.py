"""
Utility module for managing model downloads and caching.

This module provides functions to safely download models from Hugging Face Hub
with proper error handling and retry logic.
"""

import os
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from .logger import logger

# Type variable for function decorator
T = TypeVar("T")

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_BACKOFF_FACTOR = 2  # exponential backoff factor

# Error messages
INTERNET_CONNECTION_MSG = (
    "Could not download model files. Please check your internet connection."
)
HF_TOKEN_MSG = (
    "Could not access the model. Make sure your Hugging Face token (HF_TOKEN) "
    "is valid and has permission to access this model."
)
MODEL_NOT_FOUND_MSG = "The specified model was not found on Hugging Face Hub."
REVISION_NOT_FOUND_MSG = "The specified model version/revision was not found."
GENERAL_DOWNLOAD_MSG = "An error occurred while downloading the model."


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retry_exceptions: tuple = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
    ),
) -> Callable:
    """
    Decorator to retry a function on exception with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which the delay increases for each retry
        retry_exceptions: Exception types that should trigger a retry

    Returns:
        Decorator function that adds retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = retry_delay

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(
                            "Retry attempt %d/%d for %s after %0.1f seconds",
                            attempt,
                            max_retries,
                            func.__name__,
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor

                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        str(e),
                    )
                    if attempt >= max_retries:
                        logger.error(
                            "Max retries reached for %s. Last error: %s",
                            func.__name__,
                            str(e),
                        )
                        break
                except Exception as e:
                    # Do not retry on exceptions not in retry_exceptions
                    logger.error("Non-retriable error in %s: %s", func.__name__, str(e))
                    raise

            # If we get here, we've exhausted our retries
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Failed after {max_retries} retries")

        return wrapper

    return decorator


@with_retry()
def safe_hf_hub_download(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    **kwargs,
) -> str:
    """
    Safely download a file from the HF Hub with retry logic.

    Args:
        repo_id: Hugging Face Hub repository ID
        filename: Name of the file to download
        revision: The git revision to use
        cache_dir: Cache directory
        force_download: Whether to force the download even if the file exists
        **kwargs: Additional arguments for hf_hub_download

    Returns:
        Local path to the downloaded file

    Raises:
        ValueError: With a descriptive message about what went wrong
    """
    try:
        logger.debug(
            "Downloading file %s from repo %s (revision: %s)",
            filename,
            repo_id,
            revision or "default",
        )
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            **kwargs,
        )
    except RepositoryNotFoundError:
        logger.error("Repository not found: %s", repo_id)
        raise ValueError(MODEL_NOT_FOUND_MSG)
    except RevisionNotFoundError:
        logger.error("Revision not found: %s for repo %s", revision, repo_id)
        raise ValueError(REVISION_NOT_FOUND_MSG)
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while downloading from Hugging Face Hub")
        raise ValueError(INTERNET_CONNECTION_MSG)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            logger.error("Authentication error: %s", str(e))
            raise ValueError(HF_TOKEN_MSG)
        logger.error("HTTP error: %s", str(e))
        raise ValueError(GENERAL_DOWNLOAD_MSG)
    except Exception as e:
        logger.error("Unexpected error during file download: %s", str(e))
        raise ValueError(f"{GENERAL_DOWNLOAD_MSG} Details: {str(e)}")


@with_retry()
def safe_snapshot_download(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Safely download all files from the HF Hub repository with retry logic.

    Args:
        repo_id: Hugging Face Hub repository ID
        revision: The git revision to use
        cache_dir: Cache directory
        **kwargs: Additional arguments for snapshot_download

    Returns:
        Local path to the downloaded repository

    Raises:
        ValueError: With a descriptive message about what went wrong
    """
    try:
        logger.debug(
            "Downloading repository snapshot %s (revision: %s)",
            repo_id,
            revision or "default",
        )
        return snapshot_download(
            repo_id=repo_id, 
            revision=revision, 
            cache_dir=cache_dir, 
            **kwargs
        )
    except RepositoryNotFoundError:
        logger.error("Repository not found: %s", repo_id)
        raise ValueError(MODEL_NOT_FOUND_MSG)
    except RevisionNotFoundError:
        logger.error("Revision not found: %s for repo %s", revision, repo_id)
        raise ValueError(REVISION_NOT_FOUND_MSG)
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while downloading from Hugging Face Hub")
        raise ValueError(INTERNET_CONNECTION_MSG)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            logger.error("Authentication error: %s", str(e))
            raise ValueError(HF_TOKEN_MSG)
        logger.error("HTTP error: %s", str(e))
        raise ValueError(GENERAL_DOWNLOAD_MSG)
    except Exception as e:
        logger.error("Unexpected error during repository download: %s", str(e))
        raise ValueError(f"{GENERAL_DOWNLOAD_MSG} Details: {str(e)}")


def validate_huggingface_token() -> bool:
    """
    Validate that the Hugging Face token is set and working.
    
    Returns:
        bool: True if token is valid, False otherwise
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN environment variable is not set")
        return False
    
    # Test token with a simple API call
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return True
        
        logger.warning(
            "HF_TOKEN validation failed with status code %d", 
            response.status_code
        )
        return False
    except requests.exceptions.RequestException as e:
        logger.warning("Error validating HF_TOKEN: %s", str(e))
        return False


def preload_models(models: list, token: Optional[str] = None) -> None:
    """
    Preload multiple models to ensure they're available in the cache.
    
    Args:
        models: List of model repo IDs to preload
        token: Optional Hugging Face token
    """
    for model in models:
        try:
            logger.info("Preloading model: %s", model)
            safe_snapshot_download(
                repo_id=model,
                token=token or os.environ.get("HF_TOKEN")
            )
            logger.info("Successfully preloaded model: %s", model)
        except Exception as e:
            logger.error("Failed to preload model %s: %s", model, str(e))
