"""This module provides utility functions for file handling."""

import logging
import os

from fastapi import HTTPException

from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AUDIO_EXTENSIONS = Config.AUDIO_EXTENSIONS
VIDEO_EXTENSIONS = Config.VIDEO_EXTENSIONS
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS


def validate_extension(filename, allowed_extensions: dict):
    """
    Check the file extension of the given file and compare it if its is in the allowed AUDIO and VIDEO.

    Args:
        file (str): The path to the file.

    """
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in allowed_extensions:
        logger.info("Received file upload request: %s", filename)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension for file {filename} . Allowed: {allowed_extensions}",
        )


def check_file_extension(file):
    """
    Check the file extension of the given file and compare it if its is in the allowed AUDIO and VIDEO.

    Args:
        file (str): The path to the file.

    """
    validate_extension(file, ALLOWED_EXTENSIONS)


def save_temporary_file(temporary_file, original_filename):
    """
    Save the contents of a SpooledTemporaryFile to a named temporary file.

    Return the file path while preserving the original file extension.
    """
    # Extract the original file extension
    _, original_extension = os.path.splitext(original_filename)

    # Use shared uploads directory for Docker environment
    # This ensures files are accessible across containers
    uploads_dir = "/tmp/uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Create a unique filename with original extension
    import uuid
    unique_filename = f"{uuid.uuid4()}{original_extension}"
    temp_filename = os.path.join(uploads_dir, unique_filename)

    # Write the contents of the SpooledTemporaryFile to the temporary file
    with open(temp_filename, "wb") as dest:
        dest.write(temporary_file.read())

    return temp_filename
