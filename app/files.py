"""This module provides utility functions for file handling."""

import logging
import os
from tempfile import SpooledTemporaryFile

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


def save_temporary_file(
    temporary_file: SpooledTemporaryFile,
    original_filename: str,
    patient_name: str = None,
) -> str:
    """
    Save a SpooledTemporaryFile to a named temporary file with HIPAA-compliant naming.

    Args:
        temporary_file: The SpooledTemporaryFile object to save.
        original_filename: The original filename (used to extract extension).
        patient_name: Optional plain text patient name for HIPAA-compliant naming.
                     If None, a random UUID will be used instead.

    Returns:
        The path to the saved temporary file with HIPAA-compliant filename.
    """
    # Use shared uploads directory for Docker environment
    # This ensures files are accessible across containers
    uploads_dir = "/tmp/uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Extract original extension
    original_extension = os.path.splitext(original_filename)[1]

    # Import here to avoid circular dependency
    from app.patients.filename_utils import generate_anonymous_audio_filename

    # Generate HIPAA-compliant filename with patient hash (or random if no patient)
    unique_filename = generate_anonymous_audio_filename(original_extension, patient_name=patient_name)
    temp_filename = os.path.join(uploads_dir, unique_filename)

    # Write the contents of the SpooledTemporaryFile to the temporary file
    with open(temp_filename, "wb") as dest:
        dest.write(temporary_file.read())

    return temp_filename
