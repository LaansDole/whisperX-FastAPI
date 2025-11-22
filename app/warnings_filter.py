"""This module filters specific warnings from various libraries used in the project."""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()


def filter_warnings():
    """Filter specific warnings from various libraries used in the project."""
    if os.getenv("FILTER_WARNING", "true").lower() == "true":
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="asteroid_filterbanks"
        )
        # these filters doesn't seam to work
        warnings.filterwarnings(
            "ignore", message="Model was trained with pyannote.audio*"
        )
        warnings.filterwarnings("ignore", message="Model was trained with torch*")
        # Filter multiprocessing resource tracker warnings (common with transformers/torch)
        warnings.filterwarnings(
            "ignore", message="resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown"
        )
        warnings.filterwarnings(
            "ignore", module="multiprocessing.resource_tracker"
        )
