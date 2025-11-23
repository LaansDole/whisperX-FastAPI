"""Configuration module for the WhisperX FastAPI application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Config:
    """Configuration class for WhisperX FastAPI application settings."""

    LANG = os.getenv("DEFAULT_LANG", "en")
    HF_TOKEN = os.getenv("HF_TOKEN")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL")
    DIARIZATION_MODEL_PATH = os.getenv("DIARIZATION_MODEL_PATH")
    DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU for stability
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # Default to int8 for CPU compatibility
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
    LOG_LEVEL = os.getenv(
        "LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO"
    ).upper()
    
    AUDIO_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".awb",
        ".aac",
        ".ogg",
        ".oga",
        ".m4a",
        ".wma",
        ".amr",
    }
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv", ".mkv"}
    ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    
