"""Configuration module for the WhisperX FastAPI application."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def _load_yaml_config() -> dict:
    """Load configuration from config.yaml file.

    Raises:
        RuntimeError: If config.yaml exists but cannot be read or parsed.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except OSError as e:
            raise RuntimeError(f"Failed to read config.yaml: {e}") from e
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse config.yaml: {e}") from e
    return {}


# Load YAML config once at module import
_yaml = _load_yaml_config()


def _get_yaml(section: str, key: str, default=None):
    """Get a value from YAML config with fallback to default."""
    return _yaml.get(section, {}).get(key, default)


def _get_yaml_nested(section: str, subsection: str, key: str, default=None):
    """Get a nested value from YAML config with fallback to default."""
    return _yaml.get(section, {}).get(subsection, {}).get(key, default)


class Config:
    """Configuration class for WhisperX FastAPI application settings."""

    # ========== FROM .env (secrets/environment-specific) ==========
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO").upper()
    TIMEZONE_NAME = os.getenv("TIMEZONE", "Asia/Ho_Chi_Minh")
    TIMEZONE = ZoneInfo(TIMEZONE_NAME)

    # Feature flags
    FILTER_WARNING = os.getenv("FILTER_WARNING", "true").lower() == "true"
    DB_FRESH_START = os.getenv("DB_FRESH_START", "false").lower() == "true"

    # HIPAA (secrets)
    HIPAA_ENCRYPTION_KEY = os.getenv("HIPAA_ENCRYPTION_KEY")
    HIPAA_SALT = os.getenv("HIPAA_SALT", "default_salt_change_in_production")
    # Prevent insecure default HIPAA_SALT from being used in production
    if ENVIRONMENT == "production" and HIPAA_SALT == "default_salt_change_in_production":
        raise RuntimeError(
            "HIPAA_SALT must be set to a secure, non-default value in production. "
            "Current value is the insecure default 'default_salt_change_in_production'."
        )
    # Optional paths from env
    DIARIZATION_MODEL_PATH = os.getenv("DIARIZATION_MODEL_PATH")

    # ========== LM STUDIO (from .env - required for medical) ==========
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")
    LM_STUDIO_MAX_TOKENS = int(os.getenv("LM_STUDIO_MAX_TOKENS", "8192"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-reranker-v2-m3")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    # LM Studio advanced settings (from YAML)
    LM_STUDIO_ENABLED = _get_yaml_nested("medical", "lm_studio", "enabled", False)
    LM_STUDIO_TIMEOUT = _get_yaml_nested("medical", "lm_studio", "timeout", 120)
    LM_STUDIO_TEMPERATURE = _get_yaml_nested("medical", "lm_studio", "temperature", 0.1)
    LM_STUDIO_MAX_RETRIES = _get_yaml_nested("medical", "lm_studio", "max_retries", 3)

    # ========== FROM config.yaml (runtime defaults) ==========

    # WhisperX
    LANG = _get_yaml("whisperx", "language", "en")
    WHISPER_MODEL = _get_yaml("whisperx", "model", "base")
    DEVICE = os.getenv("DEVICE", "cpu")
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

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

    # Medical RAG
    MEDICAL_RAG_ENABLED = _get_yaml("medical", "enabled", False)
    ENABLE_PHI_DETECTION = _get_yaml_nested("medical", "features", "phi_detection", True)
    ENABLE_ENTITY_EXTRACTION = _get_yaml_nested("medical", "features", "entity_extraction", True)
    ENABLE_SOAP_GENERATION = _get_yaml_nested("medical", "features", "soap_generation", True)
    ENABLE_DOCUMENT_STRUCTURING = _get_yaml_nested("medical", "features", "document_structuring", True)
    ENABLE_VECTOR_STORAGE = _get_yaml_nested("medical", "features", "vector_storage", False)

    # Medical Processing
    MEDICAL_BATCH_SIZE = 1
    MEDICAL_MAX_RETRIES = 2
    MEDICAL_TIMEOUT_MINUTES = 10

    # Vector Database
    VECTOR_DB_PATH = _get_yaml("vector_db", "path", "./vector_storage")
    VECTOR_INDEX_TYPE = _get_yaml("vector_db", "index_type", "IndexFlatL2")
    VECTOR_SEARCH_LIMIT = _get_yaml("vector_db", "search_limit", 10)

    # HIPAA (non-secret defaults)
    HIPAA_AUDIT_LOG_PATH = "./audit_logs"
    HIPAA_MINIMUM_NECESSARY = True

    # Authentication (disabled for POC)
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    ENABLE_AUTHENTICATION = False

    # Database paths
    MEDICAL_DB_PATH = _get_yaml("database", "medical_db_path", "./medical_metadata.db")
    PATIENT_DB_PATH = _get_yaml("database", "patient_db_path", "./data/patient_mappings.db")

    # Performance
    ENABLE_PARALLEL_MEDICAL_PROCESSING = _get_yaml("performance", "parallel_medical_processing", True)
    MEDICAL_PROCESSING_WORKERS = _get_yaml("performance", "medical_processing_workers", 2)

    # Feature Flags (hardcoded for POC)
    REAL_TIME_ENTITY_EXTRACTION = False
    BATCH_ENTITY_EXTRACTION = True
    ADVANCED_MEDICAL_REASONING = False

    # Temporal timeouts (from YAML)
    MEDICAL_ACTIVITY_TIMEOUT_MINUTES = _get_yaml_nested("temporal", "timeouts", "medical_activity_minutes", 15)
    MEDICAL_WORKFLOW_TIMEOUT_MINUTES = _get_yaml_nested("temporal", "timeouts", "medical_workflow_minutes", 30)

    @classmethod
    def validate_medical_config(cls) -> list[str]:
        """Validate medical configuration and return list of required missing settings."""
        missing = []

        if cls.MEDICAL_RAG_ENABLED:
            if cls.LM_STUDIO_ENABLED and not cls.LM_STUDIO_BASE_URL:
                missing.append("LM_STUDIO_BASE_URL required when MEDICAL_RAG_ENABLED and LM_STUDIO_ENABLED")

            if cls.ENABLE_VECTOR_STORAGE and not cls.VECTOR_DB_PATH:
                missing.append("VECTOR_DB_PATH required when ENABLE_VECTOR_STORAGE is true")

            if cls.ENABLE_AUTHENTICATION and not cls.JWT_SECRET_KEY:
                missing.append("JWT_SECRET_KEY required when ENABLE_AUTHENTICATION is true")

        return missing

    @classmethod
    def is_medical_processing_enabled(cls) -> bool:
        """Check if any medical processing features are enabled."""
        return cls.MEDICAL_RAG_ENABLED and cls.LM_STUDIO_ENABLED
