"""Configuration module for the WhisperX FastAPI application."""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Config:
    """Configuration class for WhisperX FastAPI application settings."""

    # Timezone Configuration
    TIMEZONE_NAME = os.getenv("TIMEZONE", "Asia/Bangkok")  # UTC+7 for Vietnam
    TIMEZONE = ZoneInfo(TIMEZONE_NAME)

    # Core WhisperX Configuration
    LANG = os.getenv("DEFAULT_LANG", "en")
    HF_TOKEN = os.getenv("HF_TOKEN")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL")
    DIARIZATION_MODEL_PATH = os.getenv("DIARIZATION_MODEL_PATH")
    DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU for stability
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # Default to int8 for CPU compatibility
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO").upper()

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

    # LM Studio Configuration
    LM_STUDIO_ENABLED = os.getenv("LM_STUDIO_ENABLED", "false").lower() == "true"
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "120"))
    LM_STUDIO_TEMPERATURE = float(os.getenv("LM_STUDIO_TEMPERATURE", "0.1"))
    LM_STUDIO_MAX_TOKENS = int(os.getenv("LM_STUDIO_MAX_TOKENS", "2048"))
    LM_STUDIO_MAX_RETRIES = int(os.getenv("LM_STUDIO_MAX_RETRIES", "3"))

    # LLM Model Configuration
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")  # Leave empty to use loaded model

    # Medical RAG Configuration
    MEDICAL_RAG_ENABLED = os.getenv("MEDICAL_RAG_ENABLED", "false").lower() == "true"
    ENABLE_PHI_DETECTION = os.getenv("ENABLE_PHI_DETECTION", "true").lower() == "true"
    ENABLE_ENTITY_EXTRACTION = os.getenv("ENABLE_ENTITY_EXTRACTION", "true").lower() == "true"
    ENABLE_SOAP_GENERATION = os.getenv("ENABLE_SOAP_GENERATION", "true").lower() == "true"
    ENABLE_DOCUMENT_STRUCTURING = os.getenv("ENABLE_DOCUMENT_STRUCTURING", "true").lower() == "true"
    ENABLE_VECTOR_STORAGE = os.getenv("ENABLE_VECTOR_STORAGE", "false").lower() == "true"

    # Medical Processing Configuration
    MEDICAL_BATCH_SIZE = int(os.getenv("MEDICAL_BATCH_SIZE", "1"))  # Process one transcript at a time
    MEDICAL_MAX_RETRIES = int(os.getenv("MEDICAL_MAX_RETRIES", "2"))
    MEDICAL_TIMEOUT_MINUTES = int(os.getenv("MEDICAL_TIMEOUT_MINUTES", "10"))

    # Vector Database Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_storage")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # Default for nomic-embed-text-v1.5
    VECTOR_INDEX_TYPE = os.getenv("VECTOR_INDEX_TYPE", "IndexFlatL2")  # Start with simple, can upgrade to IndexIVFFlat
    VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "10"))

    # Neo4j Knowledge Graph Configuration
    NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "false").lower() == "true"
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j-password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    NEO4J_TIMEOUT = int(os.getenv("NEO4J_TIMEOUT", "30"))

    # Neo4j Memory Configuration (for Docker deployment)
    NEO4J_HEAP_INITIAL = os.getenv("NEO4J_HEAP_INITIAL", "4G")
    NEO4J_HEAP_MAX = os.getenv("NEO4J_HEAP_MAX", "4G")
    NEO4J_PAGECACHE_SIZE = os.getenv("NEO4J_PAGECACHE_SIZE", "4G")

    # HIPAA Compliance Configuration
    HIPAA_ENCRYPTION_KEY = os.getenv("HIPAA_ENCRYPTION_KEY")  # Must be set in production
    HIPAA_SALT = os.getenv("HIPAA_SALT", "default_salt_change_in_production")
    HIPAA_AUDIT_LOG_PATH = os.getenv("HIPAA_AUDIT_LOG_PATH", "./audit_logs")
    HIPAA_MINIMUM_NECESSARY = os.getenv("HIPAA_MINIMUM_NECESSARY", "true").lower() == "true"

    # Authentication & Authorization
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true"

    # Database Configuration (for medical metadata)
    MEDICAL_DB_PATH = os.getenv("MEDICAL_DB_PATH", "./medical_metadata.db")

    # Patient Workflow Mappings Database
    PATIENT_DB_PATH = os.getenv("PATIENT_DB_PATH", "./data/patient_mappings.db")

    # Performance Configuration
    ENABLE_PARALLEL_MEDICAL_PROCESSING = os.getenv("ENABLE_PARALLEL_MEDICAL_PROCESSING", "true").lower() == "true"
    MEDICAL_PROCESSING_WORKERS = int(os.getenv("MEDICAL_PROCESSING_WORKERS", "2"))  # Conservative for medical workloads

    # Feature Flags for Gradual Rollout
    REAL_TIME_ENTITY_EXTRACTION = os.getenv("REAL_TIME_ENTITY_EXTRACTION", "false").lower() == "true"
    BATCH_ENTITY_EXTRACTION = os.getenv("BATCH_ENTITY_EXTRACTION", "true").lower() == "true"
    ADVANCED_MEDICAL_REASONING = os.getenv("ADVANCED_MEDICAL_REASONING", "false").lower() == "true"

    # Temporal Configuration for Medical Workflows
    MEDICAL_ACTIVITY_TIMEOUT_MINUTES = int(os.getenv("MEDICAL_ACTIVITY_TIMEOUT_MINUTES", "15"))
    MEDICAL_WORKFLOW_TIMEOUT_MINUTES = int(os.getenv("MEDICAL_WORKFLOW_TIMEOUT_MINUTES", "30"))

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

        if cls.NEO4J_ENABLED:
            if not cls.NEO4J_URI:
                missing.append("NEO4J_URI required when NEO4J_ENABLED is true")
            if not cls.NEO4J_PASSWORD or cls.NEO4J_PASSWORD == "neo4j-password":
                missing.append("NEO4J_PASSWORD must be changed from default in production")

        return missing

    @classmethod
    def is_medical_processing_enabled(cls) -> bool:
        """Check if any medical processing features are enabled."""
        return cls.MEDICAL_RAG_ENABLED and cls.LM_STUDIO_ENABLED
