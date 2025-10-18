from typing import Dict
from enum import Enum
from .schemas import WhisperModel

class LanguageBestInSlot(Enum):
    """Language-specific Best-in-Slot model configurations based on AudioBench results."""
    
    # Southeast Asian languages - MERaLiON AudioLLM (AudioBench best performer)
    SOUTHEAST_ASIA = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"  # Best for: Vietnamese, Singlish, English, Mandarin, Cantonese
    
    # Default fallback
    DEFAULT = WhisperModel.large_v3_turbo  # Best general-purpose model

# Language code mapping to our enum - Only include SEA languages that benefit from MERaLiON
LANGUAGE_MODEL_MAPPING: Dict[str, LanguageBestInSlot] = {
    # Southeast Asian languages supported by MERaLiON SEA-LION (AudioBench best performer)
    "vi": LanguageBestInSlot.SOUTHEAST_ASIA,      # Vietnamese
    "vie": LanguageBestInSlot.SOUTHEAST_ASIA,     # Vietnamese
    
    "en": LanguageBestInSlot.SOUTHEAST_ASIA,      # English (especially Singapore English)
    "eng": LanguageBestInSlot.SOUTHEAST_ASIA,     # English
    
    "zh": LanguageBestInSlot.SOUTHEAST_ASIA,      # Mandarin Chinese
    "cmn": LanguageBestInSlot.SOUTHEAST_ASIA,     # Mandarin
    "yue": LanguageBestInSlot.SOUTHEAST_ASIA,     # Cantonese
    
    # Singapore English variants
    "en-sg": LanguageBestInSlot.SOUTHEAST_ASIA,   # Singapore English
    "sg-en": LanguageBestInSlot.SOUTHEAST_ASIA,   # Singapore English
}

# Compute type optimization per language
COMPUTE_TYPE_MAPPING: Dict[str, str] = {
    # Southeast Asian languages - MERaLiON SEA-LION optimized for mixed precision
    "vi": "float16",     # Vietnamese
    "vie": "float16",    # Vietnamese
    "en": "float16",     # English (MERaLiON optimized)
    "eng": "float16",    # English
    "zh": "float16",     # Mandarin Chinese
    "cmn": "float16",    # Mandarin
    "yue": "float16",    # Cantonese
    "en-sg": "float16",  # Singapore English
    "sg-en": "float16",  # Singapore English
    
    # Default fallback
    "default": "float16"
}

def get_best_model_for_language(language_code: str) -> str:
    """
    Get the best model for a given language code.
    
    Args:
        language_code: ISO language code (e.g., 'en', 'zh', 'vi')
        
    Returns:
        str: The optimal model for the language
    """
    # Normalize language code
    lang_code = language_code.lower().strip()
    
    # Try exact match first
    if lang_code in LANGUAGE_MODEL_MAPPING:
        return LANGUAGE_MODEL_MAPPING[lang_code].value
    
    # Try 2-letter code if 3-letter provided
    if len(lang_code) == 3:
        for code, model in LANGUAGE_MODEL_MAPPING.items():
            if len(code) == 2 and lang_code.startswith(code):
                return model.value
    
    # Fallback to default Whisper model
    return LanguageBestInSlot.DEFAULT.value

def get_optimal_compute_type(language_code: str, device: str = "cuda") -> str:
    """
    Get the optimal compute type for a given language and device.
    
    Args:
        language_code: ISO language code
        device: Device type ('cuda' or 'cpu')
        
    Returns:
        str: Optimal compute type ('float16', 'float32', 'int8')
    """
    # CPU always uses int8 for performance
    if device == "cpu":
        return "int8"
    
    # Get language-specific compute type
    lang_code = language_code.lower().strip()
    compute_type = COMPUTE_TYPE_MAPPING.get(lang_code, COMPUTE_TYPE_MAPPING["default"])
    
    return compute_type

def get_diarization_config_for_language(language_code: str) -> dict:
    """
    Get language-specific diarization configuration.
    
    Args:
        language_code: ISO language code
        
    Returns:
        dict: Diarization configuration
    """
    lang_code = language_code.lower().strip()
    
    # Language-specific diarization settings based on cultural conversation patterns
    configs = {
        # Southeast Asian languages - often more speakers in conversations
        "vi": {"min_speakers": 1, "max_speakers": 12, "confidence_threshold": 0.45},  # Vietnamese
        "vie": {"min_speakers": 1, "max_speakers": 12, "confidence_threshold": 0.45}, # Vietnamese
        "en": {"min_speakers": 1, "max_speakers": 10, "confidence_threshold": 0.5},   # English
        "eng": {"min_speakers": 1, "max_speakers": 10, "confidence_threshold": 0.5},  # English
        "zh": {"min_speakers": 1, "max_speakers": 15, "confidence_threshold": 0.45},  # Mandarin
        "cmn": {"min_speakers": 1, "max_speakers": 15, "confidence_threshold": 0.45}, # Mandarin
        "yue": {"min_speakers": 1, "max_speakers": 15, "confidence_threshold": 0.45}, # Cantonese
        "en-sg": {"min_speakers": 1, "max_speakers": 12, "confidence_threshold": 0.45}, # Singapore English
        "sg-en": {"min_speakers": 1, "max_speakers": 12, "confidence_threshold": 0.45}, # Singapore English
        
        # Default configuration
        "default": {"min_speakers": 1, "max_speakers": 8, "confidence_threshold": 0.5}
    }
    
    return configs.get(lang_code, configs["default"])

def is_southeast_asian_language(language_code: str) -> bool:
    """
    Check if the language is supported by MERaLiON SEA-LION model.
    
    Args:
        language_code: ISO language code
        
    Returns:
        bool: True if language is supported by MERaLiON model
    """
    lang_code = language_code.lower().strip()
    return lang_code in LANGUAGE_MODEL_MAPPING or (
        len(lang_code) == 3 and any(
            len(code) == 2 and lang_code.startswith(code)
            for code in LANGUAGE_MODEL_MAPPING.keys()
        )
    )