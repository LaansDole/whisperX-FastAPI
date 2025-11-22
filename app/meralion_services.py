"""This module provides services for transcribing audio using MERaLiON-AudioLLM-Whisper-SEA-LION model."""

import gc
import logging
import atexit
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .config import Config
from .logger import logger
from .audio import get_audio_duration

# MERaLiON model configuration from Config
MERALION_REPO_ID = getattr(Config, 'MERALION_REPO_ID', "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION")
MERALION_MAX_AUDIO_LENGTH = getattr(Config, 'MERALION_MAX_AUDIO_LENGTH', 30)  # seconds
MERALION_SAMPLE_RATE = getattr(Config, 'MERALION_SAMPLE_RATE', 16000)  # Hz

# Prompt templates for MERaLiON
MERALION_PROMPT_TEMPLATE = "Given the following audio context: <SpeechHere>\n\nText instruction: {query}"
MERALION_TRANSCRIBE_QUERY = "Please transcribe this speech."
MERALION_TRANSLATE_QUERY = "Can you please translate this speech into written Chinese?"

# Fallback model hierarchy from Config
FALLBACK_MODELS = getattr(Config, 'MERALION_FALLBACK_MODELS', [
    "whisper-large-v3",
    "faster-whisper",
    "whisperX"
])

# Global cleanup function to prevent semaphore leaks
def cleanup_multiprocessing_resources():
    """Clean up multiprocessing resources to prevent semaphore leaks at shutdown."""
    try:
        # Clean up any active child processes
        if hasattr(multiprocessing, 'active_children'):
            for child in multiprocessing.active_children():
                try:
                    child.terminate()
                    child.join(timeout=1.0)
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.debug("Multiprocessing resources cleaned up at shutdown")
    except Exception as e:
        logger.debug(f"Error during multiprocessing cleanup: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_multiprocessing_resources)


class MERaLiONModelManager:
    """Manager for MERaLiON model with fallback capabilities."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the MERaLiON model manager.
        
        Args:
            device: Device to use for inference ('auto', 'cuda', 'cpu')
        """
        self.device = self._detect_device(device)
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.last_error = None
        
    def _detect_device(self, device: str) -> str:
        """
        Detect the appropriate device for inference.
        
        Args:
            device: Device preference ('auto', 'cuda', 'cpu')
            
        Returns:
            str: The detected device ('cuda' or 'cpu')
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess audio for MERaLiON model.
        
        Args:
            audio: Input audio array
            
        Returns:
            np.ndarray: Validated and preprocessed audio array
            
        Raises:
            ValueError: If audio is too long or has wrong sample rate
        """
        # Check audio duration
        duration = get_audio_duration(audio)
        if duration > MERALION_MAX_AUDIO_LENGTH:
            logger.warning(
                f"Audio duration {duration:.2f}s exceeds MERaLiON limit of {MERALION_MAX_AUDIO_LENGTH}s. "
                f"Truncating to {MERALION_MAX_AUDIO_LENGTH}s."
            )
            # Truncate audio to maximum allowed length
            max_samples = int(MERALION_MAX_AUDIO_LENGTH * MERALION_SAMPLE_RATE)
            audio = audio[:max_samples]
        
        # Ensure audio is at 16kHz
        if hasattr(audio, 'shape') and len(audio.shape) > 1:
            audio = audio.flatten()
        
        return audio
    
    def _load_model_cpu(self) -> bool:
        """
        Load MERaLiON model for CPU inference.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading MERaLiON model for CPU inference from {MERALION_REPO_ID}")
            
            self.processor = AutoProcessor.from_pretrained(
                MERALION_REPO_ID,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MERALION_REPO_ID,
                use_safetensors=True,
                trust_remote_code=True,
            )
            
            self.model_loaded = True
            logger.info("MERaLiON model loaded successfully for CPU inference")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to load MERaLiON model for CPU: {e}")
            return False
    
    def _load_model_gpu(self) -> bool:
        """
        Load MERaLiON model for GPU inference with optimizations.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading MERaLiON model for GPU inference from {MERALION_REPO_ID}")
            
            self.processor = AutoProcessor.from_pretrained(
                MERALION_REPO_ID,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MERALION_REPO_ID,
                use_safetensors=True,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            self.model_loaded = True
            logger.info("MERaLiON model loaded successfully for GPU inference")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to load MERaLiON model for GPU: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load MERaLiON model based on detected device.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            logger.debug("MERaLiON model already loaded")
            return True
        
        if self.device == "cuda":
            return self._load_model_gpu()
        else:
            return self._load_model_cpu()
    
    def _prepare_inputs_cpu(self, audio: np.ndarray, query: str) -> Dict[str, Any]:
        """
        Prepare inputs for CPU inference.
        
        Args:
            audio: Preprocessed audio array
            query: Query string (transcribe or translate)
            
        Returns:
            Dict: Prepared inputs for the model
        """
        # Create conversation structure
        conversation = [[{"role": "user", "content": MERALION_PROMPT_TEMPLATE.format(query=query)}]]
        
        # Apply chat template
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare audio array (duplicate as required by MERaLiON)
        audio_array = [audio] * 2
        
        # Process inputs
        inputs = self.processor(text=chat_prompt, audios=audio_array)
        
        return inputs
    
    def _prepare_inputs_gpu(self, audio: np.ndarray, query: str) -> Dict[str, Any]:
        """
        Prepare inputs for GPU inference with proper tensor placement.
        
        Args:
            audio: Preprocessed audio array
            query: Query string (transcribe or translate)
            
        Returns:
            Dict: Prepared inputs for the model
        """
        # Create conversation structure
        conversation = [[{"role": "user", "content": MERALION_PROMPT_TEMPLATE.format(query=query)}]]
        
        # Apply chat template
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare audio array (duplicate as required by MERaLiON)
        audio_array = [audio] * 2
        
        # Process inputs
        inputs = self.processor(text=chat_prompt, audios=audio_array)
        
        # Move tensors to GPU and convert dtype
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
                if value.dtype == torch.float32:
                    inputs[key] = inputs[key].to(torch.bfloat16)
        
        return inputs
    
    def _cleanup_model(self):
        """Clean up model resources to prevent memory leaks."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.model_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up any multiprocessing resources to prevent semaphore leaks
        try:
            # Close any open multiprocessing connections
            if hasattr(multiprocessing, 'active_children'):
                for child in multiprocessing.active_children():
                    try:
                        child.terminate()
                        child.join(timeout=1.0)
                    except:
                        pass
        except Exception as e:
            logger.debug(f"Error during multiprocessing cleanup: {e}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        task: str = "transcribe",
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Transcribe audio using MERaLiON model.
        
        Args:
            audio: Input audio array
            task: Task type ('transcribe' or 'translate')
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict: Transcription result with metadata
            
        Raises:
            RuntimeError: If model fails to load or inference fails
        """
        # Validate and preprocess audio
        audio = self._validate_audio(audio)
        
        # Load model if not already loaded
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load MERaLiON model: {self.last_error}")
        
        try:
            # Select appropriate query based on task
            query = MERALION_TRANSCRIBE_QUERY if task == "transcribe" else MERALION_TRANSLATE_QUERY
            
            # Prepare inputs based on device
            if self.device == "cuda":
                inputs = self._prepare_inputs_gpu(audio, query)
            else:
                inputs = self._prepare_inputs_cpu(audio, query)
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # Process outputs
            generated_ids = outputs[:, inputs['input_ids'].size(1):]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Extract text from response
            if isinstance(response, list) and len(response) > 0:
                transcription_text = response[0].strip()
            else:
                transcription_text = str(response).strip()
            
            # Return result with metadata
            return {
                "text": transcription_text,
                "model_used": "MERaLiON-AudioLLM-Whisper-SEA-LION",
                "device": self.device,
                "task": task,
                "audio_duration": get_audio_duration(audio),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error during MERaLiON inference: {e}")
            raise RuntimeError(f"MERaLiON inference failed: {str(e)}")
        
        finally:
            # Clean up model to prevent memory leaks
            self._cleanup_model()


def transcribe_with_meralion(
    audio: np.ndarray,
    task: str = "transcribe",
    device: str = "auto",
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    Transcribe audio using MERaLiON model with automatic fallback.
    
    Args:
        audio: Input audio array
        task: Task type ('transcribe' or 'translate')
        device: Device preference ('auto', 'cuda', 'cpu')
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict: Transcription result with metadata
    """
    manager = MERaLiONModelManager(device=device)
    
    try:
        result = manager.transcribe(audio, task, max_new_tokens)
        logger.info(f"MERaLiON transcription successful: {result.get('text', '')[:100]}...")
        return result
    except Exception as e:
        logger.warning(f"MERaLiON transcription failed: {e}")
        return {
            "text": "",
            "model_used": "MERaLiON-AudioLLM-Whisper-SEA-LION",
            "device": device,
            "task": task,
            "success": False,
            "error": str(e)
        }


def transcribe_with_fallback(
    audio: np.ndarray,
    task: str = "transcribe",
    device: str = "auto",
    max_new_tokens: int = 256,
    fallback_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Transcribe audio using MERaLiON with fallback to other models.
    
    Args:
        audio: Input audio array
        task: Task type ('transcribe' or 'translate')
        device: Device preference ('auto', 'cuda', 'cpu')
        max_new_tokens: Maximum number of tokens to generate
        fallback_models: List of fallback model names
        
    Returns:
        Dict: Transcription result with metadata
    """
    if fallback_models is None:
        fallback_models = FALLBACK_MODELS
    
    # Try MERaLiON first
    result = transcribe_with_meralion(audio, task, device, max_new_tokens)
    
    if result.get("success", False):
        return result
    
    logger.info(f"MERaLiON failed, trying fallback models: {fallback_models}")
    
    # Import here to avoid circular imports
    from .whisperx_services import transcribe_with_whisper
    
    # Try fallback models
    for model_name in fallback_models:
        try:
            logger.info(f"Trying fallback model: {model_name}")
            
            # Use existing whisperx services for fallback
            whisper_result = transcribe_with_whisper(
                audio=audio,
                task=task,
                asr_options=None,
                vad_options=None,
                language="auto",  # Auto-detect language
                model=model_name,
                device=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
                compute_type="int8" if device == "cpu" else "float16"
            )
            
            # Extract text from whisper result
            if hasattr(whisper_result, 'segments') and whisper_result.segments:
                text = " ".join([segment.text for segment in whisper_result.segments])
            elif isinstance(whisper_result, dict) and "segments" in whisper_result:
                text = " ".join([segment.get("text", "") for segment in whisper_result["segments"]])
            else:
                text = str(whisper_result)
            
            logger.info(f"Fallback model {model_name} succeeded")
            
            return {
                "text": text,
                "model_used": model_name,
                "device": device,
                "task": task,
                "success": True,
                "fallback_used": True,
                "original_error": result.get("error", "Unknown MERaLiON error")
            }
            
        except Exception as e:
            logger.warning(f"Fallback model {model_name} failed: {e}")
            continue
    
    # All models failed
    return {
        "text": "",
        "model_used": "none",
        "device": device,
        "task": task,
        "success": False,
        "error": f"All models failed. MERaLiON error: {result.get('error', 'Unknown')}"
    }