"""Unit tests for MERaLiON services."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

# Import warnings filter first to prevent resource tracker warnings
from app.warnings_filter import filter_warnings
filter_warnings()

from app.meralion_services import (
    MERaLiONModelManager,
    transcribe_with_meralion,
    transcribe_with_fallback
)


class TestMERaLiONModelManager:
    """Test cases for MERaLiONModelManager class."""
    
    def test_detect_device_auto(self):
        """Test automatic device detection."""
        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            manager = MERaLiONModelManager(device="auto")
            assert manager.device == "cuda"
        
        # Test without CUDA
        with patch('torch.cuda.is_available', return_value=False):
            manager = MERaLiONModelManager(device="auto")
            assert manager.device == "cpu"
    
    def test_detect_device_explicit(self):
        """Test explicit device specification."""
        manager = MERaLiONModelManager(device="cpu")
        assert manager.device == "cpu"
        
        manager = MERaLiONModelManager(device="cuda")
        assert manager.device == "cuda"
    
    def test_validate_audio_duration(self):
        """Test audio validation for duration limits."""
        manager = MERaLiONModelManager()
        
        # Create audio longer than 30 seconds (assuming 16kHz sample rate)
        long_audio = np.random.randn(16000 * 35)  # 35 seconds
        
        # Should truncate to 30 seconds
        validated = manager._validate_audio(long_audio)
        expected_length = 16000 * 30  # 30 seconds
        assert len(validated) == expected_length
    
    def test_validate_audio_shape(self):
        """Test audio validation for shape normalization."""
        manager = MERaLiONModelManager()
        
        # Test with multi-dimensional audio
        multi_dim_audio = np.random.randn(16000, 2)
        validated = manager._validate_audio(multi_dim_audio)
        assert len(validated.shape) == 1  # Should be flattened
    
    @patch('app.meralion_services.AutoProcessor')
    @patch('app.meralion_services.AutoModelForSpeechSeq2Seq')
    def test_load_model_cpu_success(self, mock_model_class, mock_processor_class):
        """Test successful CPU model loading."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test CPU loading
        manager = MERaLiONModelManager(device="cpu")
        result = manager._load_model_cpu()
        
        assert result is True
        assert manager.model_loaded is True
        assert manager.processor == mock_processor
        assert manager.model == mock_model
        
        # Verify correct parameters were used
        mock_processor_class.from_pretrained.assert_called_once_with(
            "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
            trust_remote_code=True
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
            use_safetensors=True,
            trust_remote_code=True
        )
    
    @patch('app.meralion_services.AutoProcessor')
    @patch('app.meralion_services.AutoModelForSpeechSeq2Seq')
    def test_load_model_cpu_failure(self, mock_processor_class):
        """Test CPU model loading failure."""
        # Setup mocks to raise exception
        mock_processor_class.from_pretrained.side_effect = Exception("Test error")
        
        manager = MERaLiONModelManager(device="cpu")
        result = manager._load_model_cpu()
        
        assert result is False
        assert manager.model_loaded is False
        assert manager.last_error == "Test error"
    
    @patch('app.meralion_services.AutoProcessor')
    @patch('app.meralion_services.AutoModelForSpeechSeq2Seq')
    @patch('torch.cuda.is_available', return_value=True)
    def test_load_model_gpu_success(self, mock_model_class, mock_processor_class):
        """Test successful GPU model loading."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test GPU loading
        manager = MERaLiONModelManager(device="cuda")
        result = manager._load_model_gpu()
        
        assert result is True
        assert manager.model_loaded is True
        
        # Verify correct parameters were used
        mock_processor_class.from_pretrained.assert_called_once_with(
            "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
            trust_remote_code=True
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        mock_model.to.assert_called_once_with("cuda")
    
    @patch('app.meralion_services.get_audio_duration')
    def test_prepare_inputs_cpu(self, mock_get_duration):
        """Test CPU input preparation."""
        mock_get_duration.return_value = 10.0
        
        manager = MERaLiONModelManager(device="cpu")
        manager.processor = Mock()
        manager.processor.tokenizer.apply_chat_template.return_value = "test_prompt"
        manager.processor.return_value = {"input_ids": "test_ids"}
        
        audio = np.random.randn(16000)  # 1 second of audio
        query = "Please transcribe this speech."
        
        inputs = manager._prepare_inputs_cpu(audio, query)
        
        # Verify chat template was applied
        manager.processor.tokenizer.apply_chat_template.assert_called_once()
        
        # Verify processor was called with correct parameters
        call_args = manager.processor.call_args
        assert "text" in call_args.kwargs
        assert "audios" in call_args.kwargs
        assert len(call_args.kwargs["audios"]) == 2  # Duplicated as required
    
    @patch('app.meralion_services.get_audio_duration')
    def test_prepare_inputs_gpu(self, mock_get_duration):
        """Test GPU input preparation with tensor placement."""
        mock_get_duration.return_value = 10.0
        
        manager = MERaLiONModelManager(device="cuda")
        manager.processor = Mock()
        manager.processor.tokenizer.apply_chat_template.return_value = "test_prompt"
        
        # Mock tensors
        mock_tensor_float32 = Mock()
        mock_tensor_float32.dtype = torch.float32
        mock_tensor_float32.to.return_value = Mock()
        
        mock_tensor_bfloat16 = Mock()
        mock_tensor_bfloat16.dtype = torch.bfloat16
        mock_tensor_bfloat16.to.return_value = mock_tensor_bfloat16
        
        manager.processor.return_value = {
            "input_ids": mock_tensor_float32,
            "attention_mask": mock_tensor_bfloat16
        }
        
        audio = np.random.randn(16000)  # 1 second of audio
        query = "Please transcribe this speech."
        
        inputs = manager._prepare_inputs_gpu(audio, query)
        
        # Verify tensors were moved to GPU
        mock_tensor_float32.to.assert_called_with("cuda")
        mock_tensor_bfloat16.to.assert_called_with("cuda")
        
        # Verify float32 tensor was converted to bfloat16
        mock_tensor_float32.to.return_value.to.assert_called_with(torch.bfloat16)
    
    @patch('app.meralion_services.gc.collect')
    @patch('app.meralion_services.torch.cuda.empty_cache')
    @patch('app.meralion_services.torch.cuda.is_available', return_value=True)
    def test_cleanup_model(self, mock_cuda_available, mock_empty_cache, mock_gc_collect):
        """Test model cleanup."""
        manager = MERaLiONModelManager()
        manager.model = Mock()
        manager.processor = Mock()
        manager.model_loaded = True
        
        manager._cleanup_model()
        
        assert manager.model is None
        assert manager.processor is None
        assert manager.model_loaded is False
        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()


class TestTranscriptionFunctions:
    """Test cases for transcription functions."""
    
    @patch('app.meralion_services.MERaLiONModelManager')
    def test_transcribe_with_meralion_success(self, mock_manager_class):
        """Test successful MERaLiON transcription."""
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.transcribe.return_value = {
            "text": "Test transcription",
            "model_used": "MERaLiON-AudioLLM-Whisper-SEA-LION",
            "success": True
        }
        mock_manager_class.return_value = mock_manager
        
        audio = np.random.randn(16000)
        result = transcribe_with_meralion(audio, task="transcribe")
        
        assert result["success"] is True
        assert result["text"] == "Test transcription"
        assert result["model_used"] == "MERaLiON-AudioLLM-Whisper-SEA-LION"
    
    @patch('app.meralion_services.MERaLiONModelManager')
    def test_transcribe_with_meralion_failure(self, mock_manager_class):
        """Test MERaLiON transcription failure."""
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.transcribe.side_effect = Exception("Test error")
        mock_manager_class.return_value = mock_manager
        
        audio = np.random.randn(16000)
        result = transcribe_with_meralion(audio, task="transcribe")
        
        assert result["success"] is False
        assert result["text"] == ""
        assert "error" in result
    
    @patch('app.meralion_services.transcribe_with_meralion')
    @patch('app.meralion_services.transcribe_with_whisper')
    def test_transcribe_with_fallback_meralion_success(self, mock_whisper, mock_meralion):
        """Test fallback when MERaLiON succeeds."""
        # Setup mocks
        mock_meralion.return_value = {
            "text": "MERaLiON transcription",
            "success": True,
            "model_used": "MERaLiON-AudioLLM-Whisper-SEA-LION"
        }
        
        audio = np.random.randn(16000)
        result = transcribe_with_fallback(audio, task="transcribe")
        
        assert result["success"] is True
        assert result["text"] == "MERaLiON transcription"
        assert result["model_used"] == "MERaLiON-AudioLLM-Whisper-SEA-LION"
        
        # Verify MERaLiON was called
        mock_meralion.assert_called_once()
        
        # Verify whisper was not called
        mock_whisper.assert_not_called()
    
    @patch('app.meralion_services.transcribe_with_meralion')
    @patch('app.meralion_services.transcribe_with_whisper')
    def test_transcribe_with_fallback_whisper_success(self, mock_whisper, mock_meralion):
        """Test fallback when MERaLiON fails but Whisper succeeds."""
        # Setup mocks
        mock_meralion.return_value = {
            "text": "",
            "success": False,
            "error": "MERaLiON failed"
        }
        
        mock_whisper_result = Mock()
        mock_whisper_result.segments = [Mock(text="Whisper transcription")]
        mock_whisper.return_value = mock_whisper_result
        
        audio = np.random.randn(16000)
        result = transcribe_with_fallback(audio, task="transcribe")
        
        assert result["success"] is True
        assert result["text"] == "Whisper transcription"
        assert result["fallback_used"] is True
        
        # Verify both were called
        mock_meralion.assert_called_once()
        mock_whisper.assert_called_once()
    
    @patch('app.meralion_services.transcribe_with_meralion')
    @patch('app.meralion_services.transcribe_with_whisper')
    def test_transcribe_with_fallback_all_fail(self, mock_whisper, mock_meralion):
        """Test fallback when all models fail."""
        # Setup mocks
        mock_meralion.return_value = {
            "text": "",
            "success": False,
            "error": "MERaLiON failed"
        }
        
        mock_whisper.side_effect = Exception("Whisper failed")
        
        audio = np.random.randn(16000)
        result = transcribe_with_fallback(audio, task="transcribe")
        
        assert result["success"] is False
        assert result["text"] == ""
        assert "All models failed" in result["error"]


class TestIntegration:
    """Integration tests for MERaLiON services."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_compatibility(self):
        """Test that GPU inference components are compatible."""
        manager = MERaLiONModelManager(device="cuda")
        assert manager.device == "cuda"
        
        # Test tensor operations
        test_tensor = torch.randn(10, dtype=torch.float32)
        if torch.cuda.is_available():
            test_tensor = test_tensor.to("cuda")
            assert test_tensor.device.type == "cuda"
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing pipeline."""
        manager = MERaLiONModelManager()
        
        # Test normal audio
        normal_audio = np.random.randn(16000)  # 1 second
        validated = manager._validate_audio(normal_audio)
        assert len(validated) == len(normal_audio)
        
        # Test long audio
        long_audio = np.random.randn(16000 * 60)  # 60 seconds
        validated = manager._validate_audio(long_audio)
        assert len(validated) == 16000 * 30  # Truncated to 30 seconds
        
        # Test multi-dimensional audio
        multi_audio = np.random.randn(16000, 2)
        validated = manager._validate_audio(multi_audio)
        assert len(validated.shape) == 1  # Flattened


if __name__ == "__main__":
    pytest.main([__file__])