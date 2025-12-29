"""
Unit tests for model caching functionality in whisperx_services.

This test module verifies thread-safe caching behavior for transcription,
alignment, and diarization models, including cache hit/miss tracking and
memory pressure detection.
"""

import threading
from unittest.mock import MagicMock, patch
import pytest

# Import the caching functions
from app.whisperx_services import (
    get_transcription_model,
    get_alignment_model,
    get_diarization_model,
    clear_transcription_model_cache,
    clear_all_model_caches,
    get_transcription_cache_status,
    get_all_cache_status,
    check_gpu_memory_pressure,
    _transcription_model_cache,
    _alignment_model_cache,
    _diarization_model_cache,
)


@pytest.fixture(autouse=True)
def clear_caches_before_test():
    """Clear all model caches before each test."""
    clear_all_model_caches()
    yield
    clear_all_model_caches()


class TestTranscriptionModelCache:
    """Test transcription model caching."""

    @patch("app.whisperx_services.load_model")
    def test_model_cached_on_first_call(self, mock_load_model):
        """Test that model is loaded and cached on first call."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # First call should load the model
        result = get_transcription_model(
            model_name="base",
            device_param="cpu",
            compute_type_param="float32",
        )

        assert result == mock_model
        mock_load_model.assert_called_once()

    @patch("app.whisperx_services.load_model")
    def test_model_reused_on_subsequent_calls(self, mock_load_model):
        """Test that cached model is reused without reloading."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # First call
        get_transcription_model(
            model_name="base",
            device_param="cpu",
            compute_type_param="float32",
        )

        # Second call should use cache
        result = get_transcription_model(
            model_name="base",
            device_param="cpu",
            compute_type_param="float32",
        )

        assert result == mock_model
        # Should only be called once (first call)
        assert mock_load_model.call_count == 1

    @patch("app.whisperx_services.load_model")
    def test_cache_hit_tracking(self, mock_load_model):
        """Test that cache hits and misses are tracked correctly."""
        clear_all_model_caches()  # Reset counters

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # First call - should be a cache miss
        get_transcription_model(
            model_name="base",
            device_param="cpu",
            compute_type_param="float32",
        )

        # Second call - should be a cache hit
        get_transcription_model(
            model_name="base",
            device_param="cpu",
            compute_type_param="float32",
        )

        status = get_transcription_cache_status()
        assert status["cache_hits"] >= 1
        assert status["cache_misses"] >= 1
        assert status["cache_size"] == 1

    def test_cache_clear_removes_models(self):
        """Test that clearing cache removes cached models."""
        with patch("app.whisperx_services.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model

            # Load a model
            get_transcription_model(
                model_name="base",
                device_param="cpu",
                compute_type_param="float32",
            )

            # Verify it's cached
            assert len(_transcription_model_cache) > 0

            # Clear cache
            clear_transcription_model_cache()

            # Verify cache is empty
            assert len(_transcription_model_cache) == 0


class TestAlignmentModelCache:
    """Test alignment model caching."""

    @patch("app.whisperx_services.load_align_model")
    def test_alignment_model_cached(self, mock_load_align_model):
        """Test that alignment model is cached correctly."""
        mock_model = MagicMock()
        mock_metadata = {"key": "value"}
        mock_load_align_model.return_value = (mock_model, mock_metadata)

        # First call
        result_model, result_metadata = get_alignment_model(
            language_code="en",
            device_param="cpu",
        )

        assert result_model == mock_model
        assert result_metadata == mock_metadata
        mock_load_align_model.assert_called_once()

    @patch("app.whisperx_services.load_align_model")
    def test_alignment_model_reused(self, mock_load_align_model):
        """Test that alignment model is reused from cache."""
        mock_model = MagicMock()
        mock_metadata = {"key": "value"}
        mock_load_align_model.return_value = (mock_model, mock_metadata)

        # First call
        get_alignment_model(language_code="en", device_param="cpu")

        # Second call
        get_alignment_model(language_code="en", device_param="cpu")

        # Should only load once
        assert mock_load_align_model.call_count == 1


class TestDiarizationModelCache:
    """Test diarization model caching."""

    @patch("app.whisperx_services.DiarizationPipeline")
    def test_diarization_model_cached(self, mock_diarization_pipeline):
        """Test that diarization model is cached correctly."""
        mock_model = MagicMock()
        mock_diarization_pipeline.return_value = mock_model

        # First call
        result = get_diarization_model(device_param="cpu")

        assert result == mock_model
        mock_diarization_pipeline.assert_called_once()

    @patch("app.whisperx_services.DiarizationPipeline")
    def test_diarization_model_reused(self, mock_diarization_pipeline):
        """Test that diarization model is reused from cache."""
        mock_model = MagicMock()
        mock_diarization_pipeline.return_value = mock_model

        # First call
        get_diarization_model(device_param="cpu")

        # Second call
        get_diarization_model(device_param="cpu")

        # Should only load once
        assert mock_diarization_pipeline.call_count == 1


class TestThreadSafety:
    """Test thread safety of model caching."""

    @patch("app.whisperx_services.load_model")
    def test_concurrent_cache_access(self, mock_load_model):
        """Test that concurrent access to cache is thread-safe."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        results = []
        errors = []

        def load_model_thread():
            try:
                result = get_transcription_model(
                    model_name="base",
                    device_param="cpu",
                    compute_type_param="float32",
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_model_thread) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all threads got the same model
        assert all(r == mock_model for r in results)

        # Verify model was only loaded once
        assert mock_load_model.call_count == 1


class TestCacheManagement:
    """Test global cache management functions."""

    @patch("app.whisperx_services.load_model")
    @patch("app.whisperx_services.load_align_model")
    @patch("app.whisperx_services.DiarizationPipeline")
    def test_clear_all_caches(self, mock_diarization, mock_align, mock_transcription):
        """Test that clear_all_model_caches clears all caches."""
        # Setup mocks
        mock_transcription.return_value = MagicMock()
        mock_align.return_value = (MagicMock(), {})
        mock_diarization.return_value = MagicMock()

        # Load models into all caches
        get_transcription_model("base", "cpu", compute_type_param="float32")
        get_alignment_model("en", "cpu")
        get_diarization_model("cpu")

        # Verify all caches have models
        assert len(_transcription_model_cache) > 0
        assert len(_alignment_model_cache) > 0
        assert len(_diarization_model_cache) > 0

        # Clear all caches
        clear_all_model_caches()

        # Verify all caches are empty
        assert len(_transcription_model_cache) == 0
        assert len(_alignment_model_cache) == 0
        assert len(_diarization_model_cache) == 0

    @patch("app.whisperx_services.load_model")
    def test_get_all_cache_status(self, mock_load_model):
        """Test that get_all_cache_status returns combined status."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Load a model
        get_transcription_model("base", "cpu", compute_type_param="float32")

        # Get status
        status = get_all_cache_status()

        # Verify structure
        assert "transcription" in status
        assert "alignment" in status
        assert "diarization" in status

        # Verify transcription cache shows the loaded model
        assert status["transcription"]["cache_size"] > 0


class TestMemoryPressureDetection:
    """Test GPU memory pressure detection."""

    @patch("app.whisperx_services.torch.cuda.is_available")
    def test_no_pressure_detection_without_cuda(self, mock_cuda_available):
        """Test that pressure detection returns False when CUDA is not available."""
        mock_cuda_available.return_value = False

        result = check_gpu_memory_pressure()

        assert result is False

    @patch("app.whisperx_services.torch.cuda.is_available")
    @patch("app.whisperx_services.torch.cuda.memory_allocated")
    @patch("app.whisperx_services.torch.cuda.get_device_properties")
    def test_pressure_detected_above_threshold(self, mock_device_props, mock_memory_allocated, mock_cuda_available):
        """Test that pressure is detected when usage exceeds threshold."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 9000 * 1024**2  # 9GB

        mock_props = MagicMock()
        mock_props.total_memory = 10000 * 1024**2  # 10GB total
        mock_device_props.return_value = mock_props

        # 90% usage with 85% threshold should detect pressure
        result = check_gpu_memory_pressure(threshold=0.85)

        assert result is True

    @patch("app.whisperx_services.torch.cuda.is_available")
    @patch("app.whisperx_services.torch.cuda.memory_allocated")
    @patch("app.whisperx_services.torch.cuda.get_device_properties")
    def test_no_pressure_below_threshold(self, mock_device_props, mock_memory_allocated, mock_cuda_available):
        """Test that no pressure is detected when usage is below threshold."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 5000 * 1024**2  # 5GB

        mock_props = MagicMock()
        mock_props.total_memory = 10000 * 1024**2  # 10GB total
        mock_device_props.return_value = mock_props

        # 50% usage with 85% threshold should NOT detect pressure
        result = check_gpu_memory_pressure(threshold=0.85)

        assert result is False
