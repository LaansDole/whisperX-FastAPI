"""
Integration test for model caching API endpoints.

Tests cache management endpoints (status and clear).

Note: These tests verify the API server's cache endpoints work correctly.
Model caching happens in the Temporal worker process (separate from API server),
so cache hit/miss metrics reflect the API server process, not the worker where
models are actually loaded. The unit tests verify the caching logic works correctly.

Prerequisites:
- Server must be running (make dev)
"""

import requests
import pytest
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_AUDIO_DIR = Path(__file__).parent.parent.parent / "datasets" / "audios" / "vn"


@pytest.mark.integration
class TestModelCaching:
    """Integration tests for model caching."""

    def setup_method(self):
        """Setup - clear caches before each test."""
        try:
            response = requests.post(f"{BASE_URL}/cache/clear", timeout=10)
            assert response.status_code == 200
            print(f"✓ Caches cleared: {response.json()}")
        except Exception as e:
            pytest.skip(f"Server not available: {e}")

    def test_cache_status_endpoint(self):
        """Test that cache status endpoint returns proper structure."""
        response = requests.get(f"{BASE_URL}/cache/status")
        assert response.status_code == 200

        data = response.json()

        # Verify structure
        assert "transcription" in data
        assert "alignment" in data
        assert "diarization" in data

        # Verify each cache has required fields
        for cache_name in ["transcription", "alignment", "diarization"]:
            cache_data = data[cache_name]
            assert "cached_models" in cache_data
            assert "cache_size" in cache_data
            assert "cache_hits" in cache_data
            assert "cache_misses" in cache_data
            assert "hit_rate_percent" in cache_data

        print("✓ Cache status structure validated")
        print(f"  Transcription: {data['transcription']['cache_size']} models cached")
        print(f"  Alignment: {data['alignment']['cache_size']} models cached")
        print(f"  Diarization: {data['diarization']['cache_size']} models cached")

    def test_cache_clear_endpoint(self):
        """Test that cache clear endpoint works."""
        response = requests.post(f"{BASE_URL}/cache/clear")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "All model caches cleared" in data["message"]
        assert "gpu_memory_freed_mb" in data

        # Verify caches are empty
        status_response = requests.get(f"{BASE_URL}/cache/status")
        status_data = status_response.json()

        assert status_data["transcription"]["cache_size"] == 0
        assert status_data["alignment"]["cache_size"] == 0
        assert status_data["diarization"]["cache_size"] == 0

        print("✓ All caches cleared successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
