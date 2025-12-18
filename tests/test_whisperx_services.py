"""Integration tests for WhisperX services.

Tests the complete STT pipeline with real Vietnamese audio files.
Requires server running on localhost:8000.
"""

import httpx
import os
import pytest
import time

# Base URL for integration tests
BASE_URL = "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=300.0)

# Vietnamese audio files
AUDIO_DIR = "datasets/audios/vn"
VN_AUDIO_1 = os.path.join(AUDIO_DIR, "vn-1.mp3")  # 2.8MB
VN_AUDIO_2 = os.path.join(AUDIO_DIR, "vn-2.mp3")  # 1.4MB

# Verify audio files exist
assert os.path.exists(VN_AUDIO_1), f"Audio file not found: {VN_AUDIO_1}"
assert os.path.exists(VN_AUDIO_2), f"Audio file not found: {VN_AUDIO_2}"


def wait_for_task_completion(workflow_id, max_wait=300, poll_interval=5):
    """Wait for a workflow to complete by polling its status.
    
    Args:
        workflow_id: Workflow identifier
        max_wait: Maximum wait time in seconds
        poll_interval: Time between polls in seconds
        
    Returns:
        Final workflow result when completed
        
    Raises:
        TimeoutError: If workflow doesn't complete within max_wait
        ValueError: If workflow fails
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check workflow status
        response = client.get(f"/temporal/workflow/{workflow_id}")
        if response.status_code == 404:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        assert response.status_code == 200, f"Failed to get workflow status: {response.text}"
        
        data = response.json()
        status = data.get("status")
        
        if status == "COMPLETED":
            # Get the result
            result_response = client.get(f"/temporal/workflow/{workflow_id}/result")
            assert result_response.status_code == 200, f"Failed to get workflow result: {result_response.text}"
            return result_response.json()
        elif status in ["FAILED", "TERMINATED", "TIMED_OUT", "CANCELED"]:
            raise ValueError(f"Workflow {status.lower()}")
        
        time.sleep(poll_interval)
    
    raise TimeoutError(f"Workflow {workflow_id} did not complete within {max_wait}s")


# ============================================================================
# Health Check Tests
# ============================================================================

def test_health_check():
    """Test basic health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_readiness_check():
    """Test readiness check."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "temporal" in data  # Changed from "database"


# ============================================================================
# Speech-to-Text Tests with Vietnamese Audio
# ============================================================================

def test_speech_to_text_vietnamese_small():
    """Test STT with smaller Vietnamese audio file."""
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        response = client.post(
            "/speech-to-text",
            files=files,
            params={
                "language": "vi",  # Vietnamese
                "device": "cpu",
                "compute_type": "int8"
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "identifier" in data
    assert "Workflow started" in data["message"]  # Changed from "Task queued"
    
    # Wait for completion
    workflow_id = data["identifier"]
    result = wait_for_task_completion(workflow_id, max_wait=180)
    
    # Verify result structure
    assert "segments" in result
    
    # Should have some transcribed text
    assert len(result["segments"]) > 0


def test_transcribe_vietnamese_large():
    """Test transcription with larger Vietnamese audio file."""
    with open(VN_AUDIO_1, "rb") as audio_file:
        files = {"file": ("vn-1.mp3", audio_file, "audio/mpeg")}
        response = client.post(
            "/speech-to-text",
            files=files,
            params={
                "language": "vi",
                "device": "cpu",
                "compute_type": "int8"
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "identifier" in data
    
    # Wait for completion
    workflow_id = data["identifier"]
    result = wait_for_task_completion(workflow_id, max_wait=240)
    
    # Verify transcription
    assert "segments" in result
    segments = result["segments"]
    assert len(segments) > 0
    
    # Verify segment structure
    first_segment = segments[0]
    assert "text" in first_segment
    assert "start" in first_segment
    assert "end" in first_segment


def test_full_pipeline_vietnamese():
    """Test complete STT pipeline with Vietnamese audio."""
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        
        # Full pipeline request
        response = client.post(
            "/speech-to-text",
            files=files,
            params={
                "language": "vi",
                "device": "cpu",
                "compute_type": "int8"
            }
        )
        
        assert response.status_code == 200
        workflow_id = response.json()["identifier"]
        
        # Wait for full pipeline
        result = wait_for_task_completion(workflow_id, max_wait=300)
        
        # Verify all pipeline stages completed
        assert "segments" in result
        
        # Check segments
        segments = result["segments"]
        if len(segments) > 0:
            first_segment = segments[0]
            assert "text" in first_segment
            assert "start" in first_segment
            assert "end" in first_segment


# ============================================================================
# Task Management Tests
# ============================================================================

def test_get_workflow_status():
    """Test getting workflow status."""
    # Submit a workflow first
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        response = client.post("/speech-to-text", files=files)
        
        assert response.status_code == 200
        workflow_id = response.json()["identifier"]
    
    # Get workflow status
    status_response = client.get(f"/temporal/workflow/{workflow_id}")
    assert status_response.status_code == 200
    
    data = status_response.json()
    assert "status" in data
    assert "workflow_id" in data
    assert data["workflow_id"] == workflow_id


def test_workflow_not_found():
    """Test getting non-existent workflow."""
    response = client.get("/temporal/workflow/non_existent_id")
    assert response.status_code == 404


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_missing_audio_file():
    """Test upload without audio file."""
    response = client.post("/speech-to-text")
    assert response.status_code == 422  # Validation error


def test_invalid_language():
    """Test with unsupported language code."""
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        response = client.post(
            "/speech-to-text",
            files=files,
            params={"language": "invalid_lang"}
        )
        
        # Should still accept but may auto-detect or fail gracefully
        assert response.status_code in [200, 422]


def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    identifiers = []
    
    # Submit multiple workflows
    for i in range(2):  # Reduced from 3 to 2 for faster testing
        with open(VN_AUDIO_2, "rb") as audio_file:
            files = {"file": (f"vn-2-{i}.mp3", audio_file, "audio/mpeg")}
            response = client.post("/speech-to-text", files=files, params={"language": "vi"})
            assert response.status_code == 200
            identifiers.append(response.json()["identifier"])
    
    # Wait for all to complete
    for workflow_id in identifiers:
        result = wait_for_task_completion(workflow_id, max_wait=300)
        assert "segments" in result
