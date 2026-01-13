"""Integration tests for WhisperX services.

Tests the complete STT pipeline with real Vietnamese audio files.
Requires server running on localhost:8000.
"""

import httpx
import os
import pytest
import time
import sys
from pathlib import Path

# Add tests directory to path for conftest import
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import wait_for_workflow_completion


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


def wait_for_task_completion(workflow_id, max_wait=300, poll_interval=30):
    """Wait for a workflow to complete. Wrapper around shared helper."""
    return wait_for_workflow_completion(client, workflow_id, max_wait=max_wait, poll_interval=poll_interval)


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
        data = {"patient_name": "Test Patient"}  # Required for HIPAA tracking
        response = client.post(
            "/speech-to-text",
            files=files,
            data=data,
            params={
                "language": "vi",  # Vietnamese
                "device": "cpu",
                "compute_type": "int8",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "identifier" in data
    assert "Workflow started" in data["message"]  # Changed from "Task queued"

    workflow_id = data["identifier"]

    # Validate Two-Phase Commit
    print(f"\nValidating two-phase commit for workflow {workflow_id}...")

    # Small delay to ensure commit completed
    time.sleep(1)

    # Check database record
    admin_response = client.get(f"/admin/workflow/{workflow_id}/patient")

    assert admin_response.status_code == 200, (
        f"Database record not found! Two-phase commit may have failed. Status: {admin_response.status_code}"
    )

    db_record = admin_response.json()
    assert db_record.get("status") == "active", f"Expected status='active', got '{db_record.get('status')}'"
    print("[OK] Two-phase commit: Database record marked ACTIVE")
    print(f"  Patient: {db_record['patient_name']}")
    print(f"  Hash: {db_record['patient_hash']}")

    # Wait for completion
    result = wait_for_task_completion(workflow_id, max_wait=180)

    # Verify result structure
    assert "segments" in result

    # Should have some transcribed text
    assert len(result["segments"]) > 0


@pytest.mark.slow
def test_transcribe_vietnamese_large():
    """Test transcription with larger Vietnamese audio file."""
    with open(VN_AUDIO_1, "rb") as audio_file:
        files = {"file": ("vn-1.mp3", audio_file, "audio/mpeg")}
        data = {"patient_name": "Test Patient Large"}
        response = client.post(
            "/speech-to-text",
            files=files,
            data=data,
            params={"language": "vi", "device": "cpu", "compute_type": "int8"},
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


@pytest.mark.slow
def test_full_pipeline_vietnamese():
    """Test complete STT pipeline with Vietnamese audio."""
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        data = {"patient_name": "Test Full Pipeline"}

        # Full pipeline request
        response = client.post(
            "/speech-to-text",
            files=files,
            data=data,
            params={"language": "vi", "device": "cpu", "compute_type": "int8"},
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
        data = {"patient_name": "Test Status Check"}
        response = client.post("/speech-to-text", files=files, data=data)

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


@pytest.mark.integration
def test_invalid_language():
    """Test with unsupported language code."""
    with open(VN_AUDIO_2, "rb") as audio_file:
        files = {"file": ("vn-2.mp3", audio_file, "audio/mpeg")}
        data = {"patient_name": "Test Invalid Lang"}
        response = client.post("/speech-to-text", files=files, data=data, params={"language": "invalid_lang"})

        # Should still accept but may auto-detect or fail gracefully
        assert response.status_code in [200, 422]


@pytest.mark.slow
def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    identifiers = []

    # Submit multiple workflows
    for i in range(2):  # Reduced from 3 to 2 for faster testing
        with open(VN_AUDIO_2, "rb") as audio_file:
            files = {"file": (f"vn-2-{i}.mp3", audio_file, "audio/mpeg")}
            data = {"patient_name": f"Test Concurrent {i}"}
            response = client.post("/speech-to-text", files=files, data=data, params={"language": "vi"})
            assert response.status_code == 200
            identifiers.append(response.json()["identifier"])

    # Wait for all to complete
    for workflow_id in identifiers:
        result = wait_for_task_completion(workflow_id, max_wait=300)
        assert "segments" in result
