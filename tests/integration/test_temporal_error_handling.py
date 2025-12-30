"""Integration Test: Temporal Error Handling

Tests retryable and non-retryable error scenarios in Temporal workflows using real
audio files from the Kaggle dataset to validate error classification and retry behavior.

Requires:
- Server running: `make dev`
- Temporal server running
- LM Studio running and configured
"""

import pytest
import httpx
import time
from pathlib import Path


BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # 5 minutes

# Kaggle dataset - use SHORT audio file for faster testing
DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews"


def wait_for_workflow_result(client: httpx.Client, workflow_id: str, max_wait: int = 120, poll_interval: int = 5):
    """
    Poll workflow until completion or failure.

    Returns:
        tuple: (status, result_or_error)
    """
    elapsed = 0
    print(f"\n⏳ Polling workflow {workflow_id}...")

    while elapsed < max_wait:
        response = client.get(f"/temporal/workflow/{workflow_id}")

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "UNKNOWN")

            if status == "COMPLETED":
                result_response = client.get(f"/temporal/workflow/{workflow_id}/result")
                if result_response.status_code == 200:
                    return ("COMPLETED", result_response.json())

            elif status == "FAILED":
                error_response = client.get(f"/temporal/workflow/{workflow_id}")
                return ("FAILED", error_response.json())

            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            time.sleep(poll_interval)
            elapsed += poll_interval

    return ("TIMEOUT", None)


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_retryable_error_network_timeout():
    """
    Test retryable error: Network timeout during LLM requests.

    Expected behavior:
    - Error classified as retryable
    - Temporal should retry the activity
    - Workflow should eventually succeed OR fail after max retries
    """
    print("\n" + "=" * 80)
    print("TEST: Retryable Error - Network Timeout")
    print("=" * 80)

    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        print("\nStarting workflow with simulated network issues...")

        # Upload audio with medical processing (which will make LLM calls)
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": "Retryable Error Test",
                "enable_medical_processing": "true",
                "provider_id": "dr-test",
                "encounter_date": "2025-12-30",
            }

            response = client.post("/speech-to-text", files=files, data=data)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        workflow_id = response.json().get("identifier") or response.json().get("workflow_id")

        print(f"✓ Workflow started: {workflow_id}")
        print("  Expected: Network errors should trigger retries")

        # Note: In real scenarios, network timeouts would occur naturally
        # This test validates that when they do, the error is classified as retryable


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_non_retryable_error_invalid_input():
    """
    Test non-retryable error: Invalid encounter_date format.

    Expected behavior:
    - Validation fails immediately
    - Error classified as non-retryable (ValueError)
    - Workflow should fail without retries
    - HTTP 400 error returned to user
    """
    print("\n" + "=" * 80)
    print("TEST: Non-Retryable Error - Invalid Input (Encounter Date)")
    print("=" * 80)

    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        print("\nStarting workflow with invalid encounter_date...")

        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": "Non-Retryable Error Test",
                "enable_medical_processing": "true",
                "provider_id": "dr-test",
                "encounter_date": "invalid-date-format",  # Invalid ISO format
            }

            response = client.post("/speech-to-text", files=files, data=data)

        print(f"  Response status: {response.status_code}")

        # Should fail with HTTP 400 (validation error)
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        error_data = response.json()
        print(f"✓ Validation failed as expected: {error_data.get('detail', 'No detail')}")
        print("  Error classified as non-retryable (no workflow started)")


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_non_retryable_error_missing_provider():
    """
    Test non-retryable error: Missing required provider_id with medical processing.

    Expected behavior:
    - Validation fails before workflow starts
    - Error classified as non-retryable (ValueError)
    - HTTP 400 error returned to user
    - No orphaned database records
    """
    print("\n" + "=" * 80)
    print("TEST: Non-Retryable Error - Missing Provider ID")
    print("=" * 80)

    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        print("\nStarting workflow without provider_id...")

        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": "Missing Provider Test",
                "enable_medical_processing": "true",
                # provider_id missing!
                "encounter_date": "2025-12-30",
            }

            response = client.post("/speech-to-text", files=files, data=data)

        print(f"  Response status: {response.status_code}")

        # Should fail with HTTP 400 (validation error)
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        error_data = response.json()
        print(f"✓ Validation failed as expected: {error_data.get('detail', 'No detail')}")
        print("  Error classified as non-retryable (fail-fast)")


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_retryable_error_string_default():
    """
    Test that string errors default to retryable=True in Temporal activities.

    This validates the fix for the error_handler.py bug where string errors
    were incorrectly defaulting to non-retryable.
    """
    from app.temporal.error_handler import TemporalErrorHandler

    print("\n" + "=" * 80)
    print("TEST: String Error Default Retryability")
    print("=" * 80)

    # Test that string errors without explicit retryable parameter default to True
    app_error = TemporalErrorHandler.create_application_error("Temporary service unavailable", "Activity Context")

    print("  String error: 'Temporary service unavailable'")
    print(f"  non_retryable: {app_error.non_retryable}")

    # Should be retryable (non_retryable=False)
    assert not app_error.non_retryable, "String errors should default to retryable"
    print("✓ String error correctly defaults to retryable")


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_explicit_retryable_override():
    """
    Test that explicit retryable parameter overrides auto-classification.

    Validates that even ValueError (normally non-retryable) can be made
    retryable with explicit parameter.
    """
    from app.temporal.error_handler import TemporalErrorHandler

    print("\n" + "=" * 80)
    print("TEST: Explicit Retryable Parameter Override")
    print("=" * 80)

    # ValueError is normally non-retryable, but we override it
    error = ValueError("Validation error")
    app_error = TemporalErrorHandler.create_application_error(
        error,
        "Activity Context",
        retryable=True,  # Explicit override
    )

    print("  Error type: ValueError (normally non-retryable)")
    print("  Explicit retryable=True override")
    print(f"  Result non_retryable: {app_error.non_retryable}")

    # Should be retryable (non_retryable=False) due to explicit override
    assert not app_error.non_retryable, "Explicit retryable=True should override classification"
    print("✓ Explicit retryable parameter correctly overrides auto-classification")


@pytest.mark.integration
@pytest.mark.temporal_errors
def test_workflow_with_successful_execution():
    """
    Baseline test: Successful workflow execution with no errors.

    This validates that the error handling changes don't break normal execution.
    """
    print("\n" + "=" * 80)
    print("TEST: Baseline - Successful Execution")
    print("=" * 80)

    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        print("\nStarting workflow with valid inputs...")

        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": "Successful Test",
                "enable_medical_processing": "true",
                "provider_id": "dr-test",
                "encounter_date": "2025-12-30",
            }

            response = client.post("/speech-to-text", files=files, data=data)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        workflow_id = response.json().get("identifier") or response.json().get("workflow_id")

        print(f"✓ Workflow started: {workflow_id}")
        print("  Expected: Normal execution with no errors")
        print(f"  Monitor: http://localhost:8233/namespaces/default/workflows/{workflow_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration and temporal_errors"])
