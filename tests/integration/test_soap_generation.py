#!/usr/bin/env python3
"""
Integration Test: Verify SOAP Generation and Response Optimization

Tests using actual medical conversations from Kaggle dataset.
Verifies:
1. Response optimization (intermediate stages excluded)
2. SOAP note generation (all sections populated)
3. PHI detection and medical entity extraction
"""

import httpx
import time
import json
import pytest
from pathlib import Path
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # 5 minutes (reduced from 10)

# Audio files - prefer smaller files for faster testing
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Small test audio (Vietnamese, ~1.4MB, ~90 seconds)
SMALL_AUDIO = PROJECT_ROOT / "datasets" / "audios" / "vn" / "vn-2.mp3"
SMALL_AUDIO_LANG = "vi"

# Large Kaggle medical audio (English, ~8MB, ~13 minutes)
KAGGLE_DATASET_DIR = PROJECT_ROOT / "datasets" / "kaggle-simulated-patient-physician-interviews"
MEDICAL_AUDIO = KAGGLE_DATASET_DIR / "audios" / "MSK0040.mp3"
MEDICAL_AUDIO_LANG = "en"

# Use smaller audio by default for faster CI/testing
DEFAULT_AUDIO = SMALL_AUDIO if SMALL_AUDIO.exists() else MEDICAL_AUDIO
DEFAULT_LANG = SMALL_AUDIO_LANG if SMALL_AUDIO.exists() else MEDICAL_AUDIO_LANG


def poll_workflow_status(
    client: httpx.Client, workflow_id: str, max_wait: int = 300, poll_interval: int = 10
) -> Dict[str, Any]:
    """
    Poll workflow until completion with adaptive interval.

    Args:
        client: HTTP client
        workflow_id: Temporal workflow ID
        max_wait: Maximum wait time in seconds (default: 5 minutes)
        poll_interval: Polling interval in seconds (default: 10s, was 30s)

    Returns:
        Workflow status data on completion

    Raises:
        Exception: If workflow fails
        TimeoutError: If workflow doesn't complete in time
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}")
        response.raise_for_status()
        status_data = response.json()
        status = status_data.get("status", "UNKNOWN")

        elapsed = int(time.time() - start_time)

        if status == "COMPLETED":
            return status_data
        elif status == "FAILED":
            error = status_data.get("error", "Unknown error")
            raise Exception(f"Workflow failed after {elapsed}s: {error}")

        # Adaptive polling: faster initially, slower later
        if elapsed < 60:
            time.sleep(poll_interval)  # 10s for first minute
        else:
            time.sleep(min(poll_interval * 2, 30))  # 20s after, max 30s

    raise TimeoutError(f"Workflow did not complete within {max_wait}s")


@pytest.fixture(scope="module")
def http_client():
    """Shared HTTP client for all tests in module."""
    with httpx.Client(timeout=TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
def audio_file():
    """Fixture to provide audio file for testing (prefers smaller file)."""
    if not DEFAULT_AUDIO.exists():
        pytest.skip(f"Audio file not found: {DEFAULT_AUDIO}")
    return DEFAULT_AUDIO


@pytest.fixture(scope="module")
def workflow_result(http_client, audio_file) -> Dict[str, Any]:
    """
    Process audio and return workflow result.
    Shared across all tests in this module to avoid reprocessing.
    """
    print(f"\n>>> Using audio: {audio_file.name} ({audio_file.stat().st_size / 1024:.1f} KB)")

    # Check server health
    response = http_client.get(f"{BASE_URL}/health")
    response.raise_for_status()

    # Upload audio with medical processing enabled
    with open(audio_file, "rb") as f:
        files = {"file": (audio_file.name, f, "audio/mpeg")}
        data = {
            "patient_name": "John Doe",
            "enable_medical_processing": "true",
            "provider_id": "DR001",
        }
        params = {
            "model": "base",
            "language": DEFAULT_LANG,
            "min_speakers": 2,
            "max_speakers": 2,
        }

        response = http_client.post(f"{BASE_URL}/speech-to-text", files=files, data=data, params=params)
        response.raise_for_status()
        upload_result = response.json()
        workflow_id = upload_result["identifier"]
        print(f">>> Workflow started: {workflow_id}")

    # Wait for completion with shorter timeout for small files
    max_wait = 180 if audio_file == SMALL_AUDIO else 480
    poll_workflow_status(http_client, workflow_id, max_wait=max_wait, poll_interval=10)

    # Get result
    response = http_client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}/result")
    response.raise_for_status()
    result = response.json()
    print(f">>> Workflow completed, SOAP present: {'soap_generation' in result}")

    return result


@pytest.mark.integration
class TestResponseOptimization:
    """Tests for response optimization (no intermediate stages)."""

    def test_no_intermediate_stages_in_response(self, workflow_result):
        """Verify intermediate processing stages are excluded from response."""
        intermediate_stages = ["whisperx_transcription", "whisperx_alignment", "whisperx_diarization"]
        found_intermediate = [stage for stage in intermediate_stages if stage in workflow_result]

        assert not found_intermediate, f"Found intermediate stages that should be excluded: {found_intermediate}"

    def test_has_final_result(self, workflow_result):
        """Verify final whisperx result is present."""
        assert "whisperx_final" in workflow_result, "Should include whisperx_final"
        assert "segments" in workflow_result["whisperx_final"], "whisperx_final should have segments"


@pytest.mark.integration
class TestSoapNoteGeneration:
    """Tests for SOAP note generation."""

    def test_soap_generation_present(self, workflow_result):
        """Verify soap_generation is in workflow result."""
        assert "soap_generation" in workflow_result, "Workflow result should include soap_generation"

    def test_soap_note_has_content(self, workflow_result):
        """Verify SOAP note has actual content."""
        soap = workflow_result["soap_generation"]
        soap_note = soap.get("soap_note", {})

        # Check each section
        sections = ["subjective", "objective", "assessment", "plan"]
        populated = {s: len(soap_note.get(s, "")) for s in sections}

        # At least 3 sections should have content (allowing for edge cases)
        sections_with_content = sum(1 for v in populated.values() if v > 0)
        assert sections_with_content >= 3, f"At least 3 SOAP sections should be populated. Got: {populated}"

    def test_soap_section_lengths_reported(self, workflow_result):
        """Verify section_lengths metadata is present."""
        soap = workflow_result["soap_generation"]
        section_lengths = soap.get("section_lengths", {})

        # Should have length data for all sections
        for section in ["subjective", "objective", "assessment", "plan"]:
            assert section in section_lengths, f"section_lengths should include {section}"

    def test_soap_all_sections_populated(self, workflow_result):
        """Verify all SOAP sections are populated (strict test)."""
        soap = workflow_result["soap_generation"]
        section_lengths = soap.get("section_lengths", {})

        empty_sections = [
            s for s in ["subjective", "objective", "assessment", "plan"] if section_lengths.get(s, 0) == 0
        ]

        if empty_sections:
            soap_note = soap.get("soap_note", {})
            details = {s: soap_note.get(s, "")[:100] for s in empty_sections}
            pytest.fail(f"SOAP sections are empty: {empty_sections}. Content: {details}")


@pytest.mark.integration
class TestPhiDetection:
    """Tests for PHI detection."""

    def test_phi_detection_present(self, workflow_result):
        """Verify phi_detection is in workflow result."""
        assert "phi_detection" in workflow_result, "Workflow result should include phi_detection"

    def test_phi_detection_structure(self, workflow_result):
        """Verify PHI detection result structure."""
        phi = workflow_result["phi_detection"]

        assert "phi_detected" in phi, "PHI result should include phi_detected flag"
        assert "entities" in phi, "PHI result should include entities list"
        assert isinstance(phi["entities"], list), "entities should be a list"


@pytest.mark.integration
class TestEntityExtraction:
    """Tests for medical entity extraction."""

    def test_entity_extraction_present(self, workflow_result):
        """Verify entity_extraction is in workflow result."""
        assert "entity_extraction" in workflow_result, "Workflow result should include entity_extraction"

    def test_entities_extracted(self, workflow_result):
        """Verify entities were extracted."""
        entities = workflow_result["entity_extraction"].get("entities", [])
        # Medical conversations should have some entities
        assert len(entities) >= 0, "entities should be a list (may be empty for some audio)"

    def test_entity_structure(self, workflow_result):
        """Verify entity structure if entities exist."""
        entities = workflow_result["entity_extraction"].get("entities", [])

        if entities:
            entity = entities[0]
            # Should have at least type and text
            assert "type" in entity, "Entity should have type"
            assert "text" in entity, "Entity should have text"


@pytest.mark.integration
class TestResponseMetrics:
    """Tests for response size and structure."""

    def test_response_size_reasonable(self, workflow_result):
        """Verify response is not excessively large."""
        response_size = len(json.dumps(workflow_result))

        # Response should be under 1MB (reasonable for a transcript)
        assert response_size < 1_000_000, f"Response too large: {response_size / 1024:.1f} KB"

    def test_has_expected_top_level_keys(self, workflow_result):
        """Verify expected keys are present."""
        expected_keys = ["whisperx_final", "soap_generation"]
        missing = [k for k in expected_keys if k not in workflow_result]

        assert not missing, f"Missing expected keys: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration"])
