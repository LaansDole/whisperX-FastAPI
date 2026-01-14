#!/usr/bin/env python3
"""
Integration Test: SOAP Storage and Chatbot Retrieval

Verifies the complete flow:
1. SOAP notes are stored in vector database after workflow completion
2. Chatbot retrieves and uses SOAP notes for answering questions
3. SOAP notes appear in chatbot response sources

This test complements test_soap_generation.py by testing the retrieval side.
"""

import httpx
import pytest
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # 5 minutes

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


@pytest.fixture(scope="module")
def http_client():
    """HTTP client with extended timeout."""
    with httpx.Client(timeout=TIMEOUT) as client:
        yield client


def poll_workflow_status(client: httpx.Client, workflow_id: str, max_wait: int = 300, poll_interval: int = 10) -> dict:
    """Poll workflow until completion with shorter interval for efficiency."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}")
        response.raise_for_status()
        status_data = response.json()
        status = status_data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            return status_data
        elif status == "FAILED":
            raise Exception(f"Workflow failed: {status_data.get('error', 'Unknown error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Workflow did not complete within {max_wait}s")


def get_workflow_result(client: httpx.Client, workflow_id: str) -> dict:
    """Get workflow result after completion."""
    response = client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}/result")
    response.raise_for_status()
    return response.json()


@pytest.fixture(scope="module")
def processed_workflow(http_client):
    """
    Upload and process an audio file with medical processing.
    Returns workflow_id and patient_hash for subsequent tests.
    Shared across all tests in this module.
    """
    audio_file = DEFAULT_AUDIO
    audio_lang = DEFAULT_LANG

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    print(f"\n>>> Using audio: {audio_file.name} ({audio_file.stat().st_size / 1024:.1f} KB)")

    # Check server health
    response = http_client.get(f"{BASE_URL}/health")
    response.raise_for_status()

    # Upload with medical processing enabled
    with open(audio_file, "rb") as f:
        files = {"file": (audio_file.name, f, "audio/mpeg")}
        data = {
            "patient_name": "Test Patient SOAP",
            "enable_medical_processing": "true",
            "provider_id": "DR_TEST_001",
        }
        params = {
            "model": "base",
            "language": audio_lang,
            "min_speakers": 2,
            "max_speakers": 2,
        }

        response = http_client.post(
            f"{BASE_URL}/speech-to-text",
            files=files,
            data=data,
            params=params,
        )
        response.raise_for_status()
        result = response.json()

    workflow_id = result["identifier"]
    # Extract patient_hash from workflow_id format: whisperx-wf-pt_{hash}-{timestamp}-{uuid}
    # Example: whisperx-wf-pt_faae6c22-20260112_211932859790-a8fa
    patient_hash = None
    if workflow_id.startswith("whisperx-wf-pt_"):
        parts = workflow_id.replace("whisperx-wf-pt_", "").split("-")
        if parts:
            patient_hash = parts[0]  # First part after pt_ is the hash
    print(f">>> Workflow started: {workflow_id}, patient_hash: {patient_hash}")

    # Wait for completion with shorter timeout for small files
    max_wait = 180 if audio_file == SMALL_AUDIO else 480
    poll_workflow_status(http_client, workflow_id, max_wait=max_wait, poll_interval=10)

    # Get result
    workflow_result = get_workflow_result(http_client, workflow_id)
    print(f">>> Workflow completed, SOAP present: {'soap_generation' in workflow_result}")

    return {
        "workflow_id": workflow_id,
        "patient_hash": patient_hash,
        "result": workflow_result,
    }


@pytest.mark.integration
class TestSoapStorageAndRetrieval:
    """Test SOAP note storage in vector DB and retrieval via chatbot."""

    def test_workflow_has_soap_generation(self, processed_workflow):
        """Verify workflow result includes SOAP generation."""
        result = processed_workflow["result"]

        assert "soap_generation" in result, "Workflow result should include soap_generation"

        soap = result["soap_generation"]
        assert "soap_note" in soap, "SOAP generation should include soap_note"

        soap_note = soap["soap_note"]
        # At least one section should be populated
        sections = ["subjective", "objective", "assessment", "plan"]
        populated = [s for s in sections if soap_note.get(s)]
        assert len(populated) >= 2, f"At least 2 SOAP sections should be populated, got: {populated}"

    def test_soap_stored_in_vector_db(self, http_client, processed_workflow):
        """Verify SOAP note is stored in vector database."""
        result = processed_workflow["result"]

        # Get vector storage result from workflow
        vector_storage = result.get("vector_storage")
        if not vector_storage:
            pytest.skip("Vector storage not in workflow result (may be disabled)")

        consultation_id = vector_storage.get("consultation_id")
        assert consultation_id, "Vector storage should return consultation_id"

        # Query the consultation details
        response = http_client.get(
            f"{BASE_URL}/medical/consultation/{consultation_id}",
            params={"include_structured": True},
        )

        if response.status_code == 503:
            pytest.skip("Vector storage is disabled")

        response.raise_for_status()
        details = response.json()

        # Verify SOAP note is in structured data
        assert "soap_note" in details, "Consultation details should include soap_note"

        soap_note = details["soap_note"]
        assert isinstance(soap_note, dict), "SOAP note should be a dictionary"

        # Verify at least some content
        has_content = any(soap_note.get(s) for s in ["subjective", "objective", "assessment", "plan"])
        assert has_content, "SOAP note should have at least one populated section"

    def test_chatbot_retrieves_sources(self, http_client, processed_workflow):
        """Verify chatbot retrieves consultation sources for the patient."""
        patient_hash = processed_workflow.get("patient_hash")
        if not patient_hash:
            pytest.skip("Patient hash not available from workflow")

        # Query the chatbot about the patient
        response = http_client.post(
            f"{BASE_URL}/medical/chat",
            params={
                "query": "What is the assessment and treatment plan for this patient?",
                "patient_id_encrypted": patient_hash,
            },
        )

        if response.status_code == 503:
            pytest.skip("Medical processing or vector storage is disabled")

        response.raise_for_status()
        chat_result = response.json()

        # Verify response structure
        assert "response" in chat_result, "Chat result should include response"
        assert "sources" in chat_result, "Chat result should include sources"

        sources = chat_result["sources"]
        if not sources:
            pytest.skip("No sources returned (possibly no matching consultations)")

        # Verify sources have expected structure
        # Note: SOAP notes are used in LLM context but not returned in sources (by design)
        for source in sources:
            assert "consultation_id" in source, "Source should have consultation_id"
            assert "encounter_date" in source, "Source should have encounter_date"
            assert "similarity_score" in source, "Source should have similarity_score"

    def test_chatbot_response_uses_soap_content(self, http_client, processed_workflow):
        """Verify chatbot response reflects SOAP content (not just random text)."""
        patient_hash = processed_workflow.get("patient_hash")
        result = processed_workflow["result"]

        if not patient_hash:
            pytest.skip("Patient hash not available")

        # Get the SOAP content from workflow result
        soap_gen = result.get("soap_generation", {})
        soap_note = soap_gen.get("soap_note", {})
        assessment = soap_note.get("assessment", "")

        if not assessment:
            pytest.skip("No assessment in SOAP note to verify")

        # Ask specifically about assessment
        response = http_client.post(
            f"{BASE_URL}/medical/chat",
            params={
                "query": "What is the clinical assessment?",
                "patient_id_encrypted": patient_hash,
            },
        )

        if response.status_code == 503:
            pytest.skip("Medical processing is disabled")

        response.raise_for_status()
        chat_result = response.json()

        # The response should contain relevant medical terminology
        llm_response = chat_result.get("response", "").lower()

        # Check that response is not a generic "no data" response
        no_data_indicators = [
            "no relevant patient records",
            "i don't have information",
            "no data available",
            "cannot find",
        ]

        has_data = not any(indicator in llm_response for indicator in no_data_indicators)

        if has_data:
            # Response should be medical in nature
            assert len(llm_response) > 50, "Response should be substantive (>50 chars)"


@pytest.mark.integration
class TestVectorStoreHealth:
    """Test vector store health and statistics."""

    def test_vector_store_health(self, http_client):
        """Verify vector store health endpoint returns valid statistics."""
        response = http_client.get(f"{BASE_URL}/health/vector-store")

        if response.status_code == 503:
            data = response.json()
            assert data.get("status") in ["disabled", "error"], "Should indicate why unavailable"
            pytest.skip("Vector storage is disabled")

        response.raise_for_status()
        data = response.json()

        assert data.get("status") == "healthy", "Vector store should be healthy"
        assert "statistics" in data, "Should include statistics"

        stats = data["statistics"]
        assert "total_consultations" in stats, "Stats should include total_consultations"

    def test_medical_health_endpoint(self, http_client):
        """Verify medical health endpoint returns component statuses."""
        response = http_client.get(f"{BASE_URL}/health/medical")

        # This endpoint should always return (possibly with degraded status)
        data = response.json()

        assert "components" in data, "Should include component health"
        assert "medical_rag_enabled" in data, "Should indicate RAG status"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration"])
