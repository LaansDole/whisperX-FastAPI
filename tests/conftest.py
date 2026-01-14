"""Mock fixtures and test data for medical endpoint tests."""

import time
from typing import Tuple, Union

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Shared Workflow Polling Helper
# =============================================================================


def wait_for_workflow_completion(
    client: httpx.Client,
    workflow_id: str,
    max_wait: int = 300,
    poll_interval: int = 30,
    return_tuple: bool = False,
    verbose: bool = True,
) -> Union[dict, Tuple[str, dict], None]:
    """
    Poll Temporal workflow until completion.

    This is the shared helper for all integration tests. Use consistent parameters
    across tests to avoid request storms against the server.

    Args:
        client: HTTP client instance
        workflow_id: Temporal workflow ID to poll
        max_wait: Maximum wait time in seconds (default: 300s = 5 minutes)
        poll_interval: Polling interval in seconds (default: 30s)
        return_tuple: If True, return (status, result) tuple instead of just result
        verbose: If True, print progress messages

    Returns:
        - If return_tuple=False: Workflow result dict if completed, None if timeout
        - If return_tuple=True: Tuple of (status, result_or_error_dict)

    Raises:
        ValueError: If workflow fails and return_tuple=False
        TimeoutError: If timeout and return_tuple=False
    """
    elapsed = 0

    if verbose:
        print(f"\nWaiting for workflow {workflow_id} to complete...")
        print(f"   Polling: every {poll_interval}s, Max wait: {max_wait}s")

    while elapsed < max_wait:
        response = client.get(f"/temporal/workflow/{workflow_id}")

        if response.status_code == 404:
            if return_tuple:
                return ("NOT_FOUND", {"error": f"Workflow not found: {workflow_id}"})
            raise ValueError(f"Workflow not found: {workflow_id}")

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "UNKNOWN")

            if verbose:
                print(f"  [{elapsed}s] Status: {status}")

            if status == "COMPLETED":
                # Fetch result
                result_response = client.get(f"/temporal/workflow/{workflow_id}/result")
                if result_response.status_code == 200:
                    result = result_response.json()
                    if verbose:
                        print(f"[OK] Workflow completed in {elapsed}s")
                    if return_tuple:
                        return ("COMPLETED", result)
                    return result
                else:
                    if verbose:
                        print(f"  Error fetching result: {result_response.status_code}")
                    time.sleep(5)
                    elapsed += 5
                    continue

            elif status in ["FAILED", "TERMINATED", "TIMED_OUT", "CANCELED"]:
                if return_tuple:
                    return (status, data)
                raise ValueError(f"Workflow {status.lower()}")

            else:  # RUNNING, PENDING
                time.sleep(poll_interval)
                elapsed += poll_interval
        else:
            if verbose:
                print(f"  [{elapsed}s] Error: HTTP {response.status_code}")
            time.sleep(poll_interval)
            elapsed += poll_interval

    if verbose:
        print(f"[TIMEOUT] Timeout after {max_wait}s")

    if return_tuple:
        return ("TIMEOUT", {"error": f"Workflow did not complete within {max_wait}s"})
    raise TimeoutError(f"Workflow {workflow_id} did not complete within {max_wait}s")


# Sample LM Studio responses for mocking
MOCK_PHI_DETECTION_RESPONSE = """
{
  "entities": [
    {"type": "name", "text": "John Doe", "start": 0, "end": 8, "confidence": 0.95},
    {"type": "date", "text": "01/15/1980", "start": 20, "end": 30, "confidence": 0.9}
  ]
}
"""

MOCK_ENTITY_EXTRACTION_RESPONSE = """
{
  "entities": [
    {
      "type": "diagnosis",
      "text": "Type 2 Diabetes Mellitus",
      "normalized": "Type 2 Diabetes Mellitus",
      "code": "E11.9",
      "confidence": 0.9
    },
    {
      "type": "medication",
      "text": "metformin 1000mg",
      "normalized": "Metformin",
      "code": "A10BA02",
      "confidence": 0.85
    }
  ]
}
"""

MOCK_SOAP_NOTE_RESPONSE = """
{
  "subjective": "Patient reports elevated blood sugar levels",
  "objective": "Fasting glucose 165 mg/dL, HbA1c 7.8%",
  "assessment": "Type 2 Diabetes Mellitus, uncontrolled",
  "plan": "Start metformin 1000mg BID, lifestyle modifications"
}
"""

MOCK_EMBEDDING_RESPONSE = [0.1] * 1024  # 1024-dimensional mock embedding


@pytest.fixture
def mock_lm_studio_client():
    """Mock LM Studio client for testing without actual LM Studio server."""
    with patch("app.llm.lm_studio_client.LMStudioClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock chat_completion method
        async def mock_chat_completion(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            last_message = messages[-1]["content"].lower() if messages else ""

            # Return different responses based on context
            if "phi" in last_message or "protected" in last_message:
                return MOCK_PHI_DETECTION_RESPONSE
            elif "entities" in last_message or "extract" in last_message:
                return MOCK_ENTITY_EXTRACTION_RESPONSE
            elif "soap" in last_message:
                return MOCK_SOAP_NOTE_RESPONSE
            else:
                return '{"result": "mock response"}'

        mock_client.chat_completion = mock_chat_completion

        # Mock embedding generation
        async def mock_generate_embedding(text, model=None):
            return MOCK_EMBEDDING_RESPONSE

        mock_client.generate_embedding = mock_generate_embedding

        # Mock health check
        async def mock_health_check():
            return {
                "status": "ok",
                "url": "http://localhost:1234/v1",
                "models_loaded": 2,
                "models": [
                    {"id": "mock-medical-model", "object": "model"},
                    {"id": "mock-embedding-model", "object": "model"},
                ],
            }

        mock_client.health_check = mock_health_check

        yield mock_client


@pytest.fixture
def mock_vector_store():
    """Mock medical document vector store for testing."""
    with patch("app.vector_store.medical_vector_store.MedicalDocumentVectorStore") as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        # Mock storage methods
        async def mock_store_consultation(*args, **kwargs):
            return 0  # Return vector_id

        async def mock_store_medical_entities(*args, **kwargs):
            return True

        async def mock_search_similar(*args, **kwargs):
            return [
                {
                    "consultation_id": "cons_mock_123",
                    "encounter_date": "2025-12-18",
                    "similarity_score": 0.85,
                    "transcript": "Mock consultation about diabetes",
                    "provider_id": "dr_mock_provider",
                    "soap_note": {
                        "subjective": "Patient reports elevated blood sugar levels",
                        "objective": "Fasting glucose 165 mg/dL, HbA1c 7.8%",
                        "assessment": "Type 2 Diabetes Mellitus, uncontrolled",
                        "plan": "Start metformin 1000mg BID, lifestyle modifications",
                    },
                    "metadata": {"has_phi": True},
                }
            ]

        mock_store.store_consultation = mock_store_consultation
        mock_store.store_medical_entities = mock_store_medical_entities
        mock_store.search_similar = mock_search_similar
        mock_store.get_statistics = lambda: {
            "total_consultations": 5,
            "unique_patients": 3,
            "vectors_in_index": 5,
            "embedding_dimension": 1024,
        }
        mock_store.close = lambda: None

        yield mock_store


@pytest.fixture
def mock_medical_chatbot():
    """Mock medical chatbot service for testing."""
    with patch("app.llm.chatbot_service.MedicalChatbotService") as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock query method
        async def mock_query(user_query, patient_id_encrypted, session_id=None):
            return {
                "response": "The patient's HbA1c level is 7.8%.",
                "session_id": session_id or "mock_session_123",
                "sources": [
                    {
                        "consultation_id": "cons_mock_123",
                        "encounter_date": "2025-12-18",
                        "similarity_score": 0.85,
                        "provider_id": "dr_mock_provider",
                        "soap_note": {
                            "subjective": "Patient reports elevated blood sugar levels",
                            "objective": "Fasting glucose 165 mg/dL, HbA1c 7.8%",
                            "assessment": "Type 2 Diabetes Mellitus, uncontrolled",
                            "plan": "Start metformin 1000mg BID, lifestyle modifications",
                        },
                        "has_phi": True,
                    }
                ],
                "context_used": True,
                "timestamp": "2025-12-18T11:00:00.000000",
            }

        mock_service.query = mock_query
        mock_service.clear_session = lambda session_id: True

        yield mock_service


@pytest.fixture
def enable_mocking(monkeypatch):
    """Enable mocking for tests by setting appropriate env vars."""
    monkeypatch.setenv("MEDICAL_RAG_ENABLED", "true")
    monkeypatch.setenv("LM_STUDIO_ENABLED", "true")
    monkeypatch.setenv("ENABLE_VECTOR_STORAGE", "true")
    monkeypatch.setenv("ENABLE_AUTHENTICATION", "false")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
