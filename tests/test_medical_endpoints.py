"""Tests for Medical RAG endpoints.

Integration tests for the medical processing pipeline.
Requires server running on localhost:8000 with LM Studio available.
"""

import httpx
import pytest

# Base URL for integration tests - tests hit the actual running server
BASE_URL = "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=120.0)


# Test data
SAMPLE_TRANSCRIPT_DIABETES = (
    "Patient reports Type 2 Diabetes Mellitus. "
    "Fasting glucose 165 mg/dL. HbA1c 7.8%. "
    "Starting metformin 1000mg twice daily. "
    "Lifestyle modifications recommended."
)

SAMPLE_TRANSCRIPT_HYPERTENSION = (
    "Patient Jane Smith presents with hypertension. "
    "Blood pressure 150/95. "
    "Started lisinopril 10mg daily. "
    "Advised dietary modifications and regular exercise."
)

SAMPLE_TRANSCRIPT_ASTHMA = (
    "Patient with acute asthma exacerbation. "
    "Wheezing and shortness of breath. "
    "Peak flow 180 L/min. "
    "Administered albuterol nebulizer. "
    "Prescribed prednisone 40mg daily for 5 days."
)


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.integration
def test_lm_studio_health():
    """Test LM Studio health check endpoint."""
    response = client.get("/health/lm-studio")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "error"]
    assert "url" in data


@pytest.mark.medical
def test_medical_health():
    """Test medical processing health check endpoint."""
    response = client.get("/health/medical")
    # Accept both 200 (when LM Studio available) and 503 (when unavailable)
    assert response.status_code in [200, 503]
    data = response.json()
    assert "components" in data or "detail" in data
    
    # If successful, verify structure
    if response.status_code == 200:
        assert "medical_rag_enabled" in data
        assert "components" in data


# ============================================================================
# Process Transcript Tests
# ============================================================================

@pytest.mark.medical
def test_process_transcript_basic():
    """Test basic transcript processing without vector storage."""
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": SAMPLE_TRANSCRIPT_DIABETES,
            "patient_id": "test_patient_001",
            "provider_id": "test_provider_001",
            "enable_vector_storage": "false"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "consultation_id" in data
    assert "patient_id_encrypted" in data
    assert "provider_id" in data
    assert "steps" in data
    assert "summary" in data


def test_process_transcript_with_phi_detection():
    """Test that PHI detection works correctly."""
    transcript_with_phi = (
        "Patient John Doe reports chronic pain. "
        "DOB: 01/15/1980. MRN: 123456. "
        "Prescribed tramadol 50mg."
    )
    
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": transcript_with_phi,
            "patient_id": "test_patient_phi",
            "provider_id": "test_provider",
            "enable_phi_detection": "true",
            "enable_vector_storage": "false"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check PHI detection ran
    phi_result = data["steps"]["phi_detection"]
    assert phi_result.get("success") is True or phi_result.get("skipped") is False


def test_process_transcript_entity_extraction():
    """Test medical entity extraction."""
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": SAMPLE_TRANSCRIPT_HYPERTENSION,
            "patient_id": "test_patient_htn",
            "provider_id": "test_provider",
            "enable_entity_extraction": "true",
            "enable_vector_storage": "false"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check entity extraction
    entity_result = data["steps"]["entity_extraction"]
    assert entity_result.get("success") is True or entity_result.get("skipped") is False


def test_process_transcript_with_vector_storage():
    """Test transcript processing with vector storage."""
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": SAMPLE_TRANSCRIPT_ASTHMA,
            "patient_id": "test_patient_asthma",
            "provider_id": "test_provider",
            "enable_vector_storage": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check all steps completed
    assert data["summary"]["total_steps"] >= 4
    
    # Verify embedding generation
    embedding_result = data["steps"].get("embedding_generation")
    if embedding_result:
        assert embedding_result.get("success") is True
        assert "dimension" in embedding_result


def test_process_transcript_missing_params():
    """Test that missing required parameters returns 422."""
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": "Test transcript"
            # Missing patient_id and provider_id
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_process_transcript_disabled_features():
    """Test processing with all optional features disabled."""
    response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": "Simple test transcript",
            "patient_id": "test_patient",
            "provider_id": "test_provider",
            "enable_phi_detection": "false",
            "enable_entity_extraction": "false",
            "enable_soap_generation": "false",
            "enable_vector_storage": "false"
        }
    )
    
    # Should still succeed but skip most processing
    assert response.status_code in [200, 503]


# ============================================================================
# Chatbot Endpoint Tests
# ============================================================================

def test_chatbot_query():
    """Test RAG chatbot query endpoint."""
    response = client.post(
        "/medical/chat",
        params={
            "query": "What is the patient's diagnosis?",
            "patient_id_encrypted": "test_encrypted_id_123"
        }
    )
    
    assert response.status_code in [200, 503]  # 503 if no data in vector store
    
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "sources" in data


def test_chatbot_missing_query():
    """Test chatbot with missing query parameter."""
    response = client.post(
        "/medical/chat",
        params={
            "patient_id_encrypted": "test_id"
            # Missing query
        }
    )
    
    assert response.status_code == 422


def test_chatbot_clear_session():
    """Test clearing a chat session."""
    session_id = "test_session_to_clear"
    
    response = client.delete(f"/medical/chat/session/{session_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert session_id in data["message"]


# ============================================================================
# Statistics Endpoint Tests
# ============================================================================

def test_medical_stats():
    """Test medical statistics endpoint."""
    response = client.get("/medical/stats")
    
    assert response.status_code in [200, 503]  # 503 if vector storage disabled
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify statistics structure
        assert "total_consultations" in data
        assert "unique_patients" in data
        assert "vectors_in_index" in data
        assert "embedding_dimension" in data
        assert "configuration" in data


# ============================================================================
# Integration Test
# ============================================================================

def test_end_to_end_pipeline():
    """Test complete pipeline: process transcript -> store -> query chatbot."""
    # Step 1: Process and store a transcript
    process_response = client.post(
        "/medical/process-transcript",
        params={
            "transcript": SAMPLE_TRANSCRIPT_DIABETES,
            "patient_id": "integration_test_patient",
            "provider_id": "integration_test_provider",
            "enable_vector_storage": "true"
        }
    )
    
    assert process_response.status_code == 200
    process_data = process_response.json()
    
    # Verify successful storage
    patient_id_encrypted = process_data["patient_id_encrypted"]
    
    # Step 2: Query the chatbot about the stored data
    chat_response = client.post(
        "/medical/chat",
        params={
            "query": "What is this patient's HbA1c level?",
            "patient_id_encrypted": patient_id_encrypted
        }
    )
    
    assert chat_response.status_code in [200, 503]
    
    if chat_response.status_code == 200:
        chat_data = chat_response.json()
        assert "response" in chat_data
        assert "sources" in chat_data
    
    # Step 3: Verify stats were updated
    stats_response = client.get("/medical/stats")
    assert stats_response.status_code in [200, 503]
