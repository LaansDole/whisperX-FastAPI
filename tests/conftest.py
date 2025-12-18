"""Mock fixtures and test data for medical endpoint tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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
    with patch('app.llm.lm_studio_client.LMStudioClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock chat_completion method
        async def mock_chat_completion(*args, **kwargs):
            messages = kwargs.get('messages', args[0] if args else [])
            last_message = messages[-1]['content'].lower() if messages else ""
            
            # Return different responses based on context
            if 'phi' in last_message or 'protected' in last_message:
                return MOCK_PHI_DETECTION_RESPONSE
            elif 'entities' in last_message or 'extract' in last_message:
                return MOCK_ENTITY_EXTRACTION_RESPONSE
            elif 'soap' in last_message:
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
                    {"id": "mock-embedding-model", "object": "model"}
                ]
            }
        
        mock_client.health_check = mock_health_check
        
        yield mock_client


@pytest.fixture
def mock_vector_store():
    """Mock medical document vector store for testing."""
    with patch('app.vector_store.medical_vector_store.MedicalDocumentVectorStore') as mock_store_class:
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
                    "transcript": "Mock consultation about diabetes"
                }
            ]
        
        mock_store.store_consultation = mock_store_consultation
        mock_store.store_medical_entities = mock_store_medical_entities
        mock_store.search_similar = mock_search_similar
        mock_store.get_statistics = lambda: {
            "total_consultations": 5,
            "unique_patients": 3,
            "vectors_in_index": 5,
            "embedding_dimension": 1024
        }
        mock_store.close = lambda: None
        
        yield mock_store


@pytest.fixture
def mock_medical_chatbot():
    """Mock medical chatbot service for testing."""
    with patch('app.llm.chatbot_service.MedicalChatbotService') as mock_service_class:
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
                        "similarity_score": 0.85
                    }
                ],
                "context_used": True,
                "timestamp": "2025-12-18T11:00:00.000000"
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
