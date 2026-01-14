"""Unit tests for chatbot SOAP note context formatting.

Verifies that SOAP notes are correctly injected into LLM context
by the MedicalChatbotService._format_context_from_results() method.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock


# Sample search results with SOAP notes
SEARCH_RESULTS_WITH_SOAP = [
    {
        "consultation_id": "cons_test_001",
        "encounter_date": "2025-01-10",
        "similarity_score": 0.85,
        "soap_note": {
            "subjective": "Patient reports persistent headaches for 3 days.",
            "objective": "BP 130/85, pulse 78. No neurological deficits.",
            "assessment": "Tension-type headache, likely stress-related.",
            "plan": "Ibuprofen 400mg PRN, stress management counseling.",
        },
        "medical_entities": [
            {"type": "diagnosis", "text": "Tension-type headache"},
            {"type": "medication", "text": "Ibuprofen 400mg"},
        ],
        "structured_document": {
            "clinical_summary": "Follow-up for headache symptoms.",
        },
    },
    {
        "consultation_id": "cons_test_002",
        "encounter_date": "2025-01-05",
        "similarity_score": 0.72,
        "soap_note": {
            "subjective": "Patient complains of fatigue and low energy.",
            "objective": "Vitals normal. Labs show low vitamin D.",
            "assessment": "Vitamin D deficiency contributing to fatigue.",
            "plan": "Vitamin D3 2000 IU daily, recheck in 3 months.",
        },
        "medical_entities": [
            {"type": "diagnosis", "text": "Vitamin D deficiency"},
            {"type": "medication", "text": "Vitamin D3 2000 IU"},
        ],
    },
]

SEARCH_RESULTS_WITHOUT_SOAP = [
    {
        "consultation_id": "cons_test_003",
        "encounter_date": "2025-01-01",
        "similarity_score": 0.65,
        "medical_entities": [
            {"type": "diagnosis", "text": "Hypertension"},
        ],
    },
]

SEARCH_RESULTS_PARTIAL_SOAP = [
    {
        "consultation_id": "cons_test_004",
        "encounter_date": "2025-01-08",
        "similarity_score": 0.78,
        "soap_note": {
            "assessment": "Suspected gastritis.",
            # Missing subjective, objective, plan
        },
    },
]

SEARCH_RESULTS_WITH_PHI = [
    {
        "consultation_id": "cons_test_005",
        "encounter_date": "2025-01-12",
        "similarity_score": 0.90,
        "metadata": {"has_phi": True},
        "soap_note": {
            "assessment": "Type 2 diabetes mellitus.",
            "plan": "Metformin 500mg twice daily.",
        },
    },
]

SEARCH_RESULTS_WITHOUT_PHI = [
    {
        "consultation_id": "cons_test_006",
        "encounter_date": "2025-01-13",
        "similarity_score": 0.88,
        "metadata": {"has_phi": False},
        "soap_note": {
            "assessment": "Seasonal allergies.",
            "plan": "Antihistamine as needed.",
        },
    },
]


class TestChatbotSoapContextFormatting:
    """Test cases for SOAP note injection into LLM context."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="Mock response")

        service = MedicalChatbotService(client=mock_client)
        return service

    def test_format_context_includes_soap_assessment(self, chatbot_service):
        """Test that SOAP Assessment is included in formatted context."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Assessment: Tension-type headache, likely stress-related." in context
        assert "Assessment: Vitamin D deficiency contributing to fatigue." in context

    def test_format_context_includes_soap_plan(self, chatbot_service):
        """Test that SOAP Plan is included in formatted context."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Plan: Ibuprofen 400mg PRN, stress management counseling." in context
        assert "Plan: Vitamin D3 2000 IU daily, recheck in 3 months." in context

    def test_format_context_includes_clinical_summary(self, chatbot_service):
        """Test that clinical summary from structured document is included."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Clinical Summary: Follow-up for headache symptoms." in context

    def test_format_context_includes_diagnoses(self, chatbot_service):
        """Test that diagnoses from medical entities are included."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Diagnoses: Tension-type headache" in context
        assert "Diagnoses: Vitamin D deficiency" in context

    def test_format_context_includes_medications(self, chatbot_service):
        """Test that medications from medical entities are included."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Medications: Ibuprofen 400mg" in context
        assert "Medications: Vitamin D3 2000 IU" in context

    def test_format_context_includes_consultation_metadata(self, chatbot_service):
        """Test that consultation metadata is included."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        assert "Consultation 1" in context
        assert "Date: 2025-01-10" in context
        assert "Consultation ID: cons_test_001" in context
        assert "0.85" in context  # Similarity score

    def test_format_context_handles_missing_soap(self, chatbot_service):
        """Test graceful handling when SOAP note is missing."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITHOUT_SOAP)

        # Should still include basic info
        assert "Consultation 1" in context
        assert "cons_test_003" in context
        # Should include diagnoses
        assert "Diagnoses: Hypertension" in context
        # Should NOT have Assessment or Plan headers without SOAP
        assert "Assessment:" not in context
        assert "Plan:" not in context

    def test_format_context_handles_partial_soap(self, chatbot_service):
        """Test handling of partial SOAP notes (some sections missing)."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_PARTIAL_SOAP)

        # Should include the assessment that exists
        assert "Assessment: Suspected gastritis." in context
        # Should NOT include Plan since it's missing
        assert "Plan:" not in context

    def test_format_context_empty_results(self, chatbot_service):
        """Test handling of empty search results."""
        context = chatbot_service._format_context_from_results([])

        assert "No relevant patient records found." in context

    def test_format_context_preserves_order(self, chatbot_service):
        """Test that consultations are formatted in order."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        # First consultation should appear before second
        idx_cons1 = context.find("cons_test_001")
        idx_cons2 = context.find("cons_test_002")
        assert idx_cons1 < idx_cons2

    def test_soap_assessment_before_plan_in_context(self, chatbot_service):
        """Test that Assessment appears before Plan in context (clinical order)."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        # For the first consultation, Assessment should come before Plan
        first_assessment_idx = context.find("Assessment: Tension-type headache")
        first_plan_idx = context.find("Plan: Ibuprofen 400mg")
        assert first_assessment_idx < first_plan_idx

    def test_format_context_includes_phi_protected_status(self, chatbot_service):
        """Test that PHI Protected status is displayed for records with PHI."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_PHI)

        assert "[PHI Protected]" in context
        assert "cons_test_005" in context

    def test_format_context_includes_no_phi_status(self, chatbot_service):
        """Test that No PHI Detected status is displayed for clean records."""
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITHOUT_PHI)

        assert "[No PHI Detected]" in context
        assert "cons_test_006" in context

    def test_format_context_defaults_to_no_phi_when_metadata_missing(self, chatbot_service):
        """Test that missing metadata defaults to No PHI Detected."""
        # Use existing test data that has no metadata field
        context = chatbot_service._format_context_from_results(SEARCH_RESULTS_WITH_SOAP)

        # Should default to [No PHI Detected] when metadata is missing
        assert "[No PHI Detected]" in context


class TestChatbotSoapContextIntegration:
    """Integration-style tests for SOAP context in LLM prompts."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="The patient has tension headache.")

        service = MedicalChatbotService(client=mock_client)
        return service

    @pytest.mark.asyncio
    async def test_soap_context_passed_to_llm(self, chatbot_service):
        """Test that SOAP context is passed to LLM in system message."""
        # Call query with pre-fetched context (bypasses vector search)
        _result = await chatbot_service.query(
            user_query="What is the patient's diagnosis?",
            patient_id_encrypted="test_patient_hash",
            session_id="test_session",
            additional_context=SEARCH_RESULTS_WITH_SOAP,
        )

        # Verify chat_completion was called
        chatbot_service.client.chat_completion.assert_called_once()

        # Get the messages passed to chat_completion
        call_args = chatbot_service.client.chat_completion.call_args
        messages = call_args[1].get("messages") or call_args[0][0]

        # Find system message
        system_message = next((m for m in messages if m["role"] == "system"), None)
        assert system_message is not None

        # Verify SOAP content is in system message
        content = system_message["content"]
        assert "Assessment: Tension-type headache" in content
        assert "Plan: Ibuprofen 400mg" in content

    @pytest.mark.asyncio
    async def test_query_returns_sources_with_soap(self, chatbot_service):
        """Test that query result includes sources for transparency."""
        result = await chatbot_service.query(
            user_query="What medications is the patient on?",
            patient_id_encrypted="test_patient_hash",
            session_id="test_session_2",
            additional_context=SEARCH_RESULTS_WITH_SOAP,
        )

        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["sources"][0]["consultation_id"] == "cons_test_001"
