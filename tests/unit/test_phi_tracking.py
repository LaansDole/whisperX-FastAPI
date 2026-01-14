"""Unit tests for PHI tracking and privacy status formatting.

Tests the metadata extraction, phi_status formatting, and privacy status display
functionality added as part of the PHI awareness feature.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestPHIMetadataExtraction:
    """Test cases for extracting has_phi from metadata."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="Mock response")

        service = MedicalChatbotService(client=mock_client)
        return service

    def test_extract_has_phi_true(self, chatbot_service):
        """Test that has_phi=True is correctly extracted from metadata."""
        results = [
            {
                "consultation_id": "cons_001",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[PHI Protected]" in context
        assert "[No PHI Detected]" not in context

    def test_extract_has_phi_false(self, chatbot_service):
        """Test that has_phi=False is correctly extracted from metadata."""
        results = [
            {
                "consultation_id": "cons_002",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[No PHI Detected]" in context
        assert "[PHI Protected]" not in context

    def test_extract_has_phi_missing_metadata(self, chatbot_service):
        """Test that missing metadata defaults to has_phi=False."""
        results = [
            {
                "consultation_id": "cons_003",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.8,
                # No metadata field at all
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[No PHI Detected]" in context
        assert "[PHI Protected]" not in context

    def test_extract_has_phi_empty_metadata(self, chatbot_service):
        """Test that empty metadata dict defaults to has_phi=False."""
        results = [
            {
                "consultation_id": "cons_004",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.75,
                "metadata": {},  # Empty dict
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[No PHI Detected]" in context
        assert "[PHI Protected]" not in context

    def test_extract_has_phi_none_metadata(self, chatbot_service):
        """Test that None metadata defaults to has_phi=False."""
        results = [
            {
                "consultation_id": "cons_005",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.7,
                "metadata": None,
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[No PHI Detected]" in context
        assert "[PHI Protected]" not in context

    def test_extract_has_phi_missing_has_phi_key(self, chatbot_service):
        """Test that metadata without has_phi key defaults to False."""
        results = [
            {
                "consultation_id": "cons_006",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.65,
                "metadata": {"other_field": "value"},  # has_phi key missing
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "[No PHI Detected]" in context
        assert "[PHI Protected]" not in context


class TestPHIStatusFormatting:
    """Test cases for PHI status string formatting."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="Mock response")

        service = MedicalChatbotService(client=mock_client)
        return service

    def test_phi_status_appears_in_context(self, chatbot_service):
        """Test that Privacy Status label appears in formatted context."""
        results = [
            {
                "consultation_id": "cons_007",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "Privacy Status:" in context

    def test_phi_status_appears_after_consultation_id(self, chatbot_service):
        """Test that Privacy Status appears after Consultation ID in context."""
        results = [
            {
                "consultation_id": "cons_008",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        cons_id_pos = context.find("Consultation ID:")
        privacy_pos = context.find("Privacy Status:")

        assert cons_id_pos < privacy_pos

    def test_phi_protected_format_exact_string(self, chatbot_service):
        """Test exact format of PHI Protected status string."""
        results = [
            {
                "consultation_id": "cons_009",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "Privacy Status: [PHI Protected]" in context

    def test_no_phi_format_exact_string(self, chatbot_service):
        """Test exact format of No PHI Detected status string."""
        results = [
            {
                "consultation_id": "cons_010",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        assert "Privacy Status: [No PHI Detected]" in context


class TestMultipleConsultationsPHIStatus:
    """Test cases for multiple consultations with mixed PHI status."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="Mock response")

        service = MedicalChatbotService(client=mock_client)
        return service

    def test_mixed_phi_status_multiple_consultations(self, chatbot_service):
        """Test that mixed PHI status is correctly displayed for multiple consultations."""
        results = [
            {
                "consultation_id": "cons_011",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            },
            {
                "consultation_id": "cons_012",
                "encounter_date": "2025-01-13",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            },
            {
                "consultation_id": "cons_013",
                "encounter_date": "2025-01-12",
                "similarity_score": 0.8,
                "metadata": {"has_phi": True},
            },
        ]

        context = chatbot_service._format_context_from_results(results)

        # Count occurrences
        phi_protected_count = context.count("[PHI Protected]")
        no_phi_count = context.count("[No PHI Detected]")

        assert phi_protected_count == 2
        assert no_phi_count == 1

    def test_all_consultations_with_phi(self, chatbot_service):
        """Test that all consultations correctly show PHI Protected."""
        results = [
            {
                "consultation_id": f"cons_01{i}",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9 - (i * 0.1),
                "metadata": {"has_phi": True},
            }
            for i in range(3)
        ]

        context = chatbot_service._format_context_from_results(results)

        phi_protected_count = context.count("[PHI Protected]")
        no_phi_count = context.count("[No PHI Detected]")

        assert phi_protected_count == 3
        assert no_phi_count == 0

    def test_all_consultations_without_phi(self, chatbot_service):
        """Test that all consultations correctly show No PHI Detected."""
        results = [
            {
                "consultation_id": f"cons_02{i}",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9 - (i * 0.1),
                "metadata": {"has_phi": False},
            }
            for i in range(3)
        ]

        context = chatbot_service._format_context_from_results(results)

        phi_protected_count = context.count("[PHI Protected]")
        no_phi_count = context.count("[No PHI Detected]")

        assert phi_protected_count == 0
        assert no_phi_count == 3

    def test_phi_status_per_consultation_order_preserved(self, chatbot_service):
        """Test that PHI status matches each consultation in order."""
        results = [
            {
                "consultation_id": "cons_phi_1",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            },
            {
                "consultation_id": "cons_no_phi_1",
                "encounter_date": "2025-01-13",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            },
        ]

        context = chatbot_service._format_context_from_results(results)

        # Find positions
        cons_phi_1_pos = context.find("cons_phi_1")
        cons_no_phi_1_pos = context.find("cons_no_phi_1")
        first_phi_protected = context.find("[PHI Protected]")
        first_no_phi = context.find("[No PHI Detected]")

        # Verify PHI Protected comes after cons_phi_1 and before cons_no_phi_1
        assert cons_phi_1_pos < first_phi_protected < cons_no_phi_1_pos
        # Verify No PHI Detected comes after cons_no_phi_1
        assert cons_no_phi_1_pos < first_no_phi


class TestPHISourcesInQueryResponse:
    """Test cases for has_phi field in query response sources."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="The patient has diabetes.")

        service = MedicalChatbotService(client=mock_client)
        return service

    @pytest.mark.asyncio
    async def test_has_phi_included_in_sources_true(self, chatbot_service):
        """Test that has_phi=True is included in query response sources."""
        mock_context = [
            {
                "consultation_id": "cons_test_phi",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "provider_id": "dr_test",
                "soap_note": {"assessment": "Test"},
                "metadata": {"has_phi": True},
            }
        ]

        result = await chatbot_service.query(
            user_query="What is the diagnosis?",
            patient_id_encrypted="test_patient",
            session_id="test_session",
            additional_context=mock_context,
        )

        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["has_phi"] is True

    @pytest.mark.asyncio
    async def test_has_phi_included_in_sources_false(self, chatbot_service):
        """Test that has_phi=False is included in query response sources."""
        mock_context = [
            {
                "consultation_id": "cons_test_no_phi",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.85,
                "provider_id": "dr_test",
                "soap_note": {"assessment": "Test"},
                "metadata": {"has_phi": False},
            }
        ]

        result = await chatbot_service.query(
            user_query="What is the diagnosis?",
            patient_id_encrypted="test_patient",
            session_id="test_session",
            additional_context=mock_context,
        )

        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["has_phi"] is False

    @pytest.mark.asyncio
    async def test_has_phi_defaults_false_missing_metadata(self, chatbot_service):
        """Test that has_phi defaults to False when metadata is missing."""
        mock_context = [
            {
                "consultation_id": "cons_test_no_metadata",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.8,
                "provider_id": "dr_test",
                "soap_note": {"assessment": "Test"},
                # No metadata field
            }
        ]

        result = await chatbot_service.query(
            user_query="What is the diagnosis?",
            patient_id_encrypted="test_patient",
            session_id="test_session",
            additional_context=mock_context,
        )

        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["has_phi"] is False

    @pytest.mark.asyncio
    async def test_has_phi_multiple_sources_mixed(self, chatbot_service):
        """Test that has_phi is correctly set for multiple sources with mixed status."""
        mock_context = [
            {
                "consultation_id": "cons_with_phi",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": True},
            },
            {
                "consultation_id": "cons_without_phi",
                "encounter_date": "2025-01-13",
                "similarity_score": 0.85,
                "metadata": {"has_phi": False},
            },
            {
                "consultation_id": "cons_no_metadata",
                "encounter_date": "2025-01-12",
                "similarity_score": 0.8,
                # No metadata
            },
        ]

        result = await chatbot_service.query(
            user_query="What is the diagnosis?",
            patient_id_encrypted="test_patient",
            session_id="test_session",
            additional_context=mock_context,
        )

        assert len(result["sources"]) == 3
        assert result["sources"][0]["has_phi"] is True
        assert result["sources"][1]["has_phi"] is False
        assert result["sources"][2]["has_phi"] is False


class TestPHIEdgeCases:
    """Test edge cases for PHI tracking."""

    @pytest.fixture
    def chatbot_service(self):
        """Create a MedicalChatbotService with mocked LM Studio client."""
        from app.llm.chatbot_service import MedicalChatbotService

        mock_client = MagicMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.chat_completion = AsyncMock(return_value="Mock response")

        service = MedicalChatbotService(client=mock_client)
        return service

    def test_has_phi_non_boolean_truthy_value(self, chatbot_service):
        """Test that truthy non-boolean values are handled correctly."""
        results = [
            {
                "consultation_id": "cons_edge_1",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.9,
                "metadata": {"has_phi": 1},  # Truthy integer
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        # Should treat truthy value as True
        assert "[PHI Protected]" in context

    def test_has_phi_non_boolean_falsy_value(self, chatbot_service):
        """Test that falsy non-boolean values are handled correctly."""
        results = [
            {
                "consultation_id": "cons_edge_2",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.85,
                "metadata": {"has_phi": 0},  # Falsy integer
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        # Should treat falsy value as False
        assert "[No PHI Detected]" in context

    def test_has_phi_string_true(self, chatbot_service):
        """Test that string 'true' is handled (edge case from JSON deserialization)."""
        results = [
            {
                "consultation_id": "cons_edge_3",
                "encounter_date": "2025-01-14",
                "similarity_score": 0.8,
                "metadata": {"has_phi": "true"},  # String instead of boolean
            }
        ]

        context = chatbot_service._format_context_from_results(results)
        # Non-empty string is truthy
        assert "[PHI Protected]" in context

    def test_empty_results_no_phi_status(self, chatbot_service):
        """Test that empty results don't show PHI status."""
        results = []

        context = chatbot_service._format_context_from_results(results)
        assert "[PHI Protected]" not in context
        assert "[No PHI Detected]" not in context
        assert "No relevant patient records found." in context
