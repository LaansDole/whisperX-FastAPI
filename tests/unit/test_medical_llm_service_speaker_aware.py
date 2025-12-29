"""Unit tests for speaker-aware medical LLM service methods.

Tests the new speaker-attributed dialogue processing methods that leverage
doctor/patient speaker roles for improved medical documentation accuracy.
"""

import pytest
from unittest.mock import AsyncMock
from app.llm.medical_llm_service import MedicalLLMService


# Sample speaker-attributed dialogue data (from TranscriptionTransformer output)
SAMPLE_DIALOGUE_DATA = {
    "dialogue": [
        {
            "speaker_id": "SPEAKER_00",
            "speaker_role": "doctor",
            "text": "What brings you in today?",
            "start_time": 0.0,
            "end_time": 2.5,
            "duration": 2.5,
            "confidence": 0.95,
            "word_count": 5,
        },
        {
            "speaker_id": "SPEAKER_01",
            "speaker_role": "patient",
            "text": "I've been having chest pain for the past week.",
            "start_time": 2.6,
            "end_time": 6.0,
            "duration": 3.4,
            "confidence": 0.92,
            "word_count": 9,
        },
        {
            "speaker_id": "SPEAKER_00",
            "speaker_role": "doctor",
            "text": "Can you describe the pain? Is it sharp or dull?",
            "start_time": 6.1,
            "end_time": 9.0,
            "duration": 2.9,
            "confidence": 0.94,
            "word_count": 10,
        },
        {
            "speaker_id": "SPEAKER_01",
            "speaker_role": "patient",
            "text": "It's a sharp pain, mostly on the left side.",
            "start_time": 9.1,
            "end_time": 12.0,
            "duration": 2.9,
            "confidence": 0.90,
            "word_count": 10,
        },
    ],
    "speaker_mapping": {
        "SPEAKER_00": {"role": "doctor", "confidence": 0.95, "method": "heuristic"},
        "SPEAKER_01": {"role": "patient", "confidence": 0.95, "method": "heuristic"},
    },
    "consultation_metadata": {
        "total_duration": 12.0,
        "total_speakers": 2,
        "total_segments": 4,
        "total_words": 34,
        "avg_confidence": 0.93,
        "has_speaker_labels": True,
        "speaker_label_coverage": 1.0,
    },
    "statistics": {
        "by_speaker": {
            "SPEAKER_00": {
                "speaker_id": "SPEAKER_00",
                "speaker_role": "doctor",
                "segment_count": 2,
                "word_count": 15,
                "speaking_time": 5.4,
                "avg_confidence": 0.945,
            },
            "SPEAKER_01": {
                "speaker_id": "SPEAKER_01",
                "speaker_role": "patient",
                "segment_count": 2,
                "word_count": 19,
                "speaking_time": 6.3,
                "avg_confidence": 0.91,
            },
        },
        "totals": {"total_speaking_time": 11.7, "total_segments": 4, "total_words": 34},
    },
}


SAMPLE_DIALOGUE_NO_ROLES = {
    "dialogue": [
        {
            "speaker_id": "SPEAKER_00",
            "speaker_role": "unknown",
            "text": "Hello, how are you feeling today?",
            "start_time": 0.0,
            "end_time": 2.0,
        },
        {
            "speaker_id": "SPEAKER_01",
            "speaker_role": "unknown",
            "text": "Not great, I have a headache.",
            "start_time": 2.1,
            "end_time": 4.0,
        },
    ],
    "speaker_mapping": {
        "SPEAKER_00": {"role": "unknown", "confidence": 0.5, "method": "default"},
        "SPEAKER_01": {"role": "unknown", "confidence": 0.5, "method": "default"},
    },
    "consultation_metadata": {},
    "statistics": {},
}


@pytest.fixture
def mock_lm_studio_client():
    """Create a mock LM Studio client."""
    client = AsyncMock()
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def medical_service(mock_lm_studio_client):
    """Create medical service with mocked client."""
    return MedicalLLMService(client=mock_lm_studio_client)


# ============================================================================
# Test _format_dialogue_for_prompt Helper
# ============================================================================


def test_format_dialogue_for_prompt(medical_service):
    """Test dialogue formatting for LLM prompts."""
    formatted = medical_service._format_dialogue_for_prompt(SAMPLE_DIALOGUE_DATA)

    # Check format
    assert "Doctor: What brings you in today?" in formatted
    assert "Patient: I've been having chest pain for the past week." in formatted
    assert "Doctor: Can you describe the pain? Is it sharp or dull?" in formatted
    assert "Patient: It's a sharp pain, mostly on the left side." in formatted

    # Check structure
    lines = formatted.split("\n")
    assert len(lines) == 4
    assert all(line.count(":") == 1 for line in lines)


def test_format_dialogue_empty(medical_service):
    """Test formatting empty dialogue."""
    empty_dialogue = {"dialogue": []}
    formatted = medical_service._format_dialogue_for_prompt(empty_dialogue)
    assert formatted == "No dialogue available"


def test_format_dialogue_with_unknown_roles(medical_service):
    """Test formatting dialogue with unknown speaker roles."""
    formatted = medical_service._format_dialogue_for_prompt(SAMPLE_DIALOGUE_NO_ROLES)

    assert "Unknown: Hello, how are you feeling today?" in formatted
    assert "Unknown: Not great, I have a headache." in formatted


# ============================================================================
# Test detect_phi_in_dialogue
# ============================================================================


@pytest.mark.asyncio
async def test_detect_phi_in_dialogue_basic(medical_service, mock_lm_studio_client):
    """Test PHI detection in speaker-attributed dialogue."""
    # Mock LLM response
    mock_response = """{
        "phi_detected": true,
        "entities": [
            {
                "type": "name",
                "text": "John Doe",
                "speaker_role": "patient",
                "start": 45,
                "end": 53,
                "confidence": 0.95
            }
        ]
    }"""
    mock_lm_studio_client.chat_completion.return_value = mock_response

    result = await medical_service.detect_phi_in_dialogue(SAMPLE_DIALOGUE_DATA)

    # Verify result structure
    assert result["phi_detected"] is True
    assert len(result["entities"]) == 1
    assert result["entities"][0]["type"] == "name"
    assert result["entities"][0]["speaker_role"] == "patient"

    # Verify LLM was called with speaker-attributed dialogue
    call_args = mock_lm_studio_client.chat_completion.call_args
    messages = call_args[1]["messages"]
    assert any("Doctor/Patient" in msg["content"] for msg in messages)


@pytest.mark.asyncio
async def test_detect_phi_in_dialogue_no_phi(medical_service, mock_lm_studio_client):
    """Test PHI detection when no PHI present."""
    mock_response = '{"phi_detected": false, "entities": []}'
    mock_lm_studio_client.chat_completion.return_value = mock_response

    result = await medical_service.detect_phi_in_dialogue(SAMPLE_DIALOGUE_DATA)

    assert result["phi_detected"] is False
    assert len(result["entities"]) == 0


@pytest.mark.asyncio
async def test_detect_phi_in_dialogue_error_handling(medical_service, mock_lm_studio_client):
    """Test error handling in PHI detection."""
    mock_lm_studio_client.chat_completion.side_effect = Exception("LLM error")

    result = await medical_service.detect_phi_in_dialogue(SAMPLE_DIALOGUE_DATA)

    assert result["phi_detected"] is False
    assert "error" in result
    assert result["error"] == "LLM error"


# ============================================================================
# Test extract_entities_with_speaker
# ============================================================================


@pytest.mark.asyncio
async def test_extract_entities_with_speaker_basic(medical_service, mock_lm_studio_client):
    """Test entity extraction with speaker attribution."""
    mock_response = """{
        "entities": [
            {
                "type": "symptom",
                "text": "chest pain",
                "speaker_role": "patient",
                "normalized": "chest pain",
                "code": "R07.9",
                "confidence": 0.95,
                "details": "Patient-reported symptom"
            },
            {
                "type": "diagnosis",
                "text": "possible angina",
                "speaker_role": "doctor",
                "normalized": "angina pectoris",
                "code": "I20.9",
                "confidence": 0.85,
                "details": "Doctor's clinical impression"
            }
        ]
    }"""
    mock_lm_studio_client.chat_completion.return_value = mock_response

    entities = await medical_service.extract_entities_with_speaker(SAMPLE_DIALOGUE_DATA)

    assert len(entities) == 2
    assert entities[0]["type"] == "symptom"
    assert entities[0]["speaker_role"] == "patient"
    assert entities[1]["type"] == "diagnosis"
    assert entities[1]["speaker_role"] == "doctor"


@pytest.mark.asyncio
async def test_extract_entities_with_speaker_no_entities(medical_service, mock_lm_studio_client):
    """Test entity extraction when no entities found."""
    mock_response = '{"entities": []}'
    mock_lm_studio_client.chat_completion.return_value = mock_response

    entities = await medical_service.extract_entities_with_speaker(SAMPLE_DIALOGUE_DATA)

    assert len(entities) == 0


@pytest.mark.asyncio
async def test_extract_entities_with_speaker_error(medical_service, mock_lm_studio_client):
    """Test error handling in entity extraction."""
    mock_lm_studio_client.chat_completion.side_effect = Exception("Extraction failed")

    entities = await medical_service.extract_entities_with_speaker(SAMPLE_DIALOGUE_DATA)

    assert entities == []


# ============================================================================
# Test generate_soap_from_dialogue
# ============================================================================


@pytest.mark.asyncio
async def test_generate_soap_from_dialogue_basic(medical_service, mock_lm_studio_client):
    """Test SOAP note generation from speaker-attributed dialogue."""
    mock_response = """
**Subjective**
Patient reports chest pain for the past week. Pain is sharp and located on the left side.

**Objective**
Patient appears in mild distress. Vital signs pending.

**Assessment**
Chest pain, likely musculoskeletal in origin. Rule out cardiac cause.

**Plan**
Order ECG and cardiac enzymes. Prescribe analgesics. Follow-up in 48 hours.
"""
    mock_lm_studio_client.chat_completion.return_value = mock_response

    soap_note = await medical_service.generate_soap_from_dialogue(SAMPLE_DIALOGUE_DATA)

    assert "subjective" in soap_note
    assert "objective" in soap_note
    assert "assessment" in soap_note
    assert "plan" in soap_note
    assert "chest pain" in soap_note["subjective"].lower()


@pytest.mark.asyncio
async def test_generate_soap_from_dialogue_with_speaker_context(medical_service, mock_lm_studio_client):
    """Test that speaker mapping is included in the prompt context."""
    mock_response = "**Subjective**\nTest\n**Objective**\nTest\n**Assessment**\nTest\n**Plan**\nTest"
    mock_lm_studio_client.chat_completion.return_value = mock_response

    await medical_service.generate_soap_from_dialogue(SAMPLE_DIALOGUE_DATA)

    # Verify speaker mapping was included in prompt
    call_args = mock_lm_studio_client.chat_completion.call_args
    messages = call_args[1]["messages"]
    user_message = next(msg["content"] for msg in messages if msg["role"] == "user")

    assert "SPEAKER_00" in user_message
    assert "doctor" in user_message.lower()
    assert "confidence" in user_message.lower()


@pytest.mark.asyncio
async def test_generate_soap_from_dialogue_error(medical_service, mock_lm_studio_client):
    """Test error handling in SOAP generation."""
    mock_lm_studio_client.chat_completion.side_effect = Exception("SOAP generation failed")

    soap_note = await medical_service.generate_soap_from_dialogue(SAMPLE_DIALOGUE_DATA)

    assert "error" in soap_note
    assert soap_note["subjective"] == ""
    assert soap_note["objective"] == ""


# ============================================================================
# Test process_speaker_dialogue (Full Pipeline)
# ============================================================================


@pytest.mark.asyncio
async def test_process_speaker_dialogue_full_pipeline(medical_service, mock_lm_studio_client):
    """Test complete speaker-dialogue processing pipeline."""
    # Setup mock responses for all three steps
    phi_response = '{"phi_detected": false, "entities": []}'
    entities_response = '{"entities": [{"type": "symptom", "text": "chest pain", "speaker_role": "patient"}]}'
    soap_response = "**Subjective**\nChest pain\n**Objective**\nTest\n**Assessment**\nTest\n**Plan**\nTest"

    mock_lm_studio_client.chat_completion.side_effect = [phi_response, entities_response, soap_response]

    result = await medical_service.process_speaker_dialogue(SAMPLE_DIALOGUE_DATA)

    # Verify structure
    assert result["has_speaker_attribution"] is True
    assert "speaker_mapping" in result
    assert "statistics" in result
    assert "metadata" in result
    assert "phi_detection" in result
    assert "entities" in result
    assert "soap_note" in result

    # Verify content
    assert result["speaker_mapping"] == SAMPLE_DIALOGUE_DATA["speaker_mapping"]
    assert len(result["entities"]) == 1
    assert result["soap_note"]["subjective"] == "Chest pain"


@pytest.mark.asyncio
async def test_process_speaker_dialogue_partial_failure(medical_service, mock_lm_studio_client):
    """Test that pipeline continues even if one step fails."""
    # First call (PHI) fails, second (entities) succeeds, third (SOAP) succeeds
    mock_lm_studio_client.chat_completion.side_effect = [
        Exception("PHI failed"),
        '{"entities": []}',
        "**Subjective**\nTest\n**Objective**\nTest\n**Assessment**\nTest\n**Plan**\nTest",
    ]

    result = await medical_service.process_speaker_dialogue(SAMPLE_DIALOGUE_DATA)

    # PHI detection should have error
    assert "error" in result["phi_detection"]
    assert result["phi_detection"]["phi_detected"] is False

    # Other steps should succeed
    assert "entities" in result
    assert "soap_note" in result


@pytest.mark.asyncio
async def test_process_speaker_dialogue_with_missing_roles(medical_service, mock_lm_studio_client):
    """Test processing dialogue when speaker roles are unknown."""
    mock_lm_studio_client.chat_completion.side_effect = [
        '{"phi_detected": false, "entities": []}',
        '{"entities": []}',
        "**Subjective**\nTest\n**Objective**\nTest\n**Assessment**\nTest\n**Plan**\nTest",
    ]

    result = await medical_service.process_speaker_dialogue(SAMPLE_DIALOGUE_NO_ROLES)

    # Should complete successfully even with unknown roles
    assert result["has_speaker_attribution"] is True
    assert result["speaker_mapping"]["SPEAKER_00"]["role"] == "unknown"
    assert "soap_note" in result


# ============================================================================
# Integration-style Tests
# ============================================================================


@pytest.mark.asyncio
async def test_speaker_aware_vs_plain_text_prompts(medical_service, mock_lm_studio_client):
    """Test that speaker-aware prompts are different from plain text prompts."""
    mock_lm_studio_client.chat_completion.return_value = (
        "**Subjective**\nTest\n**Objective**\nTest\n**Assessment**\nTest\n**Plan**\nTest"
    )

    # Call speaker-aware method
    await medical_service.generate_soap_from_dialogue(SAMPLE_DIALOGUE_DATA)
    speaker_aware_call = mock_lm_studio_client.chat_completion.call_args

    # Call plain text method
    mock_lm_studio_client.reset_mock()
    await medical_service.generate_soap_note("Patient has chest pain. Doctor prescribes medication.")
    plain_text_call = mock_lm_studio_client.chat_completion.call_args

    # Verify prompts are different
    speaker_aware_prompt = speaker_aware_call[1]["messages"][0]["content"]
    plain_text_prompt = plain_text_call[1]["messages"][0]["content"]

    assert "doctor-patient dialogue" in speaker_aware_prompt.lower()
    assert "speaker roles" in speaker_aware_prompt.lower() or "doctor:" in speaker_aware_prompt.lower()
    assert "doctor-patient dialogue" not in plain_text_prompt.lower()
