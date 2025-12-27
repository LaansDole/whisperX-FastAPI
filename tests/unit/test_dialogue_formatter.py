"""Unit tests for DialogueFormatter service."""

import pytest
from app.services.dialogue_formatter import DialogueFormatter


@pytest.fixture
def formatter():
    """Fixture for DialogueFormatter instance."""
    return DialogueFormatter()


@pytest.fixture
def sample_parsed_data():
    """Sample parsed WhisperX data."""
    return {
        "segments": [
            {
                "segment_id": 0,
                "text": "What brings you in today?",
                "start_time": 0.0,
                "end_time": 2.0,
                "duration": 2.0,
                "speaker_id": "SPEAKER_00",
                "has_speaker": True,
                "words": [],
                "word_count": 4,
                "avg_confidence": 0.95,
            },
            {
                "segment_id": 1,
                "text": "I've been having chest pain.",
                "start_time": 2.5,
                "end_time": 5.0,
                "duration": 2.5,
                "speaker_id": "SPEAKER_01",
                "has_speaker": True,
                "words": [],
                "word_count": 5,
                "avg_confidence": 0.92,
            },
            {
                "segment_id": 2,
                "text": "How long has this been going on?",
                "start_time": 5.5,
                "end_time": 7.5,
                "duration": 2.0,
                "speaker_id": "SPEAKER_00",
                "has_speaker": True,
                "words": [],
                "word_count": 6,
                "avg_confidence": 0.94,
            },
        ],
        "metadata": {
            "total_segments": 3,
            "total_words": 15,
            "total_duration": 7.5,
            "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_count": 2,
            "has_speaker_labels": True,
            "avg_confidence": 0.937,
        },
    }


@pytest.fixture
def sample_speaker_mapping():
    """Sample speaker role mapping."""
    return {
        "SPEAKER_00": {
            "role": "doctor",
            "confidence": 0.85,
            "method": "heuristic",
            "evidence": {"provider_score": 8.5, "patient_score": 2.0, "medical_terms": 5, "questions_asked": 3},
        },
        "SPEAKER_01": {
            "role": "patient",
            "confidence": 0.80,
            "method": "heuristic",
            "evidence": {"provider_score": 1.0, "patient_score": 7.5, "medical_terms": 0, "patient_patterns": 5},
        },
    }


class TestDialogueFormatter:
    """Tests for DialogueFormatter class."""

    def test_initialization(self, formatter):
        """Test formatter initializes correctly."""
        assert formatter is not None
        assert hasattr(formatter, "logger")

    def test_format_dialogue(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test dialogue formatting."""
        result = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)

        # Check structure
        assert "consultation_metadata" in result
        assert "speaker_mapping" in result
        assert "dialogue" in result
        assert "statistics" in result

        # Check dialogue segments
        dialogue = result["dialogue"]
        assert len(dialogue) == 3

        # Check first segment
        seg0 = dialogue[0]
        assert seg0["speaker_id"] == "SPEAKER_00"
        assert seg0["speaker_role"] == "doctor"
        assert seg0["text"] == "What brings you in today?"
        assert seg0["start_time"] == 0.0
        assert seg0["confidence"] == 0.95

        # Check second segment
        seg1 = dialogue[1]
        assert seg1["speaker_role"] == "patient"
        assert "chest pain" in seg1["text"]

    def test_generate_transcript_plain(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test plain text transcript generation."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        transcript = formatter.generate_transcript(dialogue_data["dialogue"], format="plain")

        assert "Doctor: What brings you in today?" in transcript
        assert "Patient: I've been having chest pain." in transcript
        assert isinstance(transcript, str)
        assert len(transcript) > 0

    def test_generate_transcript_markdown(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test markdown transcript generation."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        transcript = formatter.generate_transcript(dialogue_data["dialogue"], format="markdown")

        assert "**Doctor:**" in transcript
        assert "**Patient:**" in transcript
        assert isinstance(transcript, str)

    def test_generate_transcript_json(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test JSON transcript generation."""
        import json

        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        transcript = formatter.generate_transcript(dialogue_data["dialogue"], format="json")

        # Should be valid JSON
        parsed = json.loads(transcript)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_generate_transcript_with_timestamps(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test transcript with timestamps."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        transcript = formatter.generate_transcript(dialogue_data["dialogue"], format="plain", include_timestamps=True)

        assert "[0.00s - 2.00s]" in transcript
        assert "[2.50s - 5.00s]" in transcript

    def test_generate_transcript_with_confidence(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test transcript with confidence scores."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        transcript = formatter.generate_transcript(dialogue_data["dialogue"], format="plain", include_confidence=True)

        assert "conf:" in transcript
        assert "0.95" in transcript or "0.92" in transcript

    def test_calculate_statistics(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test statistics calculation."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        stats = dialogue_data["statistics"]

        # Check structure
        assert "by_speaker" in stats
        assert "totals" in stats

        # Check per-speaker stats
        by_speaker = stats["by_speaker"]
        assert "SPEAKER_00" in by_speaker
        assert "SPEAKER_01" in by_speaker

        # Check SPEAKER_00 (doctor)
        doc_stats = by_speaker["SPEAKER_00"]
        assert doc_stats["speaker_role"] == "doctor"
        assert doc_stats["segment_count"] == 2
        assert doc_stats["word_count"] == 10  # 4 + 6
        assert doc_stats["speaking_time"] == 4.0  # 2.0 + 2.0

        # Check SPEAKER_01 (patient)
        pat_stats = by_speaker["SPEAKER_01"]
        assert pat_stats["speaker_role"] == "patient"
        assert pat_stats["segment_count"] == 1
        assert pat_stats["word_count"] == 5
        assert pat_stats["speaking_time"] == 2.5

        # Check totals
        totals = stats["totals"]
        assert totals["total_segments"] == 3
        assert totals["total_words"] == 15
        assert totals["total_speaking_time"] == 6.5

    def test_generate_speaker_summary(self, formatter, sample_speaker_mapping):
        """Test speaker summary generation."""
        stats = {
            "by_speaker": {
                "SPEAKER_00": {"segment_count": 8, "speaking_time": 60.5},
                "SPEAKER_01": {"segment_count": 7, "speaking_time": 55.3},
            },
            "totals": {"total_speaking_time": 115.8, "total_segments": 15},
        }

        summary = formatter.generate_speaker_summary(sample_speaker_mapping, stats)

        assert "SPEAKER_00 (Doctor)" in summary
        assert "SPEAKER_01 (Patient)" in summary
        assert "0.85" in summary  # doctor confidence
        assert "0.80" in summary  # patient confidence
        assert "Total Speaking Time" in summary

    def test_format_for_llm_prompt(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test LLM prompt formatting."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        llm_prompt = formatter.format_for_llm_prompt(dialogue_data["dialogue"])

        assert "Doctor: What brings you in today?" in llm_prompt
        assert "Patient: I've been having chest pain." in llm_prompt
        # Should be clean format for LLM
        assert "[" not in llm_prompt  # No timestamps
        assert "conf:" not in llm_prompt  # No confidence scores

    def test_format_for_llm_prompt_with_max_segments(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test LLM prompt with segment limit."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        llm_prompt = formatter.format_for_llm_prompt(dialogue_data["dialogue"], max_segments=2)

        # Should only have first 2 segments
        lines = [line for line in llm_prompt.split("\n") if line.strip()]
        assert len(lines) == 2

    def test_extract_role_specific_content(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test extraction of role-specific content."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)

        # Extract doctor content
        doctor_content = formatter.extract_role_specific_content(dialogue_data["dialogue"], role="doctor")
        assert len(doctor_content) == 2
        assert "What brings you in today?" in doctor_content
        assert "How long has this been going on?" in doctor_content

        # Extract patient content
        patient_content = formatter.extract_role_specific_content(dialogue_data["dialogue"], role="patient")
        assert len(patient_content) == 1
        assert "I've been having chest pain." in patient_content

    def test_empty_dialogue(self, formatter):
        """Test handling of empty dialogue."""
        stats = formatter.calculate_statistics([])

        assert stats["by_speaker"] == {}
        assert stats["totals"]["total_segments"] == 0
        assert stats["totals"]["total_words"] == 0
        assert stats["totals"]["total_speaking_time"] == 0

    def test_dialogue_chronological_order(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test that dialogue preserves chronological order."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        dialogue = dialogue_data["dialogue"]

        # Check timestamps are in order
        for i in range(len(dialogue) - 1):
            assert dialogue[i]["start_time"] <= dialogue[i + 1]["start_time"]

    def test_speaker_labels_consistency(self, formatter, sample_parsed_data, sample_speaker_mapping):
        """Test speaker label consistency."""
        dialogue_data = formatter.format_dialogue(sample_parsed_data, sample_speaker_mapping)
        dialogue = dialogue_data["dialogue"]

        # All segments should have speaker_id and speaker_role
        for segment in dialogue:
            assert "speaker_id" in segment
            assert "speaker_role" in segment
            assert segment["speaker_role"] in ["doctor", "patient", "unknown"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
