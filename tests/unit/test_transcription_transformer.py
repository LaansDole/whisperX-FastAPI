"""Unit tests for TranscriptionTransformer service."""

import pytest
from app.services.transcription_transformer import TranscriptionTransformer, TranscriptionTransformError


@pytest.fixture
def transformer():
    """Fixture for TranscriptionTransformer instance."""
    return TranscriptionTransformer()


@pytest.fixture
def sample_whisperx_result():
    """Sample WhisperX result JSON."""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "What brings you in today?",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "What", "start": 0.0, "end": 0.3, "score": 0.95, "speaker": "SPEAKER_00"},
                    {"word": "brings", "start": 0.4, "end": 0.7, "score": 0.93, "speaker": "SPEAKER_00"},
                ],
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "I've been having chest pain.",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "I've", "start": 2.5, "end": 2.7, "score": 0.92, "speaker": "SPEAKER_01"},
                    {"word": "been", "start": 2.8, "end": 3.0, "score": 0.91, "speaker": "SPEAKER_01"},
                ],
            },
        ],
        "word_segments": [],
    }


class TestTranscriptionTransformer:
    """Tests for TranscriptionTransformer class."""

    def test_initialization(self, transformer):
        """Test transformer initializes correctly."""
        assert transformer is not None
        assert hasattr(transformer, "parser")
        assert hasattr(transformer, "identifier")
        assert hasattr(transformer, "formatter")
        assert hasattr(transformer, "logger")

    def test_transform_complete_pipeline(self, transformer, sample_whisperx_result):
        """Test complete transformation pipeline."""
        result = transformer.transform(sample_whisperx_result)

        # Check required fields
        assert "transformation_metadata" in result
        assert "consultation_metadata" in result
        assert "speaker_mapping" in result
        assert "dialogue" in result
        assert "full_transcript" in result
        assert "full_transcript_markdown" in result
        assert "statistics" in result
        assert "parsed_data" in result

        # Check dialogue
        dialogue = result["dialogue"]
        assert len(dialogue) == 2
        assert dialogue[0]["speaker_role"] in ["doctor", "patient"]

        # Check speaker mapping
        speaker_mapping = result["speaker_mapping"]
        assert len(speaker_mapping) == 2
        assert "SPEAKER_00" in speaker_mapping
        assert "SPEAKER_01" in speaker_mapping

    def test_transform_with_workflow_id(self, transformer, sample_whisperx_result):
        """Test transformation with workflow ID."""
        workflow_id = "test-workflow-123"
        result = transformer.transform(sample_whisperx_result, workflow_id=workflow_id)

        assert result["transformation_metadata"]["workflow_id"] == workflow_id

    def test_transform_with_consultation_metadata(self, transformer, sample_whisperx_result):
        """Test transformation with consultation metadata."""
        consultation_meta = {"date": "2025-12-27", "department": "cardiology", "provider": "Dr. Smith"}

        result = transformer.transform(sample_whisperx_result, consultation_metadata=consultation_meta)

        consult_meta = result["consultation_metadata"]
        assert consult_meta["date"] == "2025-12-27"
        assert consult_meta["department"] == "cardiology"
        assert consult_meta["provider"] == "Dr. Smith"

    def test_transform_invalid_input(self, transformer):
        """Test transformation with invalid input."""
        with pytest.raises(TranscriptionTransformError):
            transformer.transform({"invalid": "data"})

    def test_transform_empty_segments(self, transformer):
        """Test transformation with empty segments."""
        empty_result = {"segments": [], "word_segments": []}

        with pytest.raises(TranscriptionTransformError):
            transformer.transform(empty_result)

    def test_transform_with_overrides(self, transformer, sample_whisperx_result):
        """Test transformation with manual speaker role overrides."""
        manual_mapping = {
            "SPEAKER_00": "patient",  # Override: swap roles
            "SPEAKER_01": "doctor",
        }

        result = transformer.transform_with_overrides(sample_whisperx_result, manual_mapping)

        # Check roles were overridden
        speaker_mapping = result["speaker_mapping"]
        assert speaker_mapping["SPEAKER_00"]["role"] == "patient"
        assert speaker_mapping["SPEAKER_01"]["role"] == "doctor"
        assert speaker_mapping["SPEAKER_00"]["method"] == "manual_override"

        # Check dialogue was updated
        dialogue = result["dialogue"]
        for segment in dialogue:
            if segment["speaker_id"] == "SPEAKER_00":
                assert segment["speaker_role"] == "patient"
            elif segment["speaker_id"] == "SPEAKER_01":
                assert segment["speaker_role"] == "doctor"

        # Check metadata flag
        assert result["transformation_metadata"]["manual_overrides_applied"] is True

    def test_validate_transformation_valid(self, transformer, sample_whisperx_result):
        """Test validation of valid transformed data."""
        result = transformer.transform(sample_whisperx_result)
        validation = transformer.validate_transformation(result)

        assert validation["valid"] is True
        assert validation["status"] in ["valid", "valid_with_warnings"]
        assert len(validation["issues"]) == 0
        assert "summary" in validation

    def test_validate_transformation_missing_fields(self, transformer):
        """Test validation with missing required fields."""
        incomplete_data = {
            "dialogue": [],
            # Missing speaker_mapping, statistics, full_transcript
        }

        validation = transformer.validate_transformation(incomplete_data)

        assert validation["valid"] is False
        assert validation["status"] == "invalid"
        assert len(validation["issues"]) > 0
        assert any("speaker_mapping" in issue for issue in validation["issues"])

    def test_validate_transformation_speaker_inconsistency(self, transformer):
        """Test validation with speaker inconsistencies."""
        inconsistent_data = {
            "dialogue": [{"speaker_id": "SPEAKER_00", "text": "Hello"}, {"speaker_id": "SPEAKER_01", "text": "Hi"}],
            "speaker_mapping": {
                "SPEAKER_00": {"role": "doctor"}
                # Missing SPEAKER_01
            },
            "statistics": {"by_speaker": {}},
            "full_transcript": "Hello\nHi",
        }

        validation = transformer.validate_transformation(inconsistent_data)

        assert len(validation["issues"]) > 0
        assert any("unmapped" in issue.lower() for issue in validation["issues"])

    def test_transcript_generation(self, transformer, sample_whisperx_result):
        """Test that both plain and markdown transcripts are generated."""
        result = transformer.transform(sample_whisperx_result)

        plain = result["full_transcript"]
        markdown = result["full_transcript_markdown"]

        # Plain transcript
        assert "What brings you in today?" in plain
        assert ":" in plain  # Has speaker labels

        # Markdown transcript
        assert "**" in markdown  # Has markdown formatting
        assert "What brings you in today?" in markdown

    def test_statistics_accuracy(self, transformer, sample_whisperx_result):
        """Test that statistics are accurately calculated."""
        result = transformer.transform(sample_whisperx_result)
        stats = result["statistics"]

        # Check structure
        assert "by_speaker" in stats
        assert "totals" in stats

        # Check totals
        totals = stats["totals"]
        assert totals["total_segments"] == 2
        assert totals["total_speaking_time"] > 0

    def test_performance_typical_consultation(self, transformer, sample_whisperx_result):
        """Test transformation performance for typical consultation."""
        import time

        start = time.time()
        result = transformer.transform(sample_whisperx_result)
        duration = time.time() - start

        # Should complete in under 1 second
        assert duration < 1.0
        assert result is not None

    def test_error_handling_comprehensive_messages(self, transformer):
        """Test that error messages are comprehensive."""
        try:
            transformer.transform(None)
        except TranscriptionTransformError as e:
            # Error message should be descriptive
            assert len(str(e)) > 10
            assert "failed" in str(e).lower() or "error" in str(e).lower()

    def test_empty_result_creation(self, transformer):
        """Test creation of empty result when no speakers detected."""
        # Create WhisperX result with no speakers
        no_speakers = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test", "words": []}], "word_segments": []}

        result = transformer.transform(no_speakers)

        # Should return valid empty result
        assert result["dialogue"] == [] or len(result["speaker_mapping"]) == 0
        assert "transformation_metadata" in result

    def test_consistency_across_formats(self, transformer, sample_whisperx_result):
        """Test consistency between different output formats."""
        result = transformer.transform(sample_whisperx_result)

        dialogue = result["dialogue"]
        plain_transcript = result["full_transcript"]
        markdown_transcript = result["full_transcript_markdown"]

        # All should contain same content
        for segment in dialogue:
            text = segment["text"]
            assert text in plain_transcript or text.strip() in plain_transcript
            assert text in markdown_transcript or text.strip() in markdown_transcript


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
