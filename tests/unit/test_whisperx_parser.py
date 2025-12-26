"""Unit tests for WhisperX Parser service.

Tests individual methods and functionality in isolation using mock data.
"""

import pytest
from app.services.whisperx_parser import WhisperXParser, WhisperXParseError  


# ============================================================================
# Test Fixtures and Sample Data
# ============================================================================

@pytest.fixture
def parser():
    """Fixture for WhisperX parser instance."""
    return WhisperXParser()


@pytest.fixture
def sample_whisperx_result():
    """Sample WhisperX result based on ADR-005 schema."""
    return {
        "segments": [
            {
                "start": 0.031,
                "end": 0.252,
                "text": " Dialogue 4.",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Dialogue", "start": 0.031, "end": 0.191, "score": 0.01, "speaker": "SPEAKER_00"},
                    {"word": "4.", "start": 0.211, "end": 0.252, "score": 0.012, "speaker": "SPEAKER_00"}
                ]
            },
            {
                "start": 0.272,
                "end": 2.557,
                "text": " How is your chest pain?",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "How", "start": 0.272, "end": 0.392, "score": 0.01, "speaker": "SPEAKER_00"},
                    {"word": "is", "start": 0.412, "end": 0.472, "score": 0.01, "speaker": "SPEAKER_00"},
                    {"word": "your", "start": 0.492, "end": 0.612, "score": 0.01, "speaker": "SPEAKER_00"},
                    {"word": "chest", "start": 0.632, "end": 0.773, "score": 0.01, "speaker": "SPEAKER_00"},
                    {"word": "pain?", "start": 0.793, "end": 0.873, "score": 0.011, "speaker": "SPEAKER_00"}
                ]
            },
            {
                "start": 2.6,
                "end": 5.8,
                "text": " It hurts when I breathe deeply.",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "It", "start": 2.6, "end": 2.7, "score": 0.012, "speaker": "SPEAKER_01"},
                    {"word": "hurts", "start": 2.8, "end": 3.0, "score": 0.011, "speaker": "SPEAKER_01"}
                ]
            }
        ],
        "word_segments": [
            {"word": "Dialogue", "start": 0.031, "end": 0.191, "score": 0.01},
            {"word": "4.", "start": 0.211, "end": 0.252, "score": 0.012}
        ]
    }


# ============================================================================
# WhisperX Parser Unit Tests
# ============================================================================

class TestWhisperXParser:
    """Unit tests for WhisperXParser class."""
    
    def test_parse_valid_result(self, parser, sample_whisperx_result):
        """Test parsing a valid WhisperX result."""
        result = parser.parse(sample_whisperx_result)
        
        assert result["schema_valid"] is True
        assert "segments" in result
        assert "metadata" in result
        assert "parsed_at" in result
        assert len(result["segments"]) == 3
    
    def test_parse_segments_structure(self, parser, sample_whisperx_result):
        """Test that parsed segments have correct structure."""
        result = parser.parse(sample_whisperx_result)
        segments = result["segments"]
        
        seg0 = segments[0]
        assert seg0["segment_id"] == 0
        assert "Dialogue 4." in seg0["text"]
        assert seg0["speaker_id"] == "SPEAKER_00"
        assert seg0["has_speaker"] is True
        assert seg0["start_time"] == 0.031
        assert seg0["end_time"] == 0.252
        assert seg0["duration"] > 0
        assert seg0["word_count"] == 2
        assert "avg_confidence" in seg0
    
    def test_parse_words(self, parser, sample_whisperx_result):
        """Test that words are parsed correctly."""
        result = parser.parse(sample_whisperx_result)
        words = result["segments"][0]["words"]
        
        assert len(words) == 2
        assert words[0]["text"] == "Dialogue"
        assert words[0]["confidence"] == 0.01
        assert words[0]["speaker_id"] == "SPEAKER_00"
    
    def test_metadata_calculation(self, parser, sample_whisperx_result):
        """Test metadata calculation."""
        result = parser.parse(sample_whisperx_result)
        metadata = result["metadata"]
        
        assert metadata["total_segments"] == 3
        assert metadata["speaker_count"] == 2
        assert "SPEAKER_00" in metadata["speakers_detected"]
        assert "SPEAKER_01" in metadata["speakers_detected"]
        assert metadata["has_speaker_labels"] is True
        assert metadata["total_duration"] > 0
        assert metadata["avg_confidence"] > 0
    
    def test_speaker_statistics(self, parser, sample_whisperx_result):
        """Test speaker statistics calculation."""
        result = parser.parse(sample_whisperx_result)
        stats = parser.get_speaker_statistics(result)
        
        assert "SPEAKER_00" in stats
        assert "SPEAKER_01" in stats
        assert stats["SPEAKER_00"]["segment_count"] == 2
        assert stats["SPEAKER_00"]["speaking_time"] > 0
        assert stats["SPEAKER_01"]["segment_count"] == 1
    
    def test_transcript_generation(self, parser, sample_whisperx_result):
        """Test transcript generation with and without speakers."""
        result = parser.parse(sample_whisperx_result)
        
        # With speakers
        transcript_with_speakers = parser.get_full_transcript(result, include_speakers=True)
        assert "SPEAKER_00:" in transcript_with_speakers
        assert "SPEAKER_01:" in transcript_with_speakers
        assert "Dialogue 4." in transcript_with_speakers
        
        # Without speakers
        transcript_plain = parser.get_full_transcript(result, include_speakers=False)
        assert "SPEAKER_00:" not in transcript_plain
        assert "Dialogue 4." in transcript_plain
    
    def test_missing_required_fields(self, parser):
        """Test error handling for missing required fields."""
        with pytest.raises(WhisperXParseError, match="segments"):
            parser.parse({"word_segments": []})
        
        with pytest.raises(WhisperXParseError, match="word_segments"):
            parser.parse({"segments": []})
        
        with pytest.raises(WhisperXParseError, match="dictionary"):
            parser.parse("invalid")
    
    def test_empty_segments(self, parser):
        """Test handling of empty segments list."""
        empty_result = {"segments": [], "word_segments": []}
        with pytest.raises(WhisperXParseError, match="No segments"):
            parser.parse(empty_result)
    
    def test_segment_without_speaker(self, parser):
        """Test handling of segments without speaker labels."""
        result_no_speaker = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello", "words": []}],
            "word_segments": []
        }
        
        result = parser.parse(result_no_speaker)
        assert result["schema_valid"] is True
        assert result["metadata"]["has_speaker_labels"] is False
        assert result["segments"][0]["has_speaker"] is False
    
    def test_malformed_segment_handling(self, parser):
        """Test that malformed segments are skipped gracefully."""
        result_with_bad_segment = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Valid segment", "words": []},
                {"start": "invalid", "end": 2.0, "text": "Bad segment", "words": []}
            ],
            "word_segments": []
        }
        
        result = parser.parse(result_with_bad_segment)
        assert result["schema_valid"] is True
        assert len(result["segments"]) == 1
    
    def test_confidence_calculations(self, parser, sample_whisperx_result):
        """Test confidence score calculations."""
        result = parser.parse(sample_whisperx_result)
        metadata = result["metadata"]
        
        assert 0 <= metadata["avg_confidence"] <= 1
        assert 0 <= metadata["min_confidence"] <= 1
        assert 0 <= metadata["max_confidence"] <= 1
        assert metadata["min_confidence"] <= metadata["avg_confidence"] <= metadata["max_confidence"]
    
    def test_duration_calculations(self, parser, sample_whisperx_result):
        """Test duration calculations."""
        result = parser.parse(sample_whisperx_result)
        
        for segment in result["segments"]:
            expected_duration = segment["end_time"] - segment["start_time"]
            assert abs(segment["duration"] - expected_duration) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
