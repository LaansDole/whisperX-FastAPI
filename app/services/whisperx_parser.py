"""WhisperX result parser for transforming transcription JSON into medical-ready format.

This module parses the WhisperX transcription result schema (documented in ADR-005)
and extracts relevant fields for medical processing including speaker attribution,
timing information, and text content.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from ..config import Config

logger = logging.getLogger(__name__)


class WhisperXParseError(Exception):
    """Exception raised when WhisperX result parsing fails."""

    pass


class WhisperXParser:
    """Parser for WhisperX transcription results.

    Parses the standardized WhisperX JSON schema and extracts:
    - Segments with speaker labels and text
    - Word-level timing and confidence data
    - Metadata (duration, speaker count, etc.)

    Schema based on ADR-005: WhisperX Transcription Result JSON Schema
    """

    def __init__(self):
        """Initialize the parser."""
        self.logger = logger

    def parse(self, whisperx_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse WhisperX result into structured format.

        Args:
            whisperx_result: Raw WhisperX transcription result JSON with 'segments'
                           and 'word_segments' arrays

        Returns:
            Parsed data with segments, metadata, and validation status

        Raises:
            WhisperXParseError: If required fields are missing or invalid
        """
        try:
            self._validate_schema(whisperx_result)

            segments = whisperx_result.get("segments", [])
            word_segments = whisperx_result.get("word_segments", [])

            # Extract segments with speaker attribution
            parsed_segments = self._parse_segments(segments)

            # Calculate metadata
            metadata = self._calculate_metadata(parsed_segments, word_segments)

            # Validate parsed data
            self._validate_parsed_data(parsed_segments, metadata)

            return {
                "segments": parsed_segments,
                "metadata": metadata,
                "raw_segment_count": len(segments),
                "raw_word_count": len(word_segments),
                "parsed_at": datetime.now(Config.TIMEZONE).isoformat(),
                "schema_valid": True,
            }

        except Exception as e:
            self.logger.error(f"Failed to parse WhisperX result: {str(e)}")
            raise WhisperXParseError(f"Parsing failed: {str(e)}") from e

    def _validate_schema(self, result: Dict[str, Any]) -> None:
        """Validate that required top-level fields exist.

        Args:
            result: WhisperX result JSON

        Raises:
            WhisperXParseError: If required fields are missing
        """
        if not isinstance(result, dict):
            raise WhisperXParseError("Result must be a dictionary")

        if "segments" not in result:
            raise WhisperXParseError("Missing required field: 'segments'")

        if "word_segments" not in result:
            raise WhisperXParseError("Missing required field: 'word_segments'")

        if not isinstance(result["segments"], list):
            raise WhisperXParseError("'segments' must be a list")

        if not isinstance(result["word_segments"], list):
            raise WhisperXParseError("'word_segments' must be a list")

    def _parse_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse segments array into structured format.

        Args:
            segments: Raw segments array from WhisperX result

        Returns:
            List of parsed segment objects with validated fields
        """
        parsed = []

        for idx, segment in enumerate(segments):
            try:
                parsed_segment = {
                    "segment_id": idx,
                    "text": segment.get("text", "").strip(),
                    "start_time": float(segment.get("start", 0)),
                    "end_time": float(segment.get("end", 0)),
                    "duration": float(segment.get("end", 0)) - float(segment.get("start", 0)),
                    "speaker_id": segment.get("speaker"),  # May be None
                    "has_speaker": "speaker" in segment and segment["speaker"] is not None,
                    "words": self._parse_words(segment.get("words", [])),
                    "word_count": len(segment.get("words", [])),
                }

                # Calculate average confidence from words
                words = segment.get("words", [])
                if words:
                    scores = [w.get("score", 0) for w in words if "score" in w]
                    parsed_segment["avg_confidence"] = sum(scores) / len(scores) if scores else 0.0
                else:
                    parsed_segment["avg_confidence"] = 0.0

                parsed.append(parsed_segment)

            except Exception as e:
                self.logger.warning(f"Failed to parse segment {idx}: {str(e)}")
                # Continue with other segments
                continue

        return parsed

    def _parse_words(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse words array from a segment.

        Args:
            words: Raw words array from segment

        Returns:
            List of parsed word objects
        """
        parsed = []

        for idx, word in enumerate(words):
            try:
                parsed_word = {
                    "word_id": idx,
                    "text": word.get("word", ""),
                    "start_time": float(word.get("start", 0)),
                    "end_time": float(word.get("end", 0)),
                    "confidence": float(word.get("score", 0)),
                    "speaker_id": word.get("speaker"),  # May be None
                    "duration": float(word.get("end", 0)) - float(word.get("start", 0)),
                }
                parsed.append(parsed_word)
            except Exception as e:
                self.logger.warning(f"Failed to parse word {idx}: {str(e)}")
                continue

        return parsed

    def _calculate_metadata(
        self, parsed_segments: List[Dict[str, Any]], word_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metadata from parsed segments.

        Args:
            parsed_segments: Parsed segments array
            word_segments: Raw word_segments array for additional stats

        Returns:
            Metadata dictionary with statistics
        """
        if not parsed_segments:
            return {
                "total_segments": 0,
                "total_words": 0,
                "total_duration": 0.0,
                "speakers_detected": [],
                "speaker_count": 0,
                "has_speaker_labels": False,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        # Calculate duration
        start_time = min(s["start_time"] for s in parsed_segments)
        end_time = max(s["end_time"] for s in parsed_segments)
        total_duration = end_time - start_time

        # Collect unique speakers
        speakers = set()
        has_speakers = False
        for segment in parsed_segments:
            if segment["has_speaker"] and segment["speaker_id"]:
                speakers.add(segment["speaker_id"])
                has_speakers = True

        # Calculate confidence stats
        confidences = [s["avg_confidence"] for s in parsed_segments if s["avg_confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0

        # Count total words
        total_words = sum(s["word_count"] for s in parsed_segments)

        return {
            "total_segments": len(parsed_segments),
            "total_words": total_words,
            "total_duration": round(total_duration, 3),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "speakers_detected": sorted(list(speakers)),
            "speaker_count": len(speakers),
            "has_speaker_labels": has_speakers,
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": round(min_confidence, 4),
            "max_confidence": round(max_confidence, 4),
        }

    def _validate_parsed_data(self, segments: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """Validate parsed data for completeness and consistency.

        Args:
            segments: Parsed segments
            metadata: Calculated metadata

        Raises:
            WhisperXParseError: If validation fails
        """
        if not segments:
            raise WhisperXParseError("No segments were successfully parsed")

        if metadata["total_words"] == 0:
            self.logger.warning("No words found in any segment")

        if metadata["total_duration"] <= 0:
            raise WhisperXParseError("Invalid duration: must be positive")

        # Check for speaker label consistency
        if metadata["has_speaker_labels"]:
            segments_with_speakers = sum(1 for s in segments if s["has_speaker"])
            speaker_label_coverage = segments_with_speakers / len(segments)

            # Track coverage in metadata for downstream consumers
            metadata["speaker_label_coverage"] = round(speaker_label_coverage, 4)

            if segments_with_speakers < len(segments) * 0.5:
                self.logger.warning(f"Only {segments_with_speakers}/{len(segments)} segments have speaker labels")
                # Flag partial speaker label coverage
                metadata["speaker_labels_partial"] = True
            else:
                metadata["speaker_labels_partial"] = False

    def get_full_transcript(self, parsed_data: Dict[str, Any], include_speakers: bool = True) -> str:
        """Generate full transcript text from parsed data.

        Args:
            parsed_data: Parsed WhisperX result from parse()
            include_speakers: Whether to prepend speaker labels to text

        Returns:
            Full transcript as continuous text
        """
        segments = parsed_data.get("segments", [])
        lines = []

        for segment in segments:
            text = segment["text"]
            if include_speakers and segment["has_speaker"]:
                speaker = segment["speaker_id"]
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)

        return "\n".join(lines)

    def get_speaker_statistics(self, parsed_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics per speaker.

        Args:
            parsed_data: Parsed WhisperX result from parse()

        Returns:
            Dictionary mapping speaker_id to statistics
        """
        segments = parsed_data.get("segments", [])
        speaker_stats = {}

        for segment in segments:
            if not segment["has_speaker"]:
                continue

            speaker_id = segment["speaker_id"]
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "segment_count": 0,
                    "word_count": 0,
                    "speaking_time": 0.0,
                    "avg_confidence": 0.0,
                    "segments": [],
                }

            stats = speaker_stats[speaker_id]
            stats["segment_count"] += 1
            stats["word_count"] += segment["word_count"]
            stats["speaking_time"] += segment["duration"]
            stats["segments"].append(segment["segment_id"])

        # Calculate average confidence per speaker
        for speaker_id in speaker_stats:
            speaker_segments = [s for s in segments if s.get("speaker_id") == speaker_id]
            confidences = [s["avg_confidence"] for s in speaker_segments if s["avg_confidence"] > 0]
            speaker_stats[speaker_id]["avg_confidence"] = (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            )
            speaker_stats[speaker_id]["speaking_time"] = round(speaker_stats[speaker_id]["speaking_time"], 3)

        return speaker_stats
