"""Transcription transformation service for medical consultations.

Orchestrates the complete WhisperX-to-Medical dialogue transformation pipeline.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..config import Config

from .whisperx_parser import WhisperXParser, WhisperXParseError
from .speaker_identifier import SpeakerIdentifier
from .dialogue_formatter import DialogueFormatter

logger = logging.getLogger(__name__)


class TranscriptionTransformError(Exception):
    """Exception raised when transcription transformation fails."""

    pass


class TranscriptionTransformer:
    """Orchestrates complete WhisperX to medical dialogue transformation."""

    def __init__(
        self,
        parser: Optional[WhisperXParser] = None,
        identifier: Optional[SpeakerIdentifier] = None,
        formatter: Optional[DialogueFormatter] = None,
    ):
        """Initialize the transformation pipeline.

        Args:
            parser: WhisperX parser instance (creates new if None)
            identifier: Speaker identifier instance (creates new if None)
            formatter: Dialogue formatter instance (creates new if None)
        """
        self.parser = parser or WhisperXParser()
        self.identifier = identifier or SpeakerIdentifier()
        self.formatter = formatter or DialogueFormatter()
        self.logger = logger

    def transform(
        self,
        whisperx_result: Dict[str, Any],
        workflow_id: Optional[str] = None,
        consultation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute complete transformation pipeline.

        Args:
            whisperx_result: Raw WhisperX transcription result JSON
            workflow_id: Optional WhisperX workflow ID for tracking
            consultation_metadata: Optional consultation context

        Returns:
            Complete medical dialogue structure with:
            - dialogue: Speaker-attributed segments
            - speaker_mapping: Role assignments
            - statistics: Speaking time and counts
            - full_transcript: Formatted text transcript
            - metadata: Processing and consultation metadata

        Raises:
            TranscriptionTransformError: If transformation fails at any step
        """
        try:
            self.logger.info("Starting WhisperX to medical dialogue transformation")

            # Step 1: Parse WhisperX result
            self.logger.debug("Step 1: Parsing WhisperX result")
            try:
                parsed_data = self.parser.parse(whisperx_result)
            except WhisperXParseError as e:
                raise TranscriptionTransformError(f"WhisperX parsing failed: {str(e)}") from e

            # Step 2: Identify speaker roles
            self.logger.debug("Step 2: Identifying speaker roles")
            speaker_mapping = self.identifier.identify_roles(parsed_data)

            if not speaker_mapping:
                self.logger.warning("No speakers identified, generating empty dialogue")
                return self._create_empty_result(
                    "No speakers detected in transcription", workflow_id, consultation_metadata
                )

            # Step 3: Format dialogue
            self.logger.debug("Step 3: Formatting dialogue")
            dialogue_data = self.formatter.format_dialogue(parsed_data, speaker_mapping)

            # Step 4: Generate formatted transcripts
            self.logger.debug("Step 4: Generating formatted transcripts")
            dialogue_segments = dialogue_data.get("dialogue", [])

            full_transcript_plain = self.formatter.generate_transcript(dialogue_segments, format="plain")

            full_transcript_markdown = self.formatter.generate_transcript(dialogue_segments, format="markdown")

            # Step 5: Build complete result
            result = {
                "transformation_metadata": {
                    "workflow_id": workflow_id,
                    "transformation_timestamp": datetime.now(Config.TIMEZONE).isoformat(),
                    "parser_version": "1.0",
                    "transformer_version": "1.0",
                },
                "consultation_metadata": {
                    **(consultation_metadata or {}),
                    **dialogue_data.get("consultation_metadata", {}),
                },
                "speaker_mapping": dialogue_data.get("speaker_mapping", {}),
                "dialogue": dialogue_segments,
                "full_transcript": full_transcript_plain,
                "full_transcript_markdown": full_transcript_markdown,
                "statistics": dialogue_data.get("statistics", {}),
                "parsed_data": parsed_data,  # Include for debugging/validation
            }

            self.logger.info(
                f"Transformation complete: {len(dialogue_segments)} segments, {len(speaker_mapping)} speakers"
            )

            return result

        except TranscriptionTransformError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during transformation: {str(e)}")
            raise TranscriptionTransformError(f"Transformation failed: {str(e)}") from e

    def transform_with_overrides(
        self,
        whisperx_result: Dict[str, Any],
        manual_speaker_mapping: Dict[str, str],
        workflow_id: Optional[str] = None,
        consultation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transform with manual speaker role overrides.

        Args:
            whisperx_result: Raw WhisperX transcription result JSON
            manual_speaker_mapping: Manual role assignments {speaker_id: role}
            workflow_id: Optional WhisperX workflow ID
            consultation_metadata: Optional consultation context

        Returns:
            Complete medical dialogue structure with overridden roles
        """
        # First do standard transformation
        result = self.transform(whisperx_result, workflow_id, consultation_metadata)

        # Apply manual overrides
        speaker_mapping = result.get("speaker_mapping", {})
        for speaker_id, role in manual_speaker_mapping.items():
            if speaker_id in speaker_mapping:
                self.logger.info(f"Manually overriding {speaker_id} role to {role}")
                speaker_mapping = self.identifier.override_role(
                    speaker_mapping, speaker_id, role, reason="manual_override_via_api"
                )
            else:
                self.logger.warning(
                    f"Speaker ID '{speaker_id}' in manual_speaker_mapping not found in "
                    f"speaker_mapping. Available speakers: {list(speaker_mapping.keys())}"
                )

        # Update dialogue with new roles
        dialogue = result.get("dialogue", [])
        for segment in dialogue:
            speaker_id = segment.get("speaker_id")
            if speaker_id in speaker_mapping:
                segment["speaker_role"] = speaker_mapping[speaker_id]["role"]

        # Regenerate transcripts with updated roles
        full_transcript_plain = self.formatter.generate_transcript(dialogue, format="plain")
        full_transcript_markdown = self.formatter.generate_transcript(dialogue, format="markdown")

        # Recalculate statistics to reflect updated speaker roles
        updated_statistics = self.formatter.calculate_statistics(dialogue)

        # Update result
        result["speaker_mapping"] = speaker_mapping
        result["dialogue"] = dialogue
        result["full_transcript"] = full_transcript_plain
        result["full_transcript_markdown"] = full_transcript_markdown
        result["statistics"] = updated_statistics
        result["transformation_metadata"]["manual_overrides_applied"] = True

        return result

    def validate_transformation(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transformed data for completeness and consistency.

        Args:
            transformed_data: Output from transform() method

        Returns:
            Validation result with status and any issues found
        """
        issues = []
        warnings = []

        # Check required fields
        required_fields = ["dialogue", "speaker_mapping", "statistics", "full_transcript"]
        for field in required_fields:
            if field not in transformed_data:
                issues.append(f"Missing required field: {field}")

        # Validate dialogue
        dialogue = transformed_data.get("dialogue", [])
        if not dialogue:
            warnings.append("No dialogue segments found")

        # Validate speaker mapping
        speaker_mapping = transformed_data.get("speaker_mapping", {})
        if not speaker_mapping:
            warnings.append("No speakers were mapped to roles")

        # Check speaker consistency
        dialogue_speakers = set(s.get("speaker_id") for s in dialogue if s.get("speaker_id"))
        mapped_speakers = set(speaker_mapping.keys())

        if dialogue_speakers != mapped_speakers:
            unmapped = dialogue_speakers - mapped_speakers
            if unmapped:
                issues.append(f"Dialogue contains unmapped speakers: {unmapped}")

            unused = mapped_speakers - dialogue_speakers
            if unused:
                warnings.append(f"Speaker mapping contains unused speakers: {unused}")

        # Validate statistics
        stats = transformed_data.get("statistics", {})
        if "by_speaker" not in stats:
            warnings.append("Speaker statistics missing")

        # Determine overall status
        if issues:
            status = "invalid"
        elif warnings:
            status = "valid_with_warnings"
        else:
            status = "valid"

        return {
            "status": status,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "summary": {
                "total_segments": len(dialogue),
                "total_speakers": len(speaker_mapping),
                "speaker_roles": {sid: info.get("role") for sid, info in speaker_mapping.items()},
            },
        }

    def _create_empty_result(
        self, reason: str, workflow_id: Optional[str] = None, consultation_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create empty result when transformation cannot proceed.

        Args:
            reason: Reason for empty result
            workflow_id: Optional workflow ID
            consultation_metadata: Optional metadata

        Returns:
            Empty but valid transformation result
        """
        return {
            "transformation_metadata": {
                "workflow_id": workflow_id,
                "transformation_timestamp": datetime.now(Config.TIMEZONE).isoformat(),
                "status": "empty",
                "reason": reason,
            },
            "consultation_metadata": consultation_metadata or {},
            "speaker_mapping": {},
            "dialogue": [],
            "full_transcript": "",
            "full_transcript_markdown": "",
            "statistics": {
                "by_speaker": {},
                "totals": {"total_speaking_time": 0, "total_segments": 0, "total_words": 0},
            },
        }
