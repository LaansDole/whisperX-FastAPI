"""Dialogue formatting service for medical consultations.

Converts parsed WhisperX segments with speaker attribution into
structured medical dialogue formats suitable for LLM processing.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DialogueFormatter:
    """Formats parsed WhisperX data into medical dialogue structures."""
    
    def __init__(self):
        """Initialize the dialogue formatter."""
        self.logger = logger
    
    def format_dialogue(
        self, 
        parsed_data: Dict[str, Any], 
        speaker_mapping: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert parsed segments into structured dialogue with speaker roles.
        
        Args:
            parsed_data: Parsed WhisperX result from WhisperXParser
            speaker_mapping: Speaker role mapping from SpeakerIdentifier
            
        Returns:
            Structured dialogue data with:
            - dialogue: List of attributed dialogue segments
            - speaker_mapping: Enhanced speaker information
            - statistics: Speaking time and segment counts
            - metadata: Consultation metadata
        """
        segments = parsed_data.get("segments", [])
        metadata = parsed_data.get("metadata", {})
        
        # Build dialogue segments with speaker roles
        dialogue = []
        for segment in segments:
            speaker_id = segment.get("speaker_id")
            
            # Get speaker role information
            speaker_info = speaker_mapping.get(speaker_id, {})
            speaker_role = speaker_info.get("role", "unknown")
            
            dialogue_segment = {
                "speaker_id": speaker_id,
                "speaker_role": speaker_role,
                "text": segment.get("text", "").strip(),
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "duration": segment.get("duration"),
                "confidence": segment.get("avg_confidence"),
                "word_count": segment.get("word_count"),
                "segment_id": segment.get("segment_id")
            }
            
            dialogue.append(dialogue_segment)
        
        # Calculate speaking statistics
        statistics = self.calculate_statistics(dialogue)
        
        # Build complete dialogue structure
        return {
            "consultation_metadata": {
                "total_duration": metadata.get("total_duration"),
                "total_speakers": metadata.get("speaker_count"),
                "total_segments": len(dialogue),
                "total_words": metadata.get("total_words"),
                "avg_confidence": metadata.get("avg_confidence"),
                "has_speaker_labels": metadata.get("has_speaker_labels"),
                "speaker_label_coverage": metadata.get("speaker_label_coverage"),
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "speaker_mapping": speaker_mapping,
            "dialogue": dialogue,
            "statistics": statistics
        }
    
    def generate_transcript(
        self, 
        dialogue: List[Dict[str, Any]], 
        format: Literal["plain", "markdown", "json"] = "plain",
        include_timestamps: bool = False,
        include_confidence: bool = False
    ) -> str:
        """Generate formatted transcript from dialogue.
        
        Args:
            dialogue: List of dialogue segments
            format: Output format (plain, markdown, json)
            include_timestamps: Include timestamps in output
            include_confidence: Include confidence scores
            
        Returns:
            Formatted transcript string
        """
        if format == "json":
            import json
            return json.dumps(dialogue, indent=2)
        
        lines = []
        
        for segment in dialogue:
            speaker_role = segment.get("speaker_role", "unknown").title()
            text = segment.get("text", "")
            
            # Build line based on format
            if format == "markdown":
                line = f"**{speaker_role}:** {text}"
            else:  # plain
                line = f"{speaker_role}: {text}"
            
            # Add optional metadata
            metadata_parts = []
            if include_timestamps:
                start = segment.get("start_time", 0)
                end = segment.get("end_time", 0)
                metadata_parts.append(f"[{start:.2f}s - {end:.2f}s]")
            
            if include_confidence:
                conf = segment.get("confidence", 0)
                metadata_parts.append(f"(conf: {conf:.2f})")
            
            if metadata_parts:
                metadata_str = " ".join(metadata_parts)
                if format == "markdown":
                    line += f" *{metadata_str}*"
                else:
                    line += f" {metadata_str}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def calculate_statistics(
        self, 
        dialogue: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate speaking time and segment statistics per speaker.
        
        Args:
            dialogue: List of dialogue segments
            
        Returns:
            Dictionary with per-speaker statistics and totals
        """
        speaker_stats = {}
        
        for segment in dialogue:
            speaker_id = segment.get("speaker_id")
            speaker_role = segment.get("speaker_role", "unknown")
            
            if not speaker_id:
                continue
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "speaker_id": speaker_id,
                    "speaker_role": speaker_role,
                    "segment_count": 0,
                    "word_count": 0,
                    "speaking_time": 0.0,
                    "avg_confidence": 0.0,
                    "total_confidence": 0.0,
                    "segments_with_confidence": 0
                }
            
            stats = speaker_stats[speaker_id]
            stats["segment_count"] += 1
            stats["word_count"] += segment.get("word_count", 0)
            stats["speaking_time"] += segment.get("duration", 0.0)
            
            # Track confidence
            conf = segment.get("confidence", 0)
            if conf > 0:
                stats["total_confidence"] += conf
                stats["segments_with_confidence"] += 1
        
        # Calculate averages
        for speaker_id, stats in speaker_stats.items():
            if stats["segments_with_confidence"] > 0:
                stats["avg_confidence"] = round(
                    stats["total_confidence"] / stats["segments_with_confidence"], 
                    4
                )
            else:
                stats["avg_confidence"] = 0.0
            
            # Remove temporary fields
            del stats["total_confidence"]
            del stats["segments_with_confidence"]
            
            # Round values
            stats["speaking_time"] = round(stats["speaking_time"], 2)
        
        # Calculate totals
        total_speaking_time = sum(s["speaking_time"] for s in speaker_stats.values())
        total_segments = sum(s["segment_count"] for s in speaker_stats.values())
        total_words = sum(s["word_count"] for s in speaker_stats.values())
        
        return {
            "by_speaker": speaker_stats,
            "totals": {
                "total_speaking_time": round(total_speaking_time, 2),
                "total_segments": total_segments,
                "total_words": total_words
            }
        }
    
    def generate_speaker_summary(
        self, 
        speaker_mapping: Dict[str, Dict[str, Any]],
        statistics: Dict[str, Any]
    ) -> str:
        """Generate human-readable speaker summary.
        
        Args:
            speaker_mapping: Speaker role mapping
            statistics: Speaking statistics
            
        Returns:
            Formatted speaker summary string
        """
        lines = ["Speaker Summary:", ""]
        
        by_speaker = statistics.get("by_speaker", {})
        
        for speaker_id, speaker_info in speaker_mapping.items():
            role = speaker_info.get("role", "unknown").title()
            confidence = speaker_info.get("confidence", 0)
            method = speaker_info.get("method", "unknown")
            
            stats = by_speaker.get(speaker_id, {})
            segments = stats.get("segment_count", 0)
            speaking_time = stats.get("speaking_time", 0.0)
            
            lines.append(f"{speaker_id} ({role}):")
            lines.append(f"  Role Confidence: {confidence:.2f} ({method})")
            lines.append(f"  Segments: {segments}")
            lines.append(f"  Speaking Time: {speaking_time:.1f}s")
            lines.append("")
        
        totals = statistics.get("totals", {})
        lines.append(f"Total Speaking Time: {totals.get('total_speaking_time', 0):.1f}s")
        lines.append(f"Total Segments: {totals.get('total_segments', 0)}")
        
        return "\n".join(lines)
    
    def format_for_llm_prompt(
        self, 
        dialogue: List[Dict[str, Any]],
        max_segments: Optional[int] = None
    ) -> str:
        """Format dialogue optimized for LLM prompt consumption.
        
        Args:
            dialogue: List of dialogue segments
            max_segments: Maximum segments to include (for context limits)
            
        Returns:
            LLM-optimized dialogue string
        """
        if max_segments:
            dialogue = dialogue[:max_segments]
        
        lines = []
        for segment in dialogue:
            role = segment.get("speaker_role", "unknown").title()
            text = segment.get("text", "").strip()
            
            # Clean and format for LLM
            if text:
                lines.append(f"{role}: {text}")
        
        return "\n".join(lines)
    
    def extract_role_specific_content(
        self, 
        dialogue: List[Dict[str, Any]],
        role: Literal["doctor", "patient", "unknown"]
    ) -> List[str]:
        """Extract all text spoken by a specific role.
        
        Args:
            dialogue: List of dialogue segments
            role: Speaker role to filter by
            
        Returns:
            List of text segments from specified role
        """
        content = []
        
        for segment in dialogue:
            if segment.get("speaker_role") == role:
                text = segment.get("text", "").strip()
                if text:
                    content.append(text)
        
        return content
