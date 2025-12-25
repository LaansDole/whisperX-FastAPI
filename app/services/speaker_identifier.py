"""Speaker role identification service for medical consultations.

This module implements heuristic-based speaker role assignment to differentiate
between healthcare providers (doctors/nurses) and patients in medical transcripts.

Heuristics used:
1. First speaker is typically the healthcare provider (doctor)
2. Medical terminology usage patterns
3. Question patterns (providers ask more questions)
4. Professional language indicators
5. Speaking time ratios
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SpeakerRole(str, Enum):
    """Possible speaker roles in medical consultations."""
    DOCTOR = "doctor"
    PATIENT = "patient"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for speaker role assignments."""
    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # < 0.5


class SpeakerIdentifier:
    """Identifies speaker roles in medical consultation transcripts.
    
    Uses multiple heuristics to assign roles (doctor vs patient) to speakers
    identified by WhisperX diarization. Returns confidence scores for each assignment.
    """
    
    # Medical terminology patterns (indicators of healthcare provider)
    MEDICAL_TERMS = [
        r'\b(diagnosis|prescribed|medication|treatment|symptoms|examination|assess|vitals)\b',
        r'\b(blood pressure|heart rate|temperature|pulse|respiratory)\b',
        r'\b(test results|lab work|imaging|x-ray|MRI|CT scan)\b',
        r'\b(follow-up|referral|consultation|appointment)\b',
    ]
    
    # Question patterns (providers ask more questions)
    QUESTION_PATTERNS = [
        r'\bwhat\s+(brings|symptoms|happened|issues)',
        r'\bhow\s+(long|often|severe|bad)',
        r'\b(have you|are you|do you|did you)',
        r'\bwhen\s+did',
        r'\bany\s+(pain|discomfort|problems|concerns)',
    ]
    
    # Patient symptom/complaint patterns
    PATIENT_PATTERNS = [
        r'\b(I feel|I have|I\'ve been|my)\b',
        r'\b(hurts|painful|ache|sore|uncomfortable)\b',
        r'\b(started|began|since|for the past)\b',
    ]
    
    def __init__(self):
        """Initialize the speaker identifier."""
        self.logger = logger
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.medical_regex = re.compile('|'.join(self.MEDICAL_TERMS), re.IGNORECASE)
        self.question_regex = re.compile('|'.join(self.QUESTION_PATTERNS), re.IGNORECASE)
        self.patient_regex = re.compile('|'.join(self.PATIENT_PATTERNS), re.IGNORECASE)
    
    def identify_roles(
        self, 
        parsed_data: Dict[str, Any],
        method: str = "heuristic"
    ) -> Dict[str, Dict[str, Any]]:
        """Identify speaker roles from parsed WhisperX data.
        
        Args:
            parsed_data: Parsed WhisperX result from WhisperXParser
            method: Identification method ("heuristic", "ml", "manual")
                   Currently only "heuristic" is implemented
        
        Returns:
            Dictionary mapping speaker_id to role information:
            {
                "SPEAKER_00": {
                    "role": "doctor",
                    "confidence": 0.85,
                    "confidence_level": "high",
                    "method": "heuristic",
                    "evidence": {
                        "medical_terms": 15,
                        "questions_asked": 8,
                        "first_speaker": True
                    }
                }
            }
        """
        if method != "heuristic":
            raise NotImplementedError(f"Method '{method}' not implemented. Use 'heuristic'.")
        
        metadata = parsed_data.get("metadata", {})
        speakers = metadata.get("speakers_detected", [])
        
        if not speakers:
            self.logger.warning("No speakers detected in parsed data")
            return {}
        
        if len(speakers) == 1:
            return self._handle_single_speaker(speakers[0], parsed_data)
        
        if len(speakers) == 2:
            return self._identify_two_speakers(speakers, parsed_data)
        
        # 3+ speakers - more complex scenario
        return self._identify_multiple_speakers(speakers, parsed_data)
    
    def _handle_single_speaker(
        self, 
        speaker_id: str, 
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Handle single speaker scenario.
        
        Single speaker is likely a patient self-recording or doctor notes.
        Check content patterns to determine role.
        """
        segments = parsed_data.get("segments", [])
        speaker_segments = [s for s in segments if s.get("speaker_id") == speaker_id]
        
        # Use the standard scoring method for consistency
        scores = self._calculate_speaker_scores(speaker_segments, is_first_speaker=True)
        
        # Determine role based on scores
        provider_score = scores["provider_score"]
        patient_score = scores["patient_score"]
        
        if provider_score > patient_score * 1.5:
            role = SpeakerRole.DOCTOR
            confidence = min(0.6 + (provider_score * 0.05), 0.85)
        elif patient_score > provider_score:
            role = SpeakerRole.PATIENT
            confidence = min(0.5 + (patient_score * 0.05), 0.75)
        else:
            role = SpeakerRole.UNKNOWN
            confidence = 0.3
        
        # Add single_speaker flag to evidence
        scores["single_speaker"] = True
        
        return {
            speaker_id: {
                "role": role.value,
                "confidence": round(confidence, 3),
                "confidence_level": self._get_confidence_level(confidence).value,
                "method": "heuristic",
                "evidence": scores
            }
        }
    
    def _identify_two_speakers(
        self, 
        speakers: List[str], 
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Identify roles for two-speaker consultation (most common case).
        
        Uses multiple heuristics:
        1. First speaker heuristic (often the doctor)
        2. Medical terminology usage
        3. Question patterns
        4. Speaking time ratio
        5. Patient symptom language
        """
        segments = parsed_data.get("segments", [])
        
        # Get first speaker
        first_segment = segments[0] if segments else None
        first_speaker = first_segment.get("speaker_id") if first_segment else None
        
        # Calculate scores for each speaker
        speaker_scores = {}
        for speaker_id in speakers:
            speaker_segments = [s for s in segments if s.get("speaker_id") == speaker_id]
            scores = self._calculate_speaker_scores(speaker_segments, speaker_id == first_speaker)
            speaker_scores[speaker_id] = scores
        
        # Assign roles based on scores
        speaker1_id, speaker2_id = speakers[0], speakers[1]
        speaker1_score = speaker_scores[speaker1_id]["provider_score"]
        speaker2_score = speaker_scores[speaker2_id]["provider_score"]
        
        # Higher provider score gets doctor role
        if speaker1_score > speaker2_score:
            doctor_id, patient_id = speaker1_id, speaker2_id
            doctor_score, patient_score = speaker1_score, speaker2_score
        else:
            doctor_id, patient_id = speaker2_id, speaker1_id
            doctor_score, patient_score = speaker2_score, speaker1_score
        
        # Calculate confidence based on score difference
        score_diff = abs(doctor_score - patient_score)
        base_confidence = min(0.6 + (score_diff * 0.1), 0.95)
        
        return {
            doctor_id: {
                "role": SpeakerRole.DOCTOR.value,
                "confidence": round(base_confidence, 3),
                "confidence_level": self._get_confidence_level(base_confidence).value,
                "method": "heuristic",
                "evidence": speaker_scores[doctor_id]
            },
            patient_id: {
                "role": SpeakerRole.PATIENT.value,
                "confidence": round(base_confidence * 0.9, 3),  # Slightly lower for patient
                "confidence_level": self._get_confidence_level(base_confidence * 0.9).value,
                "method": "heuristic",
                "evidence": speaker_scores[patient_id]
            }
        }
    
    def _identify_multiple_speakers(
        self, 
        speakers: List[str], 
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Identify roles for 3+ speakers.
        
        More complex scenarios like:
        - Doctor + Patient + Family member
        - Doctor + Nurse + Patient
        - Multiple doctors consulting
        
        Returns lower confidence for these scenarios.
        """
        segments = parsed_data.get("segments", [])
        result = {}
        
        # Score all speakers
        all_scores = []
        for speaker_id in speakers:
            speaker_segments = [s for s in segments if s.get("speaker_id") == speaker_id]
            first_speaker = (segments[0].get("speaker_id") == speaker_id) if segments else False
            scores = self._calculate_speaker_scores(speaker_segments, first_speaker)
            all_scores.append((speaker_id, scores))
        
        # Sort by provider score
        all_scores.sort(key=lambda x: x[1]["provider_score"], reverse=True)
        
        # Assign roles with lower confidence
        for idx, (speaker_id, scores) in enumerate(all_scores):
            if idx == 0:
                # Highest scorer is likely doctor
                role = SpeakerRole.DOCTOR
                confidence = min(scores["provider_score"] * 0.08, 0.75)  # Lower confidence for 3+
            elif scores["patient_score"] > scores["provider_score"]:
                # High patient score suggests patient
                role = SpeakerRole.PATIENT
                confidence = min(scores["patient_score"] * 0.06, 0.65)
            else:
                # Unknown - could be family, nurse, etc.
                role = SpeakerRole.UNKNOWN
                confidence = 0.4
            
            result[speaker_id] = {
                "role": role.value,
                "confidence": round(confidence, 3),
                "confidence_level": self._get_confidence_level(confidence).value,
                "method": "heuristic",
                "evidence": scores
            }
        
        return result
    
    def _calculate_speaker_scores(
        self, 
        segments: List[Dict[str, Any]], 
        is_first_speaker: bool
    ) -> Dict[str, Any]:
        """Calculate heuristic scores for a speaker.
        
        Returns both provider_score and patient_score along with evidence.
        """
        if not segments:
            return {"provider_score": 0, "patient_score": 0}
        
        # Combine all text
        all_text = " ".join(s["text"] for s in segments)
        
        # Count pattern matches
        medical_matches = len(self.medical_regex.findall(all_text))
        question_matches = len(self.question_regex.findall(all_text))
        patient_matches = len(self.patient_regex.findall(all_text))
        
        # Calculate total speaking time
        total_time = sum(s["duration"] for s in segments)
        
        # Provider score calculation
        provider_score = 0.0
        if is_first_speaker:
            provider_score += 2.0  # First speaker bonus
        
        provider_score += medical_matches * 0.5
        provider_score += question_matches * 0.3
        
        # Patient score calculation
        patient_score = patient_matches * 0.8
        
        # Penalty for medical terminology (unlikely for patient)
        patient_score -= medical_matches * 0.3
        
        # Normalize scores
        provider_score = max(0, min(provider_score, 10))
        patient_score = max(0, min(patient_score, 10))
        
        return {
            "provider_score": round(provider_score, 2),
            "patient_score": round(patient_score, 2),
            "medical_terms": medical_matches,
            "questions_asked": question_matches,
            "patient_patterns": patient_matches,
            "speaking_time": round(total_time, 2),
            "segment_count": len(segments),
            "is_first_speaker": is_first_speaker
        }
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level category."""
        if confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def override_role(
        self,
        speaker_mapping: Dict[str, Dict[str, Any]],
        speaker_id: str,
        new_role: str,
        reason: str = "manual_override"
    ) -> Dict[str, Dict[str, Any]]:
        """Manually override a speaker's role.
        
        Useful when heuristics fail or for manual correction.
        
        Args:
            speaker_mapping: Existing speaker mapping
            speaker_id: Speaker ID to override
            new_role: New role (doctor, patient, unknown)
            reason: Reason for override
        
        Returns:
            Updated speaker mapping
        """
        if speaker_id not in speaker_mapping:
            raise ValueError(f"Speaker {speaker_id} not found in mapping")
        
        if new_role not in [r.value for r in SpeakerRole]:
            raise ValueError(f"Invalid role: {new_role}")
        
        speaker_mapping[speaker_id]["role"] = new_role
        speaker_mapping[speaker_id]["confidence"] = 1.0  # Manual override has high confidence
        speaker_mapping[speaker_id]["confidence_level"] = ConfidenceLevel.HIGH.value
        speaker_mapping[speaker_id]["method"] = "manual_override"
        speaker_mapping[speaker_id]["override_reason"] = reason
        
        return speaker_mapping
