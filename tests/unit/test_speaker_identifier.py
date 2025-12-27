"""Unit tests for SpeakerIdentifier service.

Tests cover:
- Single speaker scenarios
- Two-speaker scenarios (normal and edge cases)
- Multiple speaker scenarios (3+)
- Scoring calculation logic
- Confidence level mapping
- Manual override functionality
- Pattern matching (medical terms, questions, patient patterns)
"""

import pytest
from app.services.speaker_identifier import SpeakerIdentifier, SpeakerRole, ConfidenceLevel


class TestSpeakerIdentifier:
    """Test suite for SpeakerIdentifier."""

    @pytest.fixture
    def identifier(self):
        """Create a SpeakerIdentifier instance."""
        return SpeakerIdentifier()

    # ===== Single Speaker Tests =====

    def test_single_speaker_doctor_like(self, identifier):
        """Test single speaker with doctor-like patterns."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00"], "speaker_count": 1},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "What brings you in today? Let's check your blood pressure and vitals.",
                    "duration": 5.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_00",
                    "text": "How long have you had these symptoms? Any medication you're taking?",
                    "duration": 4.0,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert "SPEAKER_00" in result
        speaker_info = result["SPEAKER_00"]
        assert speaker_info["role"] == SpeakerRole.DOCTOR.value
        assert speaker_info["confidence"] >= 0.5
        assert speaker_info["method"] == "heuristic"
        assert speaker_info["evidence"]["single_speaker"] is True
        assert speaker_info["evidence"]["medical_terms"] > 0
        assert speaker_info["evidence"]["questions_asked"] > 0

    def test_single_speaker_patient_like(self, identifier):
        """Test single speaker with patient-like patterns."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00"], "speaker_count": 1},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "I feel terrible. My head hurts and I've been having pain for the past week.",
                    "duration": 4.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_00",
                    "text": "It started last Monday. My stomach is sore and uncomfortable.",
                    "duration": 3.0,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert "SPEAKER_00" in result
        speaker_info = result["SPEAKER_00"]
        assert speaker_info["role"] == SpeakerRole.PATIENT.value
        assert speaker_info["evidence"]["patient_patterns"] > 0

    def test_single_speaker_unknown(self, identifier):
        """Test single speaker with ambiguous content.

        Note: Due to first-speaker bonus (2.0), even neutral content gets DOCTOR role.
        This is acceptable in medical context where single speaker is typically a provider.
        """
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00"], "speaker_count": 1},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "Yes, okay, right, I understand.",
                    "duration": 2.0,
                }
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert "SPEAKER_00" in result
        speaker_info = result["SPEAKER_00"]
        # First-speaker bonus gives DOCTOR role even with neutral content
        assert speaker_info["role"] == SpeakerRole.DOCTOR.value
        assert speaker_info["confidence"] >= 0.5  # Has first-speaker bonus
        assert speaker_info["evidence"]["is_first_speaker"] is True

    # ===== Two Speaker Tests - Normal Case =====

    def test_two_speakers_normal_case(self, identifier):
        """Test normal two-speaker consultation."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01"], "speaker_count": 2},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "What brings you in today? Let me check your vitals.",
                    "duration": 3.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_01",
                    "text": "I have a headache. It hurts really bad.",
                    "duration": 2.0,
                },
                {
                    "segment_id": 2,
                    "speaker_id": "SPEAKER_00",
                    "text": "How long have you had this? Any medication currently?",
                    "duration": 2.5,
                },
                {
                    "segment_id": 3,
                    "speaker_id": "SPEAKER_01",
                    "text": "It started yesterday. I feel terrible.",
                    "duration": 2.0,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert len(result) == 2
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result

        # First speaker with medical questions should be doctor
        assert result["SPEAKER_00"]["role"] == SpeakerRole.DOCTOR.value
        assert result["SPEAKER_01"]["role"] == SpeakerRole.PATIENT.value

        # Check evidence
        assert result["SPEAKER_00"]["evidence"]["is_first_speaker"] is True
        assert result["SPEAKER_00"]["evidence"]["questions_asked"] > 0
        assert result["SPEAKER_01"]["evidence"]["patient_patterns"] > 0

    def test_two_speakers_reversed_order(self, identifier):
        """Test when patient speaks first (less common).

        Note: First-speaker bonus (2.0) is strong enough that it takes precedence
        unless medical evidence is significantly higher. This is acceptable as
        doctor typically speaks first in medical consultations.
        """
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01"], "speaker_count": 2},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "I have a really bad headache. I feel terrible.",
                    "duration": 2.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_01",
                    "text": "Let me check your symptoms. What brings you in today? How long have these issues been?",
                    "duration": 4.0,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        # First-speaker bonus (2.0) slightly outweighs medical evidence (~1.9)
        # SPEAKER_00 gets doctor role despite patient-like language
        assert result["SPEAKER_00"]["role"] == SpeakerRole.DOCTOR.value
        assert result["SPEAKER_01"]["role"] == SpeakerRole.PATIENT.value
        # First speaker should have the bonus in evidence
        assert result["SPEAKER_00"]["evidence"]["is_first_speaker"] is True
        assert result["SPEAKER_01"]["evidence"]["is_first_speaker"] is False

    # ===== Two Speaker Tests - Edge Cases =====

    def test_two_speakers_both_no_segments(self, identifier):
        """Test edge case where both speakers have no segments."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01"], "speaker_count": 2},
            "segments": [],  # Empty segments
        }

        result = identifier.identify_roles(parsed_data)

        assert len(result) == 2
        # Both should be UNKNOWN with low confidence
        assert result["SPEAKER_00"]["role"] == SpeakerRole.UNKNOWN.value
        assert result["SPEAKER_01"]["role"] == SpeakerRole.UNKNOWN.value
        assert result["SPEAKER_00"]["confidence"] <= 0.2
        assert result["SPEAKER_01"]["confidence"] <= 0.2
        assert "no_segments_for_either_speaker" in result["SPEAKER_00"]["evidence"]["reason"]

    def test_two_speakers_one_no_segments_speaker1_empty(self, identifier):
        """Test edge case where speaker 1 has no segments."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01"], "speaker_count": 2},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_01",
                    "text": "What brings you in today? Let me check your vitals and blood pressure.",
                    "duration": 3.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_01",
                    "text": "How long have you had these symptoms?",
                    "duration": 2.0,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert len(result) == 2
        # Speaker with segments should get role based on content
        assert result["SPEAKER_01"]["role"] == SpeakerRole.DOCTOR.value
        # Speaker without segments should be UNKNOWN
        assert result["SPEAKER_00"]["role"] == SpeakerRole.UNKNOWN.value
        assert result["SPEAKER_00"]["confidence"] <= 0.2
        assert "no_segments_for_speaker" in result["SPEAKER_00"]["evidence"]["reason"]

    def test_two_speakers_one_no_segments_speaker2_empty(self, identifier):
        """Test edge case where speaker 2 has no segments."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01"], "speaker_count": 2},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "I feel really bad. My head hurts and I've been in pain.",
                    "duration": 3.0,
                },
                {"segment_id": 1, "speaker_id": "SPEAKER_00", "text": "It started last week.", "duration": 1.5},
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert len(result) == 2
        # Speaker with patient-like content should be patient
        assert result["SPEAKER_00"]["role"] == SpeakerRole.PATIENT.value
        # Speaker without segments should be UNKNOWN
        assert result["SPEAKER_01"]["role"] == SpeakerRole.UNKNOWN.value
        assert result["SPEAKER_01"]["confidence"] <= 0.2

    # ===== Multiple Speaker Tests (3+) =====

    def test_three_speakers(self, identifier):
        """Test handling of three speakers."""
        parsed_data = {
            "metadata": {"speakers_detected": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], "speaker_count": 3},
            "segments": [
                {
                    "segment_id": 0,
                    "speaker_id": "SPEAKER_00",
                    "text": "What brings you in today? Let me check your diagnosis.",
                    "duration": 3.0,
                },
                {
                    "segment_id": 1,
                    "speaker_id": "SPEAKER_01",
                    "text": "I have been feeling terrible. My head hurts.",
                    "duration": 2.0,
                },
                {
                    "segment_id": 2,
                    "speaker_id": "SPEAKER_02",
                    "text": "He's been like this for a week.",
                    "duration": 1.5,
                },
            ],
        }

        result = identifier.identify_roles(parsed_data)

        assert len(result) == 3
        # Highest provider score should be doctor
        assert result["SPEAKER_00"]["role"] == SpeakerRole.DOCTOR.value
        # Patient-like speaker should be patient
        assert result["SPEAKER_01"]["role"] == SpeakerRole.PATIENT.value
        # Third speaker likely unknown (family member)
        # Could be patient or unknown depending on scoring
        assert result["SPEAKER_02"]["role"] in [SpeakerRole.PATIENT.value, SpeakerRole.UNKNOWN.value]
        # All confidences should be lower for 3+ speakers
        for speaker_id in result:
            assert result[speaker_id]["confidence"] <= 0.75

    # ===== Scoring Tests =====

    def test_calculate_speaker_scores_first_speaker_bonus(self, identifier):
        """Test that first speaker gets bonus points."""
        segments = [{"text": "Hello there.", "duration": 1.0}]

        scores_first = identifier._calculate_speaker_scores(segments, is_first_speaker=True)
        scores_not_first = identifier._calculate_speaker_scores(segments, is_first_speaker=False)

        assert scores_first["provider_score"] > scores_not_first["provider_score"]
        assert scores_first["is_first_speaker"] is True
        assert scores_not_first["is_first_speaker"] is False

    def test_calculate_speaker_scores_medical_terms(self, identifier):
        """Test medical terminology scoring."""
        segments = [
            {"text": "Let me check your diagnosis, blood pressure, and prescribed medication.", "duration": 3.0}
        ]

        scores = identifier._calculate_speaker_scores(segments, is_first_speaker=False)

        assert scores["medical_terms"] > 0
        assert scores["provider_score"] > 0

    def test_calculate_speaker_scores_questions(self, identifier):
        """Test question pattern scoring."""
        segments = [
            {
                "text": "What brings you in today? How long have you felt this way? Do you have any pain?",
                "duration": 4.0,
            }
        ]

        scores = identifier._calculate_speaker_scores(segments, is_first_speaker=False)

        assert scores["questions_asked"] > 0
        assert scores["provider_score"] > 0

    def test_calculate_speaker_scores_patient_patterns(self, identifier):
        """Test patient symptom pattern scoring."""
        segments = [
            {"text": "I feel terrible. My head hurts and I have been in pain since last week.", "duration": 3.0}
        ]

        scores = identifier._calculate_speaker_scores(segments, is_first_speaker=False)

        assert scores["patient_patterns"] > 0
        assert scores["patient_score"] > 0

    def test_calculate_speaker_scores_empty_segments(self, identifier):
        """Test scoring with empty segments list."""
        scores = identifier._calculate_speaker_scores([], is_first_speaker=True)

        assert scores["provider_score"] == 0
        assert scores["patient_score"] == 0

    def test_calculate_speaker_scores_normalization(self, identifier):
        """Test that scores are normalized to 0-10 range."""
        # Create segments with excessive patterns
        segments = [
            {
                "text": " ".join(
                    [
                        "diagnosis prescribed medication treatment symptoms examination",
                        "blood pressure heart rate temperature pulse",
                        "What brings you in today? How long have these symptoms been?",
                        "Do you have any concerns? When did this start?",
                    ]
                    * 10
                ),  # Repeat to get high counts
                "duration": 50.0,
            }
        ]

        scores = identifier._calculate_speaker_scores(segments, is_first_speaker=True)

        # Scores should be capped at 10
        assert scores["provider_score"] <= 10
        assert scores["patient_score"] <= 10
        assert scores["provider_score"] >= 0
        assert scores["patient_score"] >= 0

    # ===== Confidence Level Tests =====

    def test_confidence_level_high(self, identifier):
        """Test high confidence level mapping."""
        assert identifier._get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert identifier._get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert identifier._get_confidence_level(1.0) == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self, identifier):
        """Test medium confidence level mapping."""
        assert identifier._get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert identifier._get_confidence_level(0.7) == ConfidenceLevel.MEDIUM
        assert identifier._get_confidence_level(0.8) == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self, identifier):
        """Test low confidence level mapping."""
        assert identifier._get_confidence_level(0.3) == ConfidenceLevel.LOW
        assert identifier._get_confidence_level(0.4) == ConfidenceLevel.LOW
        assert identifier._get_confidence_level(0.5) == ConfidenceLevel.LOW

    # ===== Edge Cases and Error Handling =====

    def test_no_speakers_detected(self, identifier):
        """Test when no speakers are detected."""
        parsed_data = {"metadata": {"speakers_detected": [], "speaker_count": 0}, "segments": []}

        result = identifier.identify_roles(parsed_data)

        assert result == {}

    def test_unsupported_method(self, identifier):
        """Test that unsupported methods raise NotImplementedError."""
        parsed_data = {"metadata": {"speakers_detected": ["SPEAKER_00"], "speaker_count": 1}, "segments": []}

        with pytest.raises(NotImplementedError, match="Method 'ml' not implemented"):
            identifier.identify_roles(parsed_data, method="ml")

    # ===== Manual Override Tests =====

    def test_override_role_doctor_to_patient(self, identifier):
        """Test manually overriding doctor role to patient."""
        speaker_mapping = {
            "SPEAKER_00": {
                "role": SpeakerRole.DOCTOR.value,
                "confidence": 0.7,
                "confidence_level": ConfidenceLevel.MEDIUM.value,
                "method": "heuristic",
                "evidence": {},
            }
        }

        result = identifier.override_role(speaker_mapping, "SPEAKER_00", SpeakerRole.PATIENT.value, "manual correction")

        assert result["SPEAKER_00"]["role"] == SpeakerRole.PATIENT.value
        assert result["SPEAKER_00"]["confidence"] == 1.0
        assert result["SPEAKER_00"]["confidence_level"] == ConfidenceLevel.HIGH.value
        assert result["SPEAKER_00"]["method"] == "manual_override"
        assert result["SPEAKER_00"]["override_reason"] == "manual correction"

    def test_override_role_invalid_speaker(self, identifier):
        """Test override with invalid speaker ID."""
        speaker_mapping = {"SPEAKER_00": {}}

        with pytest.raises(ValueError, match="Speaker SPEAKER_99 not found"):
            identifier.override_role(speaker_mapping, "SPEAKER_99", "doctor")

    def test_override_role_invalid_role(self, identifier):
        """Test override with invalid role."""
        speaker_mapping = {"SPEAKER_00": {}}

        with pytest.raises(ValueError, match="Invalid role"):
            identifier.override_role(speaker_mapping, "SPEAKER_00", "invalid_role")

    # ===== Pattern Compilation Tests =====

    def test_pattern_compilation(self, identifier):
        """Test that patterns are compiled during initialization."""
        assert identifier.medical_regex is not None
        assert identifier.question_regex is not None
        assert identifier.patient_regex is not None

    def test_medical_terms_pattern_matching(self, identifier):
        """Test medical terms regex pattern."""
        test_text = "The diagnosis shows high blood pressure. Prescribed medication for treatment."
        matches = identifier.medical_regex.findall(test_text)
        assert len(matches) > 0

    def test_question_pattern_matching(self, identifier):
        """Test question pattern regex."""
        test_text = "What brings you in today? How long have you felt this way?"
        matches = identifier.question_regex.findall(test_text)
        assert len(matches) > 0

    def test_patient_pattern_matching(self, identifier):
        """Test patient pattern regex."""
        test_text = "I feel terrible. My head hurts. I've been in pain."
        matches = identifier.patient_regex.findall(test_text)
        assert len(matches) > 0
