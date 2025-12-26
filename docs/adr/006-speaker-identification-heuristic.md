# ADR-006: Speaker Identification Heuristic Algorithm

**Status:** Accepted  
**Date:** 2025-12-26  

## Context

The WhisperX API provides speaker diarization that assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.) but does not identify the role of each speaker. In medical consultation transcriptions, it is critical to distinguish between healthcare providers (doctors, nurses) and patients for downstream analysis, compliance, and reporting purposes.

A heuristic-based algorithm is needed to automatically assign speaker roles based on speech patterns, content, and contextual clues. This ADR documents the scoring formulas and confidence calculations used in the speaker identification system.

## Decision

Implement a heuristic-based speaker identification system that assigns speaker roles (doctor, patient, unknown) based on pattern-matching and scoring algorithms. The system provides confidence scores to indicate reliability of role assignments.

### Scoring System

Each speaker receives two scores based on their speech patterns:

1. **Provider Score**: Likelihood of being a healthcare provider (doctor/nurse)
2. **Patient Score**: Likelihood of being a patient

#### Provider Score Formula

```
provider_score = base_score + medical_bonus + question_bonus
```

Where:
- `base_score = FIRST_SPEAKER_BONUS (2.0)` if speaker is first, else `0`
- `medical_bonus = medical_term_count × MEDICAL_TERM_WEIGHT (0.5)`
- `question_bonus = question_count × QUESTION_WEIGHT (0.3)`

**Score Range**: Normalized to `[0, 10]`

**Example**:
- First speaker with 8 medical terms and 5 questions:
  - `provider_score = 2.0 + (8 × 0.5) + (5 × 0.3) = 2.0 + 4.0 + 1.5 = 7.5`

#### Patient Score Formula

```
patient_score = patient_pattern_bonus - medical_penalty
```

Where:
- `patient_pattern_bonus = patient_pattern_count × PATIENT_PATTERN_WEIGHT (0.8)`
- `medical_penalty = medical_term_count × MEDICAL_TERM_PENALTY (0.3)`

**Score Range**: Normalized to `[0, 10]`

**Example**:
- Speaker with 10 patient patterns ("I feel", "hurts") and 2 medical terms:
  - `patient_score = (10 × 0.8) - (2 × 0.3) = 8.0 - 0.6 = 7.4`

### Scoring Weights

| Constant | Value | Purpose |
|----------|-------|---------|
| `FIRST_SPEAKER_BONUS` | 2.0 | First speaker is typically the provider |
| `MEDICAL_TERM_WEIGHT` | 0.5 | Weight for medical terminology usage |
| `QUESTION_WEIGHT` | 0.3 | Weight for question patterns |
| `PATIENT_PATTERN_WEIGHT` | 0.8 | Weight for patient symptom language |
| `MEDICAL_TERM_PENALTY` | 0.3 | Penalty on patient score for medical terms |

### Pattern Detection

#### Medical Terms (Provider Indicators)
- Diagnosis, prescribed, medication, treatment, symptoms, examination
- Blood pressure, heart rate, temperature, pulse, respiratory
- Test results, lab work, imaging, x-ray, MRI, CT scan
- Follow-up, referral, consultation, appointment

#### Question Patterns (Provider Indicators)
- "what brings", "what symptoms", "what happened"
- "how long", "how often", "how severe"
- "have you", "are you", "do you", "did you"
- "when did", "any pain", "any discomfort"

#### Patient Patterns
- "I feel", "I have", "I've been", "my"
- "hurts", "painful", "ache", "sore", "uncomfortable"
- "started", "began", "since", "for the past"

## Confidence Calculation Formulas

Confidence levels indicate certainty of role assignments:
- **HIGH**: > 0.8
- **MEDIUM**: 0.5 - 0.8
- **LOW**: < 0.5

### Scenario 1: Two Speakers (Most Common)

This is the most reliable scenario with highest confidence.

#### Doctor Confidence

```
confidence = min(0.6 + (score_diff × 0.1), 0.95)
```

Where `score_diff = |doctor_provider_score - patient_provider_score|`

**Range**: `[0.6, 0.95]`

**Examples**:
- Score difference of 0: `min(0.6 + 0, 0.95) = 0.60` (MEDIUM)
- Score difference of 2: `min(0.6 + 0.2, 0.95) = 0.80` (HIGH)
- Score difference of 5: `min(0.6 + 0.5, 0.95) = 0.95` (HIGH, capped)

#### Patient Confidence

```
confidence = doctor_confidence × 0.9
```

**Range**: `[0.54, 0.855]`

**Rationale**: Patient assignment is slightly less reliable than doctor assignment, hence 90% of doctor confidence.

### Scenario 2: Single Speaker

Lower confidence due to ambiguity (could be patient self-recording or doctor notes).

#### If Provider Score > Patient Score × 1.5 → DOCTOR

```
confidence = min(0.6 + (provider_score × 0.05), 0.85)
```

**Range**: `[0.6, 0.85]`

**Examples**:
- `provider_score = 0`: `min(0.6, 0.85) = 0.60`
- `provider_score = 5`: `min(0.6 + 0.25, 0.85) = 0.75` (MEDIUM)
- `provider_score = 10`: `min(0.6 + 0.50, 0.85) = 0.85` (HIGH, capped)

#### If Patient Score > Provider Score → PATIENT

```
confidence = min(0.5 + (patient_score × 0.05), 0.75)
```

**Range**: `[0.5, 0.75]`

**Examples**:
- `patient_score = 0`: `min(0.5, 0.75) = 0.50` (MEDIUM)
- `patient_score = 5`: `min(0.5 + 0.25, 0.75) = 0.75` (MEDIUM, capped)

#### Otherwise → UNKNOWN

```
confidence = 0.3
```

### Scenario 3: Three or More Speakers

Significantly lower confidence due to complexity (family members, nurses, etc.).

#### Highest Provider Score → DOCTOR

```
confidence = min(0.4 + (provider_score × 0.06), 0.75)
```

**Range**: `[0.4, 0.75]`

**Examples**:
- `provider_score = 0`: `min(0.4, 0.75) = 0.40` (LOW)
- `provider_score = 5`: `min(0.4 + 0.30, 0.75) = 0.70` (MEDIUM)
- `provider_score = 10`: `min(0.4 + 0.60, 0.75) = 0.75` (MEDIUM, capped)

#### If Patient Score > Provider Score → PATIENT

```
confidence = min(patient_score × 0.08, 0.65)
```

**Range**: `[0.0, 0.65]`

**Known Issue**: This formula lacks a base confidence value, which can produce very low confidence scores.

**Examples**:
- `patient_score = 0`: `min(0, 0.65) = 0.00` (Below LOW threshold)
- `patient_score = 3`: `min(0.24, 0.65) = 0.24` (LOW)
- `patient_score = 5`: `min(0.40, 0.65) = 0.40` (LOW)
- `patient_score = 8`: `min(0.64, 0.65) = 0.64` (MEDIUM)

**Proposed Fix**:
```
confidence = min(0.35 + (patient_score × 0.04), 0.65)
```

This would provide:
- `patient_score = 0`: 0.35 (LOW but reasonable)
- `patient_score = 5`: 0.55 (MEDIUM)
- `patient_score = 10`: 0.65 (MEDIUM, capped)

#### Otherwise → UNKNOWN

```
confidence = 0.4
```

## Formula Consistency Analysis

### Base Confidence Comparison

| Scenario | Doctor Base | Patient Base | Notes |
|----------|-------------|--------------|-------|
| 2 speakers | 0.60 | 0.54 | Highest confidence |
| 1 speaker (doctor) | 0.60 | - | Same base as 2-speaker |
| 1 speaker (patient) | - | 0.50 | Slightly lower |
| 3+ speakers (doctor) | 0.40 | - | Lower confidence |
| 3+ speakers (patient) | - | **0.00** | Issue: No base confidence |

## Testing Examples

### Test Case 1: Two-Speaker Consultation

```
Speaker A: provider_score=7.5, patient_score=2.0
Speaker B: provider_score=3.0, patient_score=6.5

Score diff = 7.5 - 3.0 = 4.5
Doctor (A) confidence = min(0.6 + 4.5×0.1, 0.95) = 0.95 (HIGH)
Patient (B) confidence = 0.95 × 0.9 = 0.855 (HIGH)
```

### Test Case 2: Single Speaker (Doctor Notes)

```
Speaker A: provider_score=8.0, patient_score=1.0

provider_score > patient_score × 1.5? → 8.0 > 1.5 = true
Doctor confidence = min(0.6 + 8.0×0.05, 0.85) = 0.85 (HIGH)
```

### Test Case 3: Three Speakers

```
Speaker A: provider_score=7.0, patient_score=1.5 → DOCTOR
Speaker B: provider_score=2.0, patient_score=7.0 → PATIENT  
Speaker C: provider_score=3.0, patient_score=3.5 → UNKNOWN

Doctor (A) confidence = min(0.4 + 7.0×0.06, 0.75) = 0.75 (MEDIUM)
Patient (B) confidence = min(7.0×0.08, 0.65) = 0.56 (MEDIUM)
Unknown (C) confidence = 0.4 (LOW)
```

## Consequences

### Positive

1. **Automated Role Assignment**: Eliminates manual speaker role labeling in most cases
2. **Confidence Metrics**: Provides transparency about reliability of assignments
3. **Pattern-Based**: Uses domain-specific medical conversation patterns
4. **Scalable**: Handles 1, 2, or 3+ speaker scenarios
5. **Tunable**: Constants can be adjusted based on validation results

### Negative

1. **Heuristic Limitations**: May fail with non-standard conversations or informal language
2. **Language Dependency**: Pattern matching tuned for English; other languages require adjustment
3. **No Learning**: Does not improve over time without manual constant tuning
4. **Low Confidence Edge Cases**: 3+ speaker patient formula may produce very low confidence scores

### Neutral

1. **Manual Override Available**: System includes `override_role()` method for corrections
2. **Validation Required**: Confidence scores should be validated against real-world transcriptions
3. **Future ML Integration**: Heuristic approach can be replaced or augmented with ML models

## Alternatives Considered

### Machine Learning Approach

**Pros**: Could learn from labeled data, better generalization
**Cons**: Requires labeled training data, more complex deployment, higher computational cost

**Decision**: Start with heuristics for faster implementation and lower resource requirements. ML can be added later as enhancement.

### Rule-Based Expert System

**Pros**: Explicit rules easier to maintain and explain
**Cons**: Brittle, requires extensive rule engineering for edge cases

**Decision**: Scoring approach provides more flexibility than hard rules.

### Manual Labeling Only

**Pros**: 100% accuracy when done correctly
**Cons**: Not scalable, time-consuming, error-prone for large volumes

**Decision**: Automated approach with manual override capability provides best balance.

## Implementation

- **File**: [`app/services/speaker_identifier.py`](file:///Users/laansdole/Projects/whisperX-FastAPI/app/services/speaker_identifier.py)
- **Class**: `SpeakerIdentifier`
- **Methods**:
  - `identify_roles()`: Main entry point
  - `_identify_two_speakers()`: Two-speaker scenario
  - `_identify_multiple_speakers()`: Three or more speakers
  - `_handle_single_speaker()`: Single speaker scenario
  - `_calculate_speaker_scores()`: Core scoring algorithm
  - `override_role()`: Manual role override

## Versioning

**Current Version:** 1.0  
**Last Updated:** 2025-12-26

Future improvements:
- Add base confidence to 3+ speaker patient formula
- Tune constants based on validation results
- Add support for non-English pattern matching
- Consider ML-based enhancement

## References

- WhisperX Diarization: https://github.com/m-bain/whisperX
- Related ADR: ADR-005 (WhisperX Transcription JSON Schema)
- Validation Report: [`docs/validation/speaker-diarization-validation-report.md`](../validation/speaker-diarization-validation-report.md)
