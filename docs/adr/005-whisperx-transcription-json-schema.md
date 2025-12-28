# ADR-005: WhisperX Transcription Result JSON Schema

- **Status:** Accepted  
- **Date:** 2025-12-23  

## Context

The WhisperX API processes audio files and returns transcription results through the `/temporal/workflow/{workflow_id}/result` endpoint. This ADR documents the standardized JSON schema that is returned from all transcription workflows, ensuring consistency across different audio sources, languages, and processing configurations.

### Validation Testing

The schema has been validated across multiple audio sources:
- Vietnamese audio (vn-2.mp3, language: vi)
- Singapore audio (sg-1.WAV, language: zh)

Both tests confirmed identical schema structure, validating consistency across different:
- File formats (MP3, WAV)
- Languages (Vietnamese, Chinese)
- Audio characteristics

## Decision

The transcription result JSON schema consists of two top-level arrays providing dual representation of the same transcription data.

### Root-Level Structure

```json
{
  "segments": [...],
  "word_segments": [...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| segments | Array of Objects | Hierarchical transcription segments with speaker labels and word alignments |
| word_segments | Array of Objects | Flat list of all words with timestamps (no speaker information) |

### Schema Hierarchy

```
Root
├── segments (array)
│   └── Segment Object
│       ├── start (number)
│       ├── end (number)
│       ├── text (string)
│       ├── speaker (string, optional)
│       └── words (array)
│           └── Word Object
│               ├── word (string)
│               ├── start (number)
│               ├── end (number)
│               ├── score (number)
│               └── speaker (string, optional)
│
└── word_segments (array)
    └── Word Segment Object
        ├── word (string)
        ├── start (number)
        ├── end (number)
        └── score (number)
```

## Detailed Specifications

### Segment Object

**Purpose:** Contains transcription divided into logical segments with speaker attribution

**Required Fields:**
- `start` (number): Segment start time in seconds (float)
- `end` (number): Segment end time in seconds (float)
- `text` (string): Full transcribed text for this segment
- `words` (array): Array of word objects with detailed timing

**Optional Fields:**
- `speaker` (string): Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")
  - Present when diarization is enabled or detected
  - Pattern: SPEAKER_XX where XX is zero-padded number

**Example:**
```json
{
  "start": 0.031,
  "end": 0.252,
  "text": " Dialogue 4.",
  "speaker": "SPEAKER_00",
  "words": [...]
}
```

### Word Object (within segments)

**Purpose:** Word-level breakdown with precise timestamps and speaker attribution

**Required Fields:**
- `word` (string): Individual word or token
- `start` (number): Word start time in seconds (float)
- `end` (number): Word end time in seconds (float)
- `score` (number): Confidence score (0.0 - 1.0, typically 0.01 - 0.012)

**Optional Fields:**
- `speaker` (string): Speaker label for this specific word

**Example:**
```json
{
  "word": "Dialogue",
  "start": 0.031,
  "end": 0.191,
  "score": 0.01,
  "speaker": "SPEAKER_00"
}
```

### Word Segment Object

**Purpose:** Simplified word representation without speaker context

**Required Fields:**
- `word` (string): Individual word or token
- `start` (number): Word start time in seconds (float)
- `end` (number): Word end time in seconds (float)
- `score` (number): Confidence score (0.0 - 1.0)

**Note:** Does NOT include speaker field

**Example:**
```json
{
  "word": "Dialogue",
  "start": 0.031,
  "end": 0.191,
  "score": 0.01
}
```

## Data Type Definitions

### TypeScript Interface

```typescript
interface TranscriptionResult {
  segments: Segment[];
  word_segments: WordSegment[];
}

interface Segment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
  words: WordInSegment[];
}

interface WordInSegment {
  word: string;
  start: number;
  end: number;
  score: number;
  speaker?: string;
}

interface WordSegment {
  word: string;
  start: number;
  end: number;
  score: number;
}
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["segments", "word_segments"],
  "properties": {
    "segments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["start", "end", "text", "words"],
        "properties": {
          "start": {"type": "number"},
          "end": {"type": "number"},
          "text": {"type": "string"},
          "speaker": {"type": "string"},
          "words": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["word", "start", "end", "score"],
              "properties": {
                "word": {"type": "string"},
                "start": {"type": "number"},
                "end": {"type": "number"},
                "score": {"type": "number"},
                "speaker": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "word_segments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["word", "start", "end", "score"],
        "properties": {
          "word": {"type": "string"},
          "start": {"type": "number"},
          "end": {"type": "number"},
          "score": {"type": "number"}
        }
      }
    }
  }
}
```

## Field Characteristics

### Timestamps

- **Format:** Floating-point numbers in seconds
- **Precision:** Up to 3 decimal places (milliseconds)
- **Examples:**
  - 0.031 = 31 milliseconds
  - 2.557 = 2 seconds 557 milliseconds
  - 33.295 = 33 seconds 295 milliseconds

### Speaker Labels

- **Format:** String with pattern SPEAKER_XX
- **Examples:** SPEAKER_00, SPEAKER_01, SPEAKER_02
- **Availability:** Conditionally present when speaker diarization is enabled via `min_speakers` or `max_speakers` parameters

### Confidence Scores

- **Format:** Floating-point number
- **Range:** 0.0 to 1.0
- **Typical Range:** 0.01 to 0.012 in production
- **Interpretation:**
  - Higher score = Higher confidence
  - Lower score = Lower confidence or uncertain word

## Use Cases

### Use segments When:
- Displaying transcription in UI with speaker labels
- Generating subtitles/captions with speaker attribution
- Analyzing conversation flow between speakers
- Requiring both text and timing information together

### Use word_segments When:
- Creating word clouds or word frequency analysis
- Analyzing word timing without speaker context
- Building search indexes of words
- Only needing word-level timestamps

### Use segments[].words When:
- Requiring speaker information for each word
- Creating karaoke-style highlighting with speaker colors
- Analyzing speech patterns per speaker
- Building speaker-aware word-level features

## Consequences

### Positive

1. **Dual Representation:** Provides both hierarchical and flat views of the same data, supporting different use cases without additional processing
2. **Consistent Structure:** Validated across multiple audio sources, languages, and formats
3. **Flexible Speaker Attribution:** Speaker information optional and conditionally included based on processing parameters
4. **Precise Timing:** Millisecond-precision timestamps enable accurate synchronization and analysis
5. **Confidence Metrics:** Word-level confidence scores support quality filtering and uncertain word detection

### Negative

1. **Data Duplication:** Words appear in both segments[].words and word_segments, increasing payload size
2. **Optional Fields:** Speaker field optionality requires clients to handle both cases
3. **Nested Structure:** segments[].words nesting may require additional parsing in some languages

### Neutral

1. **No Language Field:** Language information not included in result (captured in workflow metadata)
2. **No Metadata:** Processing parameters, audio duration, and other metadata not included (available via workflow status endpoint)
3. **No Error Information:** Error handling managed through workflow status, not in result structure

## Compliance

This schema aligns with:
- WhisperX output format specification
- Temporal workflow result patterns
- RESTful API best practices for JSON responses

## Versioning

**Current Version:** 1.0  
**Last Updated:** 2025-12-23

Future schema changes will be versioned and documented as separate ADRs. Breaking changes will require major version increment.

## References

- WhisperX Documentation: https://github.com/m-bain/whisperX
- API Endpoint: `/temporal/workflow/{workflow_id}/result`
- Related ADRs: ADR-002 (LM Studio Integration Strategy)

## Appendix

### Validation Test Results

**Test 1: Vietnamese Audio**
- File: vn-2.mp3 (1.4 MB)
- Language: vi
- Segments: 5
- Word segments: 67
- Schema: Confirmed

**Test 2: Singapore Audio**
- File: sg-1.WAV (2.1 MB)
- Language: zh
- Segments: Varies
- Word segments: Varies
- Schema: Confirmed identical to Test 1

Both tests confirmed 100% schema consistency across:
- Root-level fields: segments, word_segments
- Segment structure: start, end, text, speaker, words
- Word structure (in segments): word, start, end, score, speaker
- Word segment structure: word, start, end, score
