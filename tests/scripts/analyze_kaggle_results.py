"""Script to test Phase 1 with Kaggle medical dataset.

This script processes a medical interview audio through WhisperX,
runs Phase 1 data transformation,  and compares results with ground truth.
"""

import json
import sys
from pathlib import Path

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.whisperx_parser import WhisperXParser
from app.services.speaker_identifier import SpeakerIdentifier


def load_ground_truth(transcript_path):
    """Load and parse ground truth transcript."""
    with open(transcript_path, 'r') as f:
        content = f.read()
    
    # Parse lines into doctor/patient segments
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    segments = []
    for line in lines:
        if line.startswith('D:'):
            segments.append(('DOCTOR', line[2:].strip()))
        elif line.startswith('P:'):
            segments.append(('PATIENT', line[2:].strip()))
    
    return segments


def analyze_whisperx_result(result_path, ground_truth_path):
    """Analyze WhisperX result and compare to ground truth."""
    
    # Load WhisperX result
    with open(result_path, 'r') as f:
        whisperx_result = json.load(f)
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Initialize Phase 1 services
    parser = WhisperXParser()
    identifier = SpeakerIdentifier()
    
    # Parse WhisperX result
    parsed = parser.parse(whisperx_result)
    
    # Identify speaker roles
    speaker_mapping = identifier.identify_roles(parsed)
    
    # Generate transcript
    transcript = parser.get_full_transcript(parsed, include_speakers=True)
    
    # Get statistics
    stats = parser.get_speaker_statistics(parsed)
    
    # Analysis
    print("=" * 80)
    print("PHASE 1 ANALYSIS - Kaggle Medical Dataset")
    print("=" * 80)
    
    print(f"\n[Ground Truth]")
    print(f"  Total segments: {len(ground_truth)}")
    doctor_count = sum(1 for role, _ in ground_truth if role == 'DOCTOR')
    patient_count = sum(1 for role, _ in ground_truth if role == 'PATIENT')
    print(f"  Doctor segments: {doctor_count}")
    print(f"  Patient segments: {patient_count}")
    
    print(f"\n[WhisperX + Phase 1]")
    print(f"  Total segments: {parsed['metadata']['total_segments']}")
    print(f"  Total words: {parsed['metadata']['total_words']}")
    print(f"  Duration: {parsed['metadata']['total_duration']:.1f}s")
    print(f"  Speakers detected: {parsed['metadata']['speaker_count']}")
    print(f"  Avg confidence: {parsed['metadata']['avg_confidence']:.3f}")
    
    print(f"\n[Speaker Role Identification]")
    for speaker_id, info in speaker_mapping.items():
        print(f"  {speaker_id}:")
        print(f"    Role: {info['role']}")
        print(f"    Confidence: {info['confidence']} ({info['confidence_level']})")
        print(f"    Evidence: provider_score={info['evidence']['provider_score']}, patient_score={info['evidence']['patient_score']}")
    
    print(f"\n[Speaker Statistics]")
    for speaker_id, speaker_stats in stats.items():
        role = speaker_mapping[speaker_id]['role']
        print(f"  {speaker_id} ({role}):")
        print(f"    Segments: {speaker_stats['segment_count']}")
        print(f"    Words: {speaker_stats['word_count']}")
        print(f"    Speaking time: {speaker_stats['speaking_time']:.1f}s")
        print(f"    Avg confidence: {speaker_stats['avg_confidence']:.3f}")
    
    print(f"\n[Transcript Preview]")
    preview_lines = transcript.split('\n\n')[:5]
    for line in preview_lines:
        print(f"  {line}")
    print(f"  ...")
    
    print(f"\n[Ground Truth Preview]")
    for i, (role, text) in enumerate(ground_truth[:5]):
        print(f"  {role}: {text[:80]}...")
        if i >= 4:
            break
    print(f"  ...")
    
    print("\n" + "=" * 80)
    
    # Return results for further analysis
    return {
        'parsed': parsed,
        'speaker_mapping': speaker_mapping,
        'transcript': transcript,
        'stats': stats,
        'ground_truth': ground_truth
    }


if __name__ == "__main__":
    result_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kaggle_res0001_result.json"
    ground_truth_file = sys.argv[2] if len(sys.argv) > 2 else "datasets/kaggle-simulated-patient-physicia-interviews/transcripts/RES0001.txt"
    
    analyze_whisperx_result(result_file, ground_truth_file)
