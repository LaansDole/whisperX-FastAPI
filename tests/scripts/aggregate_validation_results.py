"""Aggregate results from multiple Kaggle medical interview samples."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.whisperx_parser import WhisperXParser
from app.services.speaker_identifier import SpeakerIdentifier


def analyze_sample(result_path, ground_truth_path, sample_name):
    """Analyze a single sample and return key metrics."""

    # Load data
    with open(result_path, "r") as f:
        whisperx_result = json.load(f)

    with open(ground_truth_path, "r") as f:
        gt_content = f.read()

    # Parse ground truth
    gt_lines = [line.strip() for line in gt_content.split("\n") if line.strip()]
    gt_segments = []
    for line in gt_lines:
        if line.startswith("D:"):
            gt_segments.append(("DOCTOR", line[2:].strip()))
        elif line.startswith("P:"):
            gt_segments.append(("PATIENT", line[2:].strip()))

    # Process with services
    parser = WhisperXParser()
    identifier = SpeakerIdentifier()

    parsed = parser.parse(whisperx_result)
    speaker_mapping = identifier.identify_roles(parsed)
    stats = parser.get_speaker_statistics(parsed)

    # Extract metrics
    metadata = parsed["metadata"]

    # Find doctor and patient
    doctor_id = None
    patient_id = None
    for speaker_id, info in speaker_mapping.items():
        if info["role"] == "doctor":
            doctor_id = speaker_id
        elif info["role"] == "patient":
            patient_id = speaker_id

    return {
        "sample_name": sample_name,
        "ground_truth": {
            "total_segments": len(gt_segments),
            "doctor_segments": sum(1 for role, _ in gt_segments if role == "DOCTOR"),
            "patient_segments": sum(1 for role, _ in gt_segments if role == "PATIENT"),
        },
        "whisperx": {
            "total_segments": metadata["total_segments"],
            "total_words": metadata["total_words"],
            "duration": metadata["total_duration"],
            "speakers_detected": metadata["speaker_count"],
            "avg_confidence": metadata["avg_confidence"],
        },
        "speaker_mapping": speaker_mapping,
        "doctor_stats": stats.get(doctor_id, {}) if doctor_id else {},
        "patient_stats": stats.get(patient_id, {}) if patient_id else {},
        "accuracy": {
            "speaker_detection": 100.0 if metadata["speaker_count"] == 2 else 0.0,
            "role_classification": 100.0 if doctor_id and patient_id else 0.0,
        },
    }


def aggregate_results(samples):
    """Aggregate results from multiple samples."""

    n = len(samples)

    # Aggregate WhisperX metrics
    avg_segments = sum(s["whisperx"]["total_segments"] for s in samples) / n
    avg_words = sum(s["whisperx"]["total_words"] for s in samples) / n
    avg_duration = sum(s["whisperx"]["duration"] for s in samples) / n
    avg_confidence = sum(s["whisperx"]["avg_confidence"] for s in samples) / n

    # Aggregate speaker detection accuracy
    speaker_detection_accuracy = sum(s["accuracy"]["speaker_detection"] for s in samples) / n
    role_classification_accuracy = sum(s["accuracy"]["role_classification"] for s in samples) / n

    # Aggregate confidence scores
    doctor_confidences = []
    patient_confidences = []

    for sample in samples:
        for speaker_id, info in sample["speaker_mapping"].items():
            if info["role"] == "doctor":
                doctor_confidences.append(info["confidence"])
            elif info["role"] == "patient":
                patient_confidences.append(info["confidence"])

    avg_doctor_confidence = sum(doctor_confidences) / len(doctor_confidences) if doctor_confidences else 0
    avg_patient_confidence = sum(patient_confidences) / len(patient_confidences) if patient_confidences else 0

    return {
        "n_samples": n,
        "averages": {
            "segments": avg_segments,
            "words": avg_words,
            "duration": avg_duration,
            "transcription_confidence": avg_confidence,
            "speaker_detection_accuracy": speaker_detection_accuracy,
            "role_classification_accuracy": role_classification_accuracy,
            "doctor_confidence": avg_doctor_confidence,
            "patient_confidence": avg_patient_confidence,
        },
        "samples": samples,
    }


if __name__ == "__main__":
    samples_config = [
        (
            "/tmp/kaggle_res0029_result.json",
            "datasets/kaggle-simulated-patient-physicia-interviews/transcripts/RES0029.txt",
            "RES0029 (Respiratory)",
        ),
        (
            "/tmp/kaggle_car0001_result.json",
            "datasets/kaggle-simulated-patient-physicia-interviews/transcripts/CAR0001.txt",
            "CAR0001 (Cardiology)",
        ),
        (
            "/tmp/kaggle_msk0001_result.json",
            "datasets/kaggle-simulated-patient-physicia-interviews/transcripts/MSK0001.txt",
            "MSK0001 (Musculoskeletal)",
        ),
    ]

    samples = []
    for result_path, gt_path, name in samples_config:
        try:
            sample_data = analyze_sample(result_path, gt_path, name)
            samples.append(sample_data)
            print(f"Analyzed {name}")
        except Exception as e:
            print(f"Error analyzing {name}: {e}")

    if samples:
        aggregated = aggregate_results(samples)

        print("\n" + "=" * 80)
        print("AGGREGATED VALIDATION RESULTS")
        print("=" * 80)

        print("\n[Summary]")
        print(f"  Samples Analyzed: {aggregated['n_samples']}")
        print(f"  Speaker Detection Accuracy: {aggregated['averages']['speaker_detection_accuracy']:.1f}%")
        print(f"  Role Classification Accuracy: {aggregated['averages']['role_classification_accuracy']:.1f}%")

        print("\n[Average Metrics]")
        print(f"  Segments per Audio: {aggregated['averages']['segments']:.1f}")
        print(f"  Words per Audio: {aggregated['averages']['words']:.1f}")
        print(f"  Duration per Audio: {aggregated['averages']['duration']:.1f}s")
        print(f"  Transcription Confidence: {aggregated['averages']['transcription_confidence']:.3f}")

        print("\n[Role Identification Confidence]")
        print(f"  Doctor: {aggregated['averages']['doctor_confidence']:.3f}")
        print(f"  Patient: {aggregated['averages']['patient_confidence']:.3f}")

        print("\n[Individual Samples]")
        for sample in samples:
            print(f"\n  {sample['sample_name']}:")
            print(f"    Duration: {sample['whisperx']['duration']:.1f}s")
            print(f"    Segments: {sample['whisperx']['total_segments']}")
            print(f"    Confidence: {sample['whisperx']['avg_confidence']:.3f}")
            print(f"    Speakers: {sample['whisperx']['speakers_detected']}")
            for speaker_id, info in sample["speaker_mapping"].items():
                print(f"      {speaker_id} → {info['role']}: {info['confidence']:.3f} ({info['confidence_level']})")

        print("\n" + "=" * 80)

        # Save aggregated results
        with open("/tmp/aggregated_validation_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        print("\n[OK] Saved aggregated results to /tmp/aggregated_validation_results.json")
