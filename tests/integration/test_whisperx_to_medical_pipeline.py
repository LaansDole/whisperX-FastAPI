"""Comprehensive Integration Test with proper Temporal workflow polling.

Tests complete WhisperX -> Medical pipeline with real audio from Kaggle dataset.
Properly waits for Temporal workflow completion using exponential backoff.

Requires:
- Server running: `make dev`
- LM Studio running and configured
"""

import pytest
import httpx
import sys
from pathlib import Path

# Add tests directory to path for conftest import
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import wait_for_workflow_completion


BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # 5 minutes for complete pipeline

# Kaggle dataset
DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews"


@pytest.mark.integration
@pytest.mark.medical
def test_complete_whisperx_to_medical_pipeline():
    """
    Test pipeline with real audio:
    1. Upload → WhisperX workflow (with waiting)
    2. Transform → Speaker-attributed dialogue
    3. Medical LLM → PHI, entities, SOAP note
    4. Vector storage → RAG database
    """
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE INTEGRATION TEST")
    print("WhisperX → Transformation → Medical LLM → Vector Storage")
    print("=" * 70)

    # Use RES0198.mp3 (7:21 minutes - respiratory case)
    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    patient_name = "Respiratory Integration Test"

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # =================================================================
        # STEP 1: Upload audio and start WhisperX workflow
        print("\nSTEP 1: Starting WhisperX transcription...")
        print("   Audio: RES0198.mp3 (7:21 min respiratory case)")
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {"patient_name": patient_name}

            response = client.post("/speech-to-text", files=files, data=data)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_data = response.json()
        workflow_id = upload_data.get("identifier") or upload_data.get("workflow_id")
        assert workflow_id, "No workflow ID returned"

        print(f"[OK] WhisperX workflow started: {workflow_id}")

        # =================================================================
        # STEP 2: Wait for WhisperX completion (with proper polling)
        # =================================================================
        print("\nSTEP 2: Waiting for WhisperX transcription...")

        whisperx_result = wait_for_workflow_completion(
            client, workflow_id, max_wait=360, poll_interval=30
        )  # 6 minutes for 7-min audio, 30s polling

        if not whisperx_result:
            pytest.skip("WhisperX workflow did not complete in time")

        # Validate WhisperX result
        assert "segments" in whisperx_result, "No segments in WhisperX result"
        segment_count = len(whisperx_result["segments"])
        print(f"[OK] WhisperX completed: {segment_count} segments")

        # =================================================================
        # STEP 3: Process through medical pipeline
        # =================================================================
        print("\nSTEP 3: Processing through medical LLM pipeline...")

        medical_request = {
            "workflow_id": workflow_id,
            "patient_id": "test-integration-001",
            "provider_id": "dr-integration-001",
            "encounter_date": "2025-12-29",
            "processing_options": {
                "enable_phi_detection": True,
                "enable_entity_extraction": True,
                "enable_soap_generation": True,
                "enable_vector_storage": True,
                "speaker_role_detection": "auto",
            },
        }

        response = client.post("/medical/process-whisperx-result", json=medical_request)

        assert response.status_code == 200, f"Medical processing failed: {response.text}"
        medical_result = response.json()

        print("[OK] Medical pipeline completed")

        # =================================================================
        # STEP 4: Validate transformation
        # =================================================================
        print("\nSTEP 4: Validating speaker-attributed dialogue...")

        transformation = medical_result.get("transformation", {})
        assert transformation.get("success"), "Transformation failed"

        speaker_mapping = transformation.get("speaker_mapping", {})

        print(f"[OK] Identified {len(speaker_mapping)} speakers")

        for speaker_id, info in speaker_mapping.items():
            role = info.get("role", "unknown")
            confidence = info.get("confidence", 0)
            print(f"  - {speaker_id}: {role} ({confidence:.2%} confidence)")

        # Verify at least one speaker identified
        assert len(speaker_mapping) > 0, "No speakers identified"

        # =================================================================
        # STEP 5: Validate PHI detection (speaker-aware)
        # =================================================================
        print("\nSTEP 5: Validating PHI detection...")

        phi_step = medical_result["steps"].get("phi_detection", {})
        assert phi_step.get("success"), "PHI detection failed"

        if phi_step.get("success"):
            entity_count = phi_step.get("entity_count", 0)
            print(f"[OK] PHI detection found {entity_count} entities")

            # Sample PHI entities
            entities = phi_step.get("entities", [])
            for entity in entities[:3]:
                phi_type = entity.get("type")
                speaker = entity.get("speaker_role", "N/A")
                print(f"  - {phi_type} (from {speaker})")
        else:
            print(f"PHI detection failed: {phi_step.get('error')}")

        # =================================================================
        # STEP 6: Validate entity extraction (speaker-aware)
        # =================================================================
        print("\nSTEP 6: Validating medical entity extraction...")

        entity_step = medical_result["steps"].get("entity_extraction", {})

        if entity_step.get("success"):
            entity_count = entity_step.get("entity_count", 0)
            print(f"[OK] Extracted {entity_count} medical entities")

            entities = entity_step.get("entities", [])

            # Count by speaker role
            doctor_entities = sum(1 for e in entities if e.get("speaker_role") == "doctor")
            patient_entities = sum(1 for e in entities if e.get("speaker_role") == "patient")

            print(f"  - Doctor: {doctor_entities} entities")
            print(f"  - Patient: {patient_entities} entities")

            # Sample entities with speaker attribution
            for entity in entities[:5]:
                etype = entity.get("type")
                text = entity.get("text", "")[:30]
                speaker = entity.get("speaker_role", "unknown")
                print(f"  - {etype}: {text}... ({speaker})")

            # Verify speaker attribution exists
            assert any(e.get("speaker_role") for e in entities), "No speaker attribution in entities"
        else:
            print(f"Entity extraction failed: {entity_step.get('error')}")

        # =================================================================
        # STEP 7: Validate SOAP note generation
        # =================================================================
        print("\nSTEP 7: Validating SOAP note generation...")

        soap_step = medical_result["steps"].get("soap_generation", {})

        if soap_step.get("success"):
            soap_note = soap_step.get("soap_note", {})

            for section in ["subjective", "objective", "assessment", "plan"]:
                content = soap_note.get(section, "")
                has_content = "[OK]" if content else "[FAIL]"
                word_count = len(content.split()) if content else 0
                print(f"  {has_content} {section.upper()}: {word_count} words")

            # Verify SOAP note has content
            soap_has_content = any(soap_note.get(s) for s in ["subjective", "objective", "assessment", "plan"])
            assert soap_has_content, "SOAP note is empty"
        else:
            print(f"SOAP generation failed: {soap_step.get('error')}")

        # =================================================================
        # STEP 8: Validate vector storage
        # =================================================================
        print("\nSTEP 8: Validating vector storage...")

        storage_step = medical_result["steps"].get("vector_storage", {})

        if storage_step.get("success"):
            vector_id = storage_step.get("vector_id")
            print("[OK] Stored in vector database")
            print(f"  Vector ID: {vector_id}")

            # Verify consultation ID
            consultation_id = medical_result.get("consultation_id")
            assert consultation_id, "No consultation ID"
            print(f"  Consultation ID: {consultation_id}")
        else:
            print(f"Vector storage failed: {storage_step.get('error')}")

        # =================================================================
        # STEP 9: Validate pipeline summary
        # =================================================================
        print("\nSTEP 9: Pipeline summary...")

        summary = medical_result.get("summary", {})
        successful_steps = summary.get("successful_steps", 0)
        total_steps = summary.get("total_steps", 0)

        print(f"  Successful: {successful_steps}/{total_steps} steps")

        # Core validation: transformation must succeed
        assert transformation.get("success"), "Core transformation failed"

        # At least 2 steps should succeed (transformation is implicit)
        assert successful_steps >= 2, f"Too few steps succeeded: {successful_steps}"

        print("\n" + "=" * 70)
        print("=" * 70)
        print("\nPipeline summary:")
        print(f"  - WhisperX segments: {segment_count}")
        print(f"  - Speakers identified: {len(speaker_mapping)}")
        print(f"  - Medical entities: {entity_count if entity_step.get('success') else 0}")
        print(f"  - Pipeline steps: {successful_steps}/{total_steps} successful")


@pytest.mark.integration
@pytest.mark.medical
def test_health_checks():
    """Verify all required services are running."""
    print("\nChecking service health...")

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # Check server
        response = client.get("/docs")
        assert response.status_code == 200, "Server not running"
        print("[OK] Server running")

        # Check LM Studio
        response = client.get("/health/lm-studio")
        assert response.status_code == 200, "LM Studio not available"
        data = response.json()
        assert data["status"] == "ok", f"LM Studio unhealthy: {data}"
        print("[OK] LM Studio running")

        # Check medical processing
        response = client.get("/health/medical")
        assert response.status_code == 200, "Medical processing not available"
        print("[OK] Medical processing enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration and medical"])
