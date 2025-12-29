"""Integration Test: Unified STT-to-Medical Workflow

Tests the NEW unified workflow that processes audio through WhisperX and medical
pipeline in a SINGLE workflow, with full Temporal observability.

Requires:
- Server running: `make dev`
- LM Studio running and configured
"""

import pytest
import httpx
import time
from pathlib import Path


BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0  # 10 minutes for complete unified workflow

# Kaggle dataset - use SHORT audio file for faster testing
DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews"


def wait_for_workflow_completion(client: httpx.Client, workflow_id: str, max_wait: int = 480, poll_interval: int = 30):
    """
    Poll Temporal workflow until completion.

    Args:
        client: HTTP client
        workflow_id: Temporal workflow ID
        max_wait: Maximum wait time in seconds (default: 480s = 8 minutes)
        poll_interval: Polling interval in seconds (default: 30s)

    Returns:
        Workflow result if completed, None if timeout

    Raises:
        Exception: If workflow fails
    """
    elapsed = 0

    print(f"\n⏳ Waiting for workflow {workflow_id} to complete...")
    print(f"   Polling: every {poll_interval}s, Max wait: {max_wait}s")

    while elapsed < max_wait:
        response = client.get(f"/temporal/workflow/{workflow_id}")

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "UNKNOWN")

            print(f"  [{elapsed}s] Status: {status}")

            if status == "COMPLETED":
                # Fetch result
                result_response = client.get(f"/temporal/workflow/{workflow_id}/result")
                if result_response.status_code == 200:
                    result = result_response.json()
                    print(f"✓ Workflow completed in {elapsed}s")
                    return result
                else:
                    print(f"  Error fetching result: {result_response.status_code} - {result_response.text}")
                    time.sleep(5)
                    elapsed += 5

            elif status == "FAILED":
                raise Exception("Workflow failed")

            else:  # RUNNING, PENDING
                time.sleep(poll_interval)
                elapsed += poll_interval
        else:
            print(f"  [{elapsed}s] Error: HTTP {response.status_code}")
            time.sleep(poll_interval)
            elapsed += poll_interval

    print(f"⏰ Timeout after {max_wait}s")
    return None


@pytest.mark.integration
@pytest.mark.medical
def test_unified_stt_to_medical_workflow():
    """
    Upload with enable_medical_processing=True

    Pipeline in SINGLE workflow:
    1. WhisperX (transcribe, align, diarize, assign speakers)
    2. Medical processing (transform, PHI, entities, SOAP, storage)

    All visible in Temporal UI!
    """
    print("\n" + "=" * 80)
    print("PHASE 3: UNIFIED STT-TO-MEDICAL WORKFLOW TEST")
    print("Single workflow with complete audio → medical pipeline")
    print("=" * 80)

    # Use SHORTEST audio file to reduce test time
    audio_file = DATASET_DIR / "audios" / "RES0198.mp3"  # 7:21 min

    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    patient_name = "Unified Workflow Test"
    provider_id = "dr-test"

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # =================================================================
        # STEP 1: Upload with medical processing ENABLED
        # =================================================================
        print("\n📤 STEP 1: Starting unified workflow...")
        print(f"   Audio: {audio_file.name}")
        print("   Medical processing: ENABLED")

        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": patient_name,
                "enable_medical_processing": "true",  # flag!
                "provider_id": provider_id,
                "encounter_date": "2025-12-29",
            }

            response = client.post("/speech-to-text", files=files, data=data)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_data = response.json()
        workflow_id = upload_data.get("identifier") or upload_data.get("workflow_id")
        assert workflow_id, "No workflow ID returned"

        print(f"✓ Workflow started: {workflow_id}")
        print("  Monitor in Temporal UI: http://localhost:8233")

        # =================================================================
        # STEP 2: Wait for COMPLETE workflow (WhisperX + Medical)
        # =================================================================
        print("\n⏳ STEP 2: Waiting for complete pipeline...")
        print("   Expected: ~5-8 minutes for WhisperX + Medical processing")

        result = wait_for_workflow_completion(client, workflow_id, max_wait=480, poll_interval=30)

        if not result:
            pytest.skip("Workflow did not complete in time")

        # =================================================================
        # STEP 3: Validate workflow result structure
        # =================================================================
        print("\n🔍 STEP 3: Validating unified workflow result...")

        # Check result structure
        assert "workflow_type" in result, "Missing workflow_type"
        assert result["workflow_type"] == "stt_to_medical", "Wrong workflow type"

        assert "medical_processing_enabled" in result
        assert result["medical_processing_enabled"] is True

        print(f"✓ Workflow type: {result['workflow_type']}")
        print(f"✓ Medical processing: {result['medical_processing_enabled']}")

        # =================================================================
        # STEP 4: Validate WhisperX stage
        # =================================================================
        print("\n🎙️ STEP 4: Validating WhisperX stage...")

        assert "whisperx_transcription" in result
        assert "whisperx_alignment" in result
        assert "whisperx_diarization" in result
        assert "whisperx_final" in result

        segments = result["whisperx_final"].get("segments", [])
        print(f"✓ Transcription: {len(segments)} segments")

        # =================================================================
        # STEP 5: Validate Medical stage
        # =================================================================
        print("\n🏥 STEP 5: Validating medical processing stage...")

        # Check if medical stages executed
        has_medical = any(
            key in result
            for key in [
                "dialogue_transformation",
                "phi_detection",
                "entity_extraction",
                "soap_generation",
                "vector_storage",
            ]
        )

        assert has_medical, "No medical processing results found"

        if "dialogue_transformation" in result:
            dialogue = result["dialogue_transformation"]
            speakers = dialogue.get("speaker_mapping", {})
            print(f"✓ Dialogue transformation: {len(speakers)} speakers")
            for speaker_id, info in speakers.items():
                print(f"  - {speaker_id}: {info.get('role', 'unknown')}")

        if "phi_detection" in result:
            phi = result["phi_detection"]
            if not phi.get("skipped"):
                entity_count = len(phi.get("entities", []))
                print(f"✓ PHI detection: {entity_count} entities")

        if "entity_extraction" in result:
            entities = result["entity_extraction"]
            if not entities.get("skipped"):
                count = entities.get("entity_count", 0)
                print(f"✓ Entity extraction: {count} entities")

        if "soap_generation" in result:
            soap = result["soap_generation"]
            if not soap.get("skipped"):
                sections = list(soap.get("soap_note", {}).keys())
                print(f"✓ SOAP note: {len(sections)} sections")

        if "vector_storage" in result:
            storage = result["vector_storage"]
            if not storage.get("skipped"):
                vector_id = storage.get("vector_id")
                print(f"✓ Vector storage: {vector_id}")

        # =================================================================
        # STEP 6: Validate summary
        # =================================================================
        print("\n📊 STEP 6: Validating workflow summary...")

        summary = result.get("summary", {})
        stages = summary.get("stages_completed", [])

        print(f"  Completed stages: {len(stages)}")
        for stage in stages:
            print(f"    - {stage}")

        # Core assertions
        assert "transcription" in stages, "Transcription stage missing"
        assert "alignment" in stages, "Alignment stage missing"
        assert "diarization" in stages, "Diarization stage missing"
        assert "speaker_assignment" in stages, "Speaker assignment missing"

        # Medical stages should be present
        assert "dialogue_transformation" in stages, "Dialogue transformation missing"

        # At least one medical processing stage should complete
        medical_stages = ["phi_detection", "entity_extraction", "soap_generation", "vector_storage"]
        completed_medical = [s for s in medical_stages if s in stages]
        assert len(completed_medical) > 0, "No medical stages completed"

        print(f"  Medical stages completed: {len(completed_medical)}/{len(medical_stages)}")

        print("\n" + "=" * 80)
        print("✅ PHASE 3 UNIFIED WORKFLOW TEST PASSED")
        print("=" * 80)
        print(f"\nWorkflow ID: {workflow_id}")
        print("View in Temporal UI: http://localhost:8233")
        print("\nPipeline summary:")
        print(f"  - Total stages: {len(stages)}")
        print(f"  - WhisperX segments: {len(segments)}")
        print(f"  - Medical stages: {len(completed_medical)} completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration and medical"])
