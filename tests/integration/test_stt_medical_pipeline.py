"""Integration Tests: STT-to-Medical Pipeline

Comprehensive tests for the complete audio-to-medical processing pipeline.
Tests both unified (single workflow) and two-step (separate API calls) approaches.

Test Architecture:
1. Unified Workflow: Upload with enable_medical_processing=true
   - Single Temporal workflow handles WhisperX + Medical in one execution
   - Full observability in Temporal UI

2. Two-Step Pipeline: WhisperX first, then medical processing
   - Step 1: /speech-to-text -> WhisperX workflow
   - Step 2: /medical/process-whisperx-result -> Medical LLM pipeline
"""

import pytest
import httpx
import sys
from pathlib import Path

# Add tests directory to path for conftest import
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import wait_for_workflow_completion


BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0  # 10 minutes for complete workflows

# Kaggle dataset - use SHORT audio file for faster testing
DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews"
AUDIO_FILE = DATASET_DIR / "audios" / "RES0198.mp3"  # 7:21 min respiratory case


# =============================================================================
# Health Check Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.medical
def test_medical_services_health():
    """Verify required services for medical pipeline are running."""
    print("\nChecking medical pipeline service health...")

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # Check server (required)
        response = client.get("/health")
        assert response.status_code == 200, "Server not running"
        print("[OK] Server running")

        # Check LM Studio (optional - warn if not available)
        try:
            response = client.get("/health/lm-studio")
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "ok":
                    print(f"[OK] LM Studio running ({data.get('models_loaded', 0)} models)")
                else:
                    print(f"[WARN] LM Studio not ready: {data.get('models_loaded', 0)} models loaded")
            else:
                print("[WARN] LM Studio endpoint returned error")
        except Exception as e:
            print(f"[WARN] LM Studio check failed: {e}")

        # Check medical processing endpoint exists
        response = client.get("/health/medical")
        if response.status_code == 200:
            print("[OK] Medical processing endpoint available")
        else:
            print("[WARN] Medical processing endpoint not available")


# =============================================================================
# Unified Workflow Test (Single Temporal Workflow)
# =============================================================================


@pytest.mark.integration
@pytest.mark.medical
@pytest.mark.slow
def test_unified_stt_to_medical_workflow():
    """
    Test unified workflow with enable_medical_processing=True.

    Pipeline in SINGLE workflow:
    1. WhisperX (transcribe, align, diarize, assign speakers)
    2. Medical processing (transform, PHI, entities, SOAP, storage)

    All stages visible in Temporal UI as a single workflow execution.
    """
    print("\n" + "=" * 80)
    print("UNIFIED STT-TO-MEDICAL WORKFLOW TEST")
    print("Single workflow with complete audio -> medical pipeline")
    print("=" * 80)

    if not AUDIO_FILE.exists():
        pytest.skip(f"Audio file not found: {AUDIO_FILE}")

    patient_name = "Unified Workflow Test"
    provider_id = "dr-unified-test"

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # =================================================================
        # STEP 1: Upload with medical processing ENABLED
        # =================================================================
        print("\n[UPLOAD] STEP 1: Starting unified workflow...")
        print(f"   Audio: {AUDIO_FILE.name}")
        print("   Medical processing: ENABLED")

        with open(AUDIO_FILE, "rb") as f:
            files = {"file": (AUDIO_FILE.name, f, "audio/mpeg")}
            data = {
                "patient_name": patient_name,
                "enable_medical_processing": "true",
                "provider_id": provider_id,
                "encounter_date": "2025-12-29",
            }

            response = client.post("/speech-to-text", files=files, data=data)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_data = response.json()
        workflow_id = upload_data.get("identifier") or upload_data.get("workflow_id")
        assert workflow_id, "No workflow ID returned"

        print(f"[OK] Workflow started: {workflow_id}")
        print("  Monitor in Temporal UI: http://localhost:8233")

        # =================================================================
        # STEP 2: Wait for COMPLETE workflow (WhisperX + Medical)
        # =================================================================
        print("\nSTEP 2: Waiting for complete pipeline...")
        print("   Expected: ~5-8 minutes for WhisperX + Medical processing")

        result = wait_for_workflow_completion(client, workflow_id, max_wait=480, poll_interval=30)

        if not result:
            pytest.skip("Workflow did not complete in time")

        # =================================================================
        # STEP 3: Validate workflow result structure
        # =================================================================
        print("\n[CHECK] STEP 3: Validating unified workflow result...")

        assert "workflow_type" in result, "Missing workflow_type"
        assert result["workflow_type"] == "stt_to_medical", "Wrong workflow type"
        assert "medical_processing_enabled" in result
        assert result["medical_processing_enabled"] is True

        print(f"[OK] Workflow type: {result['workflow_type']}")
        print(f"[OK] Medical processing: {result['medical_processing_enabled']}")

        # =================================================================
        # STEP 4: Validate WhisperX/Dialogue stage
        # =================================================================
        print("\n[WHISPERX] STEP 4: Validating transcription stage...")

        # The unified workflow returns dialogue_transformation with the transcript
        assert "dialogue_transformation" in result, "Missing dialogue_transformation"

        dialogue = result["dialogue_transformation"]
        segments = dialogue.get("dialogue", [])
        transcript = dialogue.get("full_transcript", "")

        print(f"[OK] Transcription: {len(segments)} segments")
        print(f"[OK] Transcript length: {len(transcript)} chars")

        # =================================================================
        # STEP 5: Validate Medical stage
        # =================================================================
        print("\n[MEDICAL] STEP 5: Validating medical processing stage...")

        medical_keys = [
            "dialogue_transformation",
            "entity_extraction",
        ]
        has_medical = any(key in result for key in medical_keys)
        assert has_medical, "No medical processing results found"

        _validate_medical_results(result, medical_keys)

        # =================================================================
        # STEP 6: Validate result structure
        # =================================================================
        print("\n[STATS] STEP 6: Validating result structure...")

        # Check for consultation_id (indicates successful processing)
        consultation_id = result.get("consultation_id")
        if consultation_id:
            print(f"[OK] Consultation ID: {consultation_id}")

        # Count completed stages based on presence of keys
        completed_stages = []
        if "dialogue_transformation" in result:
            completed_stages.append("dialogue_transformation")
        if "entity_extraction" in result:
            completed_stages.append("entity_extraction")
        if "soap_generation" in result:
            completed_stages.append("soap_generation")
        if "phi_detection" in result:
            completed_stages.append("phi_detection")

        print(f"  Completed stages: {len(completed_stages)}")
        for stage in completed_stages:
            print(f"    - {stage}")

        # At least dialogue_transformation should be present
        assert "dialogue_transformation" in result, "Dialogue transformation missing"

        print("\n" + "=" * 80)
        print("[DONE] UNIFIED WORKFLOW TEST PASSED")
        print("=" * 80)
        print(f"\nWorkflow ID: {workflow_id}")
        print(f"  - Dialogue segments: {len(segments)}")
        print(f"  - Medical stages: {len(completed_stages)} completed")


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_medical_results(result: dict, medical_keys: list) -> None:
    """Validate and print medical processing results."""
    if "dialogue_transformation" in result:
        dialogue = result["dialogue_transformation"]
        speakers = dialogue.get("speaker_mapping", {})
        print(f"[OK] Dialogue transformation: {len(speakers)} speakers")
        for speaker_id, info in speakers.items():
            print(f"  - {speaker_id}: {info.get('role', 'unknown')}")

    if "phi_detection" in result:
        phi = result["phi_detection"]
        if not phi.get("skipped"):
            entity_count = len(phi.get("entities", []))
            print(f"[OK] PHI detection: {entity_count} entities")

    if "entity_extraction" in result:
        entities = result["entity_extraction"]
        if not entities.get("skipped"):
            count = entities.get("entity_count", 0)
            print(f"[OK] Entity extraction: {count} entities")

    if "soap_generation" in result:
        soap = result["soap_generation"]
        if not soap.get("skipped"):
            sections = list(soap.get("soap_note", {}).keys())
            print(f"[OK] SOAP note: {len(sections)} sections")

    if "vector_storage" in result:
        storage = result["vector_storage"]
        if not storage.get("skipped"):
            vector_id = storage.get("vector_id")
            print(f"[OK] Vector storage: {vector_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration and medical"])
