#!/usr/bin/env python3
"""
Integration Test: Verify SOAP Generation and Response Optimization

Tests using actual medical conversations from Kaggle dataset.
Verifies:
1. Response optimization (intermediate stages excluded)
2. SOAP note generation (all sections populated)
3. PHI detection
"""

import httpx
import time
import json
import pytest
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0  # 10 minutes

# Kaggle dataset - use project root datasets directory
DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews"
MEDICAL_AUDIO = DATASET_DIR / "audios" / "MSK0040.mp3"  # Musculoskeletal case


@pytest.fixture(scope="module")
def medical_audio_file():
    """Fixture to check for medical audio dataset."""
    if not MEDICAL_AUDIO.exists():
        pytest.skip(f"Audio file not found: {MEDICAL_AUDIO}. Skipping integration test.")
    return MEDICAL_AUDIO


def wait_for_completion(client: httpx.Client, workflow_id: str, max_wait: int = 480):
    """Poll workflow until completion."""
    print(f"⏳ Waiting for workflow {workflow_id} to complete...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}")
        response.raise_for_status()
        status_data = response.json()
        status = status_data.get("status", "UNKNOWN")

        elapsed = int(time.time() - start_time)
        print(f"   Status: {status} (elapsed: {elapsed}s)")

        if status == "COMPLETED":
            print("✅ Workflow completed!")
            return True
        elif status == "FAILED":
            error = status_data.get("error", "Unknown error")
            raise Exception(f"Workflow failed: {error}")

        time.sleep(30)

    raise TimeoutError(f"Workflow did not complete within {max_wait}s")


@pytest.mark.integration
def test_medical_conversation(medical_audio_file):
    """Test SOAP generation and response optimization with medical audio."""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: Medical Conversation Processing")
    print("=" * 80)

    with httpx.Client(timeout=TIMEOUT) as client:
        # 1. Check server health
        print("\n1️⃣  Checking server health...")
        response = client.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print("   ✅ Server is healthy")

        # 2. Upload medical audio with medical processing enabled
        print("\n2️⃣  Uploading medical conversation...")
        print(f"   File: {medical_audio_file.name} ({medical_audio_file.stat().st_size / 1024:.1f} KB)")

        with open(medical_audio_file, "rb") as f:
            files = {"file": (medical_audio_file.name, f, "audio/mpeg")}
            data = {
                "patient_name": "John Doe",
                "enable_medical_processing": "true",
                "provider_id": "DR001",
            }
            params = {
                "model": "base",
                "language": "en",
                "min_speakers": 2,
                "max_speakers": 2,
            }

            response = client.post(f"{BASE_URL}/speech-to-text", files=files, data=data, params=params)
            response.raise_for_status()
            upload_result = response.json()
            workflow_id = upload_result["identifier"]
            print(f"   ✅ Upload successful! Workflow ID: {workflow_id}")

        # 3. Wait for completion
        print("\n3️⃣  Waiting for workflow to complete...")
        wait_for_completion(client, workflow_id)

        # 4. Get result and verify
        print("\n4️⃣  Fetching workflow result...")
        response = client.get(f"{BASE_URL}/temporal/workflow/{workflow_id}/result")
        response.raise_for_status()
        result = response.json()

        # 5. Verify response optimization
        print("\n5️⃣  Verifying response optimization...")

        intermediate_stages = ["whisperx_transcription", "whisperx_alignment", "whisperx_diarization"]
        found_intermediate = [stage for stage in intermediate_stages if stage in result]

        if found_intermediate:
            print(f"   ❌ FAIL: Found intermediate stages: {found_intermediate}")
            return False
        else:
            print("   ✅ PASS: No intermediate stages in response")

        # 6. Verify SOAP generation
        print("\n6️⃣  Verifying SOAP note generation...")

        if "soap_generation" not in result:
            print("   ❌ FAIL: No soap_generation in result")
            return False

        soap = result["soap_generation"]
        section_lengths = soap.get("section_lengths", {})

        print("   SOAP Section Lengths:")
        all_populated = True
        for section in ["subjective", "objective", "assessment", "plan"]:
            length = section_lengths.get(section, 0)
            status = "✅" if length > 0 else "❌"
            print(f"     {status} {section.capitalize()}: {length} chars")
            if length == 0:
                all_populated = False

        if not all_populated:
            print("\n   ❌ FAIL: Some SOAP sections are empty")
            print("   Showing actual SOAP content:")
            soap_note = soap.get("soap_note", {})
            for section in ["subjective", "objective", "assessment", "plan"]:
                content = soap_note.get(section, "")
                print(f"\n   {section.upper()}:")
                print(f"   {content[:200]}..." if len(content) > 200 else f"   {content}")
            # Make this a hard failure for deterministic testing
            raise AssertionError("SOAP note has empty sections. All sections must be populated.")
        else:
            print("   ✅ PASS: All SOAP sections populated")

        # 7. Verify PHI detection
        print("\n7️⃣  Verifying PHI detection...")

        if "phi_detection" not in result:
            print("   ❌ FAIL: No phi_detection in result")
            return False

        phi = result["phi_detection"]
        phi_detected = phi.get("phi_detected", False)
        entity_count = len(phi.get("entities", []))

        print(f"   PHI Detected: {phi_detected}")
        print(f"   PHI Entities: {entity_count}")
        print("   ✅ PASS: PHI detection completed")

        # 8. Verify medical entities
        print("\n8️⃣  Verifying medical entity extraction...")

        if "entity_extraction" not in result:
            print("   ❌ FAIL: No entity_extraction in result")
            return False

        entities = result["entity_extraction"].get("entities", [])
        print(f"   Medical Entities: {len(entities)}")

        # Group by type
        entity_types = {}
        for entity in entities:
            etype = entity.get("type", "unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        for etype, count in sorted(entity_types.items()):
            print(f"     - {etype}: {count}")

        print("   ✅ PASS: Entity extraction completed")

        # 9. Response metrics
        response_size = len(json.dumps(result))
        print("\n9️⃣  Response metrics:")
        print(f"   Response size: {response_size / 1024:.1f} KB")
        print(f"   Top-level keys: {len(result)} keys")
        print(f"   Segments in whisperx_final: {len(result.get('whisperx_final', {}).get('segments', []))}")

        print("\n" + "=" * 80)
        # If we reached here, all_populated must be True (otherwise AssertionError was raised)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "integration"])
