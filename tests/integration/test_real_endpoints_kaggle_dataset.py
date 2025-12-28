"""Real endpoint tests with Kaggle dataset and HIPAA-compliant filenames.

Tests actual Temporal workflow execution with real audio files.
"""

import pytest
import requests
import time
import os
from pathlib import Path


BASE_URL = "http://localhost:8000"

# Make dataset path configurable for different environments
DATASET_DIR = (
    Path(os.getenv("KAGGLE_DATASET_DIR"))
    if os.getenv("KAGGLE_DATASET_DIR")
    else (Path(__file__).resolve().parents[2] / "datasets" / "kaggle-simulated-patient-physician-interviews")
)


class TestRealEndpointWithHIPAAFilenames:
    """Test real endpoints with actual audio files and HIPAA filename handling."""

    @pytest.fixture
    def sample_audio_files(self):
        """Get sample audio files from Kaggle dataset."""
        audio_dir = DATASET_DIR / "audios"

        return {
            "cardiology": audio_dir / "CAR0001.mp3",
            "gastro": audio_dir / "GAS0001.mp3",
            "musculoskeletal": audio_dir / "MSK0001.mp3",
        }

    def test_server_running(self):
        """Verify server is accessible."""
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=5)
            assert response.status_code == 200
            print("\n✓ Server is running")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running. Start with 'make dev'")

    def test_speech_to_text_with_patient_cardiology(self, sample_audio_files):
        """Test /speech-to-text endpoint with cardiology patient (spaces in name)."""
        audio_file = sample_audio_files["cardiology"]

        if not audio_file.exists():
            pytest.skip(f"Audio file not found: {audio_file}")

        # Patient name with spaces (as doctor would type)
        patient_name = "John Michael Smith"

        print(f"\n📁 Uploading: {audio_file.name}")
        print(f"👤 Patient: {patient_name}")

        # Upload file with patient_name (plain text)
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            params = {"patient_name": patient_name}  # Plain text name, not encrypted

            response = requests.post(f"{BASE_URL}/speech-to-text", files=files, params=params, timeout=30)

        print(f"Response: {response.status_code}")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        print(f"✓ Response: {data}")

        # Should have workflow ID
        assert "identifier" in data or "workflow_id" in data
        workflow_id = data.get("identifier") or data.get("workflow_id")

        print(f"🔄 Workflow ID: {workflow_id}")

        # Calculate patient hash for verification
        from app.patients.filename_utils import generate_patient_file_id

        patient_hash = generate_patient_file_id(patient_name)

        # Verify HIPAA-compliant workflow ID format
        # New format: whisperx-wf-pt_{hash}-{timestamp}
        assert f"pt_{patient_hash}" in workflow_id, (
            f"Expected patient hash {patient_hash} in workflow ID, got {workflow_id}"
        )
        print(f"✓ Workflow ID contains patient hash: {patient_hash}")

        # Verify patient name NOT in workflow ID
        assert "john" not in workflow_id.lower()
        assert "michael" not in workflow_id.lower()
        assert "smith" not in workflow_id.lower()

        print("✓ Patient name NOT in workflow ID")

        # Wait a bit for workflow to start
        time.sleep(2)

        # Check workflow status
        status_response = requests.get(f"{BASE_URL}/temporal/workflow/{workflow_id}", timeout=10)

        print(f"Workflow Status: {status_response.status_code}")

        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"✓ Workflow State: {status_data.get('status', 'unknown')}")
            print("✓ Workflow started successfully")

            # Test patient-based query
            # Wait a moment for Temporal to index the workflow
            print("\n🔍 Testing patient-based query...")
            print("⏳ Waiting for Temporal to index workflow...")
            time.sleep(2)  # Give Temporal time to index the new workflow

            patient_workflows_response = requests.get(
                f"{BASE_URL}/temporal/patient/{patient_hash}/workflows", timeout=10
            )

            if patient_workflows_response.status_code == 200:
                patient_data = patient_workflows_response.json()
                print(f"✓ Found {patient_data.get('total_found', 0)} workflows for patient")

                # Verify our workflow is in the list
                workflow_ids = [w["workflow_id"] for w in patient_data.get("workflows", [])]
                assert workflow_id in workflow_ids, (
                    f"Workflow {workflow_id} not found in patient query. Found: {workflow_ids}"
                )
                print("✓ Patient query successfully found workflow")
            else:
                print(f"⚠ Patient query endpoint returned {patient_workflows_response.status_code}")
        else:
            print(f"⚠ Could not get workflow status: {status_response.text}")

    def test_speech_to_text_multiple_departments(self, sample_audio_files):
        """Test multiple departments with different patient names - ONE AT A TIME."""
        test_cases = [
            {"audio": "cardiology", "patient": "Sarah Elizabeth Johnson", "department": "Cardiology Department"},
            {"audio": "gastro", "patient": "Robert Lee O'Brien", "department": "Gastroenterology Clinic"},
            {"audio": "musculoskeletal", "patient": "María García López", "department": "Orthopedics/MSK Unit"},
        ]

        workflow_ids = []

        for test_case in test_cases:
            audio_file = sample_audio_files[test_case["audio"]]

            if not audio_file.exists():
                print(f"⚠ Skipping {test_case['audio']}: file not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"Testing: {test_case['department']}")
            print(f"Patient: {test_case['patient']}")
            print(f"Audio: {audio_file.name}")

            # Upload with patient_name (plain text)
            with open(audio_file, "rb") as f:
                files = {"file": (audio_file.name, f, "audio/mpeg")}
                params = {"patient_name": test_case["patient"]}  # Plain text name

                response = requests.post(f"{BASE_URL}/speech-to-text", files=files, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                workflow_id = data.get("identifier") or data.get("workflow_id")
                workflow_ids.append(workflow_id)

                print(f"✓ Workflow started: {workflow_id}")

                # Verify HIPAA-compliant workflow ID
                from app.patients.filename_utils import generate_patient_file_id

                patient_hash = generate_patient_file_id(test_case["patient"])
                assert f"pt_{patient_hash}" in workflow_id, "Expected patient hash in workflow ID"

                # Verify patient name NOT in workflow ID
                for name_part in test_case["patient"].split():
                    assert name_part.lower() not in workflow_id.lower()

                print("✓ HIPAA-compliant workflow ID")
                print("✓ Patient name protected")
            else:
                print(f"✗ Upload failed: {response.status_code}")

            # Wait before next upload to avoid overwhelming the server
            time.sleep(1)

        print(f"\n{'=' * 60}")
        print(f"✅ Tested {len(workflow_ids)} workflows")
        print(f"All workflow IDs: {workflow_ids}")

        assert len(workflow_ids) > 0, "No workflows were started"

    def test_workflow_result_hipaa_filename(self, sample_audio_files):
        """Test that workflow result files use HIPAA-compliant filenames."""
        audio_file = sample_audio_files["cardiology"]

        if not audio_file.exists():
            pytest.skip("Audio file not found")

        patient_name = "Jane Emily Doe"

        print("\n📁 Testing workflow result filename")
        print(f"👤 Patient: {patient_name}")

        # Upload and start workflow with patient_name
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg")}
            params = {"patient_name": patient_name}

            response = requests.post(f"{BASE_URL}/speech-to-text", files=files, params=params, timeout=30)

        if response.status_code != 200:
            pytest.skip("Could not start workflow")

        data = response.json()
        workflow_id = data.get("identifier") or data.get("workflow_id")

        print(f"🔄 Workflow: {workflow_id}")

        # Wait for workflow to complete (or at least start processing)
        time.sleep(5)

        # Check workflow status
        status_response = requests.get(f"{BASE_URL}/temporal/workflow/{workflow_id}", timeout=10)

        if status_response.status_code == 200:
            status_data = status_response.json()
            print("✓ Workflow status retrieved")

            # Check if result contains filename info
            result = status_data.get("result", {})
            result_str = str(result)

            # Verify patient name NOT in any result data
            assert "jane" not in result_str.lower()
            assert "emily" not in result_str.lower()
            assert "doe" not in result_str.lower()

            print("✓ Patient name NOT in workflow result")

    def test_check_uploaded_file_naming(self):
        """Test that uploaded files use HIPAA-compliant naming."""
        uploads_dir = Path("/tmp/uploads")

        if not uploads_dir.exists():
            print("\n⚠ Uploads directory doesn't exist yet")
            return

        files = list(uploads_dir.glob("*"))
        print(f"\n📂 Found {len(files)} files in /tmp/uploads")

        # Sample some recent files
        recent_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]

        for f in recent_files:
            filename = f.name
            print(f"  📄 {filename}")

            # Should not contain obvious patient names
            # (This is a basic check - in production, would check against known names)
            assert " " not in filename, f"Filename contains space: {filename}"
            assert filename.count("-") >= 4 or filename.count("_") >= 1, (
                f"Filename should be UUID or HIPAA format: {filename}"
            )

        print("✓ All filenames appear HIPAA-compliant")

    def test_get_patient_latest_workflow(self):
        """Test getting latest workflow for a patient using patient hash."""
        # Use hash from existing patient
        patient_hash = "02935fa8"  # John Michael Smith hash

        print(f"\nQuerying latest workflow for patient hash: {patient_hash}")

        response = requests.get(f"{BASE_URL}/temporal/patient/{patient_hash}/latest", timeout=10)

        print(f"Response: {response.status_code}")

        # Either 200 (found) or 404 (no workflows)
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}: {response.text}"

        if response.status_code == 200:
            data = response.json()
            print(f"Latest workflow: {data.get('workflow_id')}")
            print(f"Status: {data.get('status')}")

            # Verify structure
            assert "workflow_id" in data
            assert "status" in data
            assert "patient_hash" in data
            assert data["patient_hash"] == patient_hash
            print("Latest workflow endpoint OK")

        elif response.status_code == 404:
            print("No workflows found (expected for new patient)")
    
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
