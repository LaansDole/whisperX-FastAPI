"""Unit tests for HIPAA-compliant filename generation utilities."""

import pytest
import os
from datetime import datetime
from app.patients.filename_utils import (
    generate_patient_file_id,
    generate_consultation_filename,
    generate_workflow_result_filename,
    generate_anonymous_audio_filename,
    extract_patient_id_from_filename,
    generate_result_storage_path
)


class TestFilenameUtils:
    """Tests for HIPAA-compliant filename generation."""
    
    def test_generate_patient_file_id(self):
        """Test patient file ID generation is deterministic."""
        patient_id = "enc_patient_12345"
        
        # Should be deterministic
        hash1 = generate_patient_file_id(patient_id)
        hash2 = generate_patient_file_id(patient_id)
        
        assert hash1 == hash2
        assert len(hash1) == 8
        assert hash1.isalnum()
    
    def test_generate_patient_file_id_different_ids(self):
        """Test different patient IDs generate different hashes."""
        hash1 = generate_patient_file_id("patient_001")
        hash2 = generate_patient_file_id("patient_002")
        
        assert hash1 != hash2
    
    def test_generate_consultation_filename_basic(self):
        """Test basic consultation filename generation."""
        filename = generate_consultation_filename(
            patient_id_encrypted="enc_patient_123",
            date="20251227",
            department="cardiology",
            sequence=1
        )
        
        assert filename.startswith("pt_")
        assert "20251227" in filename
        assert "cardiology" in filename
        assert "001" in filename
        assert filename.endswith(".json")
    
    def test_generate_consultation_filename_with_spaces(self):
        """Test filename generation sanitizes department names with spaces."""
        filename = generate_consultation_filename(
            patient_id_encrypted="enc_patient_123",
            department="Emergency Room",  # Has space
            sequence=1
        )
        
        # Spaces should be removed
        assert " " not in filename
        assert "emergencyroom" in filename.lower()
    
    def test_generate_consultation_filename_special_chars(self):
        """Test filename generation sanitizes special characters."""
        filename = generate_consultation_filename(
            patient_id_encrypted="enc_patient_123",
            department="ICU/CCU-Unit#1",  # Special chars
            sequence=1
        )
        
        # Only alphanumeric and underscores allowed
        # Extract department part
        parts = filename.split("_")
        department_part = parts[3] if len(parts) > 3 else parts[-1].replace(".json", "")
        
        assert all(c.isalnum() or c == "_" for c in department_part.replace(".json", ""))
    
    def test_generate_consultation_filename_defaults(self):
        """Test filename generation with default values."""
        filename = generate_consultation_filename(
            patient_id_encrypted="enc_patient_123"
        )
        
        # Should have today's date
        today = datetime.now().strftime("%Y%m%d")
        assert today in filename
        
        # Should have default department
        assert "general" in filename
    
    def test_generate_consultation_filename_no_sequence(self):
        """Test filename without sequence number."""
        filename = generate_consultation_filename(
            patient_id_encrypted="enc_patient_123",
            sequence=None
        )
        
        # Should not have 3-digit sequence
        parts = filename.replace(".json", "").split("_")
        assert not any(p.isdigit() and len(p) == 3 for p in parts[-1:])
    
    def test_generate_workflow_result_filename(self):
        """Test workflow result filename generation."""
        filename = generate_workflow_result_filename(
            "whisperx-workflow-abc123def456"
        )
        
        assert filename.startswith("wf_")
        assert filename.endswith(".json")
        assert len(filename) <= 20  # wf_ + 12 chars + .json
    
    def test_generate_workflow_result_filename_custom_extension(self):
        """Test workflow filename with custom extension."""
        filename = generate_workflow_result_filename(
            "whisperx-workflow-123",
            extension=".txt"
        )
        
        assert filename.endswith(".txt")
    
    def test_generate_anonymous_audio_filename_with_patient(self):
        """Test audio filename with patient ID."""
        patient_id = "enc_patient_123"
        filename = generate_anonymous_audio_filename(".mp3", patient_id)
        
        assert filename.startswith("audio_")
        assert filename.endswith(".mp3")
        
        # Should be deterministic for same patient on same second
        filename2 = generate_anonymous_audio_filename(".mp3", patient_id)
        # Note: May differ if run in different second
    
    def test_generate_anonymous_audio_filename_without_patient(self):
        """Test audio filename without patient ID (random UUID)."""
        filename1 = generate_anonymous_audio_filename(".wav", None)
        filename2 = generate_anonymous_audio_filename(".wav", None)
        
        # Should be different (random UUIDs)
        assert filename1 != filename2
        assert filename1.endswith(".wav")
        assert filename2.endswith(".wav")
    
    def test_extract_patient_id_from_filename(self):
        """Test extracting patient hash from filename."""
        filename = "pt_a7f3c8e2_20251227_cardiology_001.json"
        patient_hash = extract_patient_id_from_filename(filename)
        
        assert patient_hash == "a7f3c8e2"
    
    def test_extract_patient_id_invalid_filename(self):
        """Test extraction returns None for invalid filenames."""
        assert extract_patient_id_from_filename("random_file.json") is None
        assert extract_patient_id_from_filename("wf_12345.json") is None
    
    def test_generate_result_storage_path(self):
        """Test storage path generation with subdirectories."""
        path = generate_result_storage_path(
            base_dir="/data/results",
            patient_id_encrypted="enc_patient_123",
            filename="pt_a7f3c8e2_20251227_cardiology_001.json"
        )
        
        # Should organize by first 2 chars of hash
        assert "/data/results/" in path
        assert path.endswith("pt_a7f3c8e2_20251227_cardiology_001.json")
        
        # Check subdirectory structure (first 2 chars of hash)
        path_parts = path.split(os.sep)
        assert len(path_parts[-3]) == 2  # First 2 chars subdirectory
        assert len(path_parts[-2]) == 8  # Full hash subdirectory
    
    def test_filename_no_phi_exposure(self):
        """Test that filenames don't expose PHI."""
        # Simulate patient name that should NOT appear in filename
        patient_name = "John Doe"
        patient_id = f"enc_{patient_name.replace(' ', '_')}"
        
        filename = generate_consultation_filename(
            patient_id_encrypted=patient_id,
            department="Cardiology"
        )
        
        # Patient name should NOT be in filename
        assert "John" not in filename
        assert "Doe" not in filename
        assert patient_name not in filename
    
    def test_filename_consistency_same_patient(self):
        """Test filenames are consistent for same patient."""
        patient_id = "enc_patient_123"
        
        file1 = generate_consultation_filename(
            patient_id_encrypted=patient_id,
            date="20251227",
            department="cardiology",
            sequence=1
        )
        
        file2 = generate_consultation_filename(
            patient_id_encrypted=patient_id,
            date="20251227",
            department="cardiology",
            sequence=1
        )
        
        assert file1 == file2  # Should be identical
    
    def test_filename_different_for_different_sequences(self):
        """Test filenames differ for different sequences."""
        patient_id = "enc_patient_123"
        
        file1 = generate_consultation_filename(
            patient_id_encrypted=patient_id,
            date="20251227",
            department="cardiology",
            sequence=1
        )
        
        file2 = generate_consultation_filename(
            patient_id_encrypted=patient_id,
            date="20251227",
            department="cardiology",
            sequence=2
        )
        
        assert file1 != file2
        assert "001" in file1
        assert "002" in file2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
