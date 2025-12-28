"""Unit tests for patient mapping functionality."""

import pytest
import sqlite3
from contextlib import contextmanager
from unittest.mock import patch
from app.patients.mapping import (
    store_patient_workflow,
    get_patient_by_workflow,
    get_workflows_by_patient_hash,
    get_patient_name_by_hash,
)


@pytest.fixture
def test_db():
    """
    Provide an isolated in-memory SQLite database for each test.

    This fixture ensures proper test isolation by:
    - Using an in-memory database (:memory:) unique to each test
    - Automatically creating the schema before each test
    - Preventing race conditions in parallel test execution
    - Avoiding file I/O for faster test execution
    """
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create schema
    cursor.execute("""
        CREATE TABLE patient_workflow_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_hash TEXT NOT NULL,
            workflow_id TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            department TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE INDEX idx_patient_hash 
        ON patient_workflow_mappings(patient_hash)
    """)

    conn.commit()

    # Context manager that returns the test database connection
    @contextmanager
    def get_test_db_connection():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Patch the database connection to use in-memory database
    with patch("app.patients.database.get_db_connection", get_test_db_connection):
        yield conn

    # Cleanup
    conn.close()


@pytest.mark.usefixtures("test_db")
class TestPatientMapping:
    """Test patient-workflow mapping storage and retrieval."""

    def test_store_patient_workflow(self):
        """Test storing patient-workflow mapping."""
        store_patient_workflow(
            patient_name="John Michael Smith",
            patient_hash="154c26a1",
            workflow_id="whisperx-wf-pt_154c26a1-20251227_150000",
            file_path="/tmp/uploads/audio_154c26a1_20251227_150000.mp3",
            department="Cardiology",
        )

        # Verify can retrieve by workflow_id
        mapping = get_patient_by_workflow("whisperx-wf-pt_154c26a1-20251227_150000")
        assert mapping is not None
        assert mapping["patient_name"] == "John Michael Smith"
        assert mapping["patient_hash"] == "154c26a1"
        assert mapping["department"] == "Cardiology"

    def test_get_patient_by_workflow(self):
        """Test retrieving patient info by workflow ID."""
        store_patient_workflow(
            patient_name="Jane Doe", patient_hash="abc12345", workflow_id="workflow-123", file_path="/tmp/test.mp3"
        )

        result = get_patient_by_workflow("workflow-123")

        assert result is not None
        assert result["patient_name"] == "Jane Doe"
        assert result["patient_hash"] == "abc12345"
        assert result["workflow_id"] == "workflow-123"

    def test_get_patient_by_workflow_not_found(self):
        """Test getting non-existent workflow."""
        result = get_patient_by_workflow("nonexistent")
        assert result is None

    def test_get_workflows_by_patient_hash(self):
        """Test retrieving all workflows for a patient."""
        patient_hash = "154c26a1"

        # Store multiple workflows for same patient
        store_patient_workflow(
            patient_name="John Smith", patient_hash=patient_hash, workflow_id="workflow-1", file_path="/tmp/file1.mp3"
        )

        store_patient_workflow(
            patient_name="John Smith", patient_hash=patient_hash, workflow_id="workflow-2", file_path="/tmp/file2.mp3"
        )

        workflows = get_workflows_by_patient_hash(patient_hash)

        assert len(workflows) == 2
        assert all(w["patient_hash"] == patient_hash for w in workflows)
        # SQLite returns in DESC order (most recent first)
        assert workflows[0]["workflow_id"] == "workflow-2"
        assert workflows[1]["workflow_id"] == "workflow-1"

    def test_get_workflows_by_patient_hash_empty(self):
        """Test getting workflows for non-existent patient."""
        workflows = get_workflows_by_patient_hash("nonexistent")
        assert workflows == []

    def test_get_patient_name_by_hash(self):
        """Test getting patient name by hash (admin lookup)."""
        store_patient_workflow(
            patient_name="María García López",
            patient_hash="abc12345",
            workflow_id="workflow-123",
            file_path="/tmp/test.mp3",
        )

        name = get_patient_name_by_hash("abc12345")
        assert name == "María García López"

    def test_get_patient_name_by_hash_not_found(self):
        """Test getting patient name for non-existent hash."""
        name = get_patient_name_by_hash("nonexistent")
        assert name is None

    def test_multiple_patients_different_hashes(self):
        """Test storing workflows for different patients."""
        # Patient 1
        store_patient_workflow(
            patient_name="Alice Johnson", patient_hash="hash111", workflow_id="wf-1", file_path="/tmp/alice.mp3"
        )

        # Patient 2
        store_patient_workflow(
            patient_name="Bob Williams", patient_hash="hash222", workflow_id="wf-2", file_path="/tmp/bob.mp3"
        )

        # Verify both are stored correctly
        alice_name = get_patient_name_by_hash("hash111")
        bob_name = get_patient_name_by_hash("hash222")

        assert alice_name == "Alice Johnson"
        assert bob_name == "Bob Williams"

        # Verify workflow lookups
        alice_wf = get_patient_by_workflow("wf-1")
        bob_wf = get_patient_by_workflow("wf-2")

        assert alice_wf["patient_name"] == "Alice Johnson"
        assert bob_wf["patient_name"] == "Bob Williams"

    def test_patient_name_stored_as_plain_text(self):
        """Verify patient names are stored as plain text (not encrypted)."""
        store_patient_workflow(
            patient_name="Test Patient Name", patient_hash="testhash", workflow_id="test-wf", file_path="/tmp/test.mp3"
        )

        mapping = get_patient_by_workflow("test-wf")

        # Should be plain text, not encrypted/hashed
        assert mapping["patient_name"] == "Test Patient Name"
        assert isinstance(mapping["patient_name"], str)
        assert " " in mapping["patient_name"]  # Contains spaces (not hashed)

    def test_created_at_timestamp(self):
        """Test that created_at timestamp is added automatically."""
        store_patient_workflow(
            patient_name="Test Patient", patient_hash="hash123", workflow_id="wf-123", file_path="/tmp/test.mp3"
        )

        mapping = get_patient_by_workflow("wf-123")

        assert "created_at" in mapping
        # Should be ISO format timestamp
        assert "T" in mapping["created_at"] or "-" in mapping["created_at"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
