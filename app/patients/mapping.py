"""Patient workflow mapping with SQLite persistence."""

from datetime import datetime
from typing import Optional
from ..config import Config
from .database import (
    store_patient_workflow_db,
    get_patient_by_workflow_db,
    get_workflows_by_patient_hash_db,
    get_patient_name_by_hash_db,
    get_all_patients_db,
    reserve_workflow_mapping_db,
    commit_workflow_mapping_db,
    rollback_workflow_mapping_db,
)


def store_patient_workflow(
    patient_name: str, patient_hash: str, workflow_id: str, file_path: str, department: Optional[str] = None
):
    """
    Store patient-workflow mapping.

    Args:
        patient_name: Plain text patient name (stored securely in DB)
        patient_hash: 8-char hash used in filenames/workflow IDs
        workflow_id: Temporal workflow ID
        file_path: Path to audio file
        department: Optional department name
    """
    created_at = datetime.now(Config.TIMEZONE).isoformat()
    store_patient_workflow_db(
        patient_name=patient_name,
        patient_hash=patient_hash,
        workflow_id=workflow_id,
        file_path=file_path,
        department=department,
        created_at=created_at,
    )


def get_patient_by_workflow(workflow_id: str) -> Optional[dict]:
    """
    Get patient info by workflow ID.

    Args:
        workflow_id: Workflow ID

    Returns:
        Patient mapping or None
    """
    return get_patient_by_workflow_db(workflow_id)


def get_workflows_by_patient_hash(patient_hash: str) -> list:
    """
    Get all workflows for a patient by hash.

    Args:
        patient_hash: 8-char patient hash

    Returns:
        List of workflow mappings
    """
    return get_workflows_by_patient_hash_db(patient_hash)


def get_patient_name_by_hash(patient_hash: str) -> Optional[str]:
    """
    Get patient name by hash (admin lookup).

    Args:
        patient_hash: 8-char patient hash

    Returns:
        Plain text patient name or None
    """
    return get_patient_name_by_hash_db(patient_hash)


def get_all_patients() -> list:
    """
    Get all patients with workflow counts.

    Returns:
        List of patient summaries
    """
    return get_all_patients_db()


# Two-Phase Commit Functions


def reserve_patient_workflow(
    patient_name: str, patient_hash: str, workflow_id: str, file_path: str, department: Optional[str] = None
):
    """
    Phase 1: Reserve a patient-workflow mapping with 'pending' status.

    Call this BEFORE starting the Temporal workflow.

    Args:
        patient_name: Plain text patient name
        patient_hash: 8-char hash used in filenames/workflow IDs
        workflow_id: Temporal workflow ID
        file_path: Path to audio file
        department: Optional department name
    """
    created_at = datetime.now(Config.TIMEZONE).isoformat()
    reserve_workflow_mapping_db(
        patient_name=patient_name,
        patient_hash=patient_hash,
        workflow_id=workflow_id,
        file_path=file_path,
        department=department,
        created_at=created_at,
    )


def commit_patient_workflow(workflow_id: str):
    """
    Phase 2a: Mark workflow as 'active' after successful start.

    Args:
        workflow_id: Workflow ID to commit
    """
    commit_workflow_mapping_db(workflow_id)


def rollback_patient_workflow(workflow_id: str):
    """
    Phase 2b: Delete pending workflow record on failure.

    Args:
        workflow_id: Workflow ID to rollback
    """
    rollback_workflow_mapping_db(workflow_id)
