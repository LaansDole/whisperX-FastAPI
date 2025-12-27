"""Admin endpoints for patient workflow monitoring."""

from fastapi import APIRouter, HTTPException
from ..logger import logger
from ..patients.mapping import get_patient_by_workflow, get_workflows_by_patient_hash, get_patient_name_by_hash


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/patient/hash/{patient_hash}")
async def get_patient_info_by_hash(patient_hash: str):
    """
    Get patient information by hash (Admin lookup).

    **Purpose:** Admin monitoring - map file/workflow hash back to patient

    **Example:**
    - File: `audio_154c26a1_20251227.mp3`
    - Hash: `154c26a1`
    - Query: `GET /admin/patient/hash/154c26a1`
    - Returns: `{ "patient_name": "John Michael Smith", "workflows": [...] }`

    Args:
        patient_hash: 8-character patient hash from filename/workflow ID

    Returns:
        Patient name and all associated workflows
    """
    # Get patient name from DB (plain text)
    patient_name = get_patient_name_by_hash(patient_hash)

    if not patient_name:
        raise HTTPException(status_code=404, detail=f"No patient found with hash: {patient_hash}")

    # Get all workflows for this patient
    workflows = get_workflows_by_patient_hash(patient_hash)

    logger.info(f"Admin lookup: patient hash {patient_hash} → {len(workflows)} workflows")

    return {
        "patient_hash": patient_hash,
        "patient_name": patient_name,  # Plain text from DB
        "total_workflows": len(workflows),
        "workflows": workflows,
    }


@router.get("/workflow/{workflow_id}/patient")
async def get_patient_by_workflow_id(workflow_id: str):
    """
    Get patient information by workflow ID.

    **Example:**
    - Workflow ID: `whisperx-wf-pt_154c26a1-20251227_145128`
    - Query: `GET /admin/workflow/whisperx-wf-pt_154c26a1-20251227_145128/patient`
    - Returns: `{ "patient_name": "John Michael Smith", ... }`

    Args:
        workflow_id: Temporal workflow ID

    Returns:
        Patient information for the workflow
    """
    mapping = get_patient_by_workflow(workflow_id)

    if not mapping:
        raise HTTPException(status_code=404, detail=f"No patient mapping found for workflow: {workflow_id}")

    logger.info(f"Admin lookup: workflow {workflow_id} → patient {mapping.get('patient_name')}")

    return mapping


@router.get("/patients")
async def list_all_patients():
    """
    List all patients with workflow counts (Admin overview).

    Returns:
        Summary of all patients and their workflow counts
    """
    from ..patients.mapping import get_all_patients

    patients = get_all_patients()

    return {"total_patients": len(patients), "patients": patients}


@router.get("/database/stats", tags=["Admin"])
async def get_database_stats():
    """Get real-time database statistics for monitoring."""
    from ..patients.database import get_db_connection

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total mappings
        cursor.execute("SELECT COUNT(*) FROM patient_workflow_mappings")
        total_mappings = cursor.fetchone()[0]

        # Unique patients
        cursor.execute("SELECT COUNT(DISTINCT patient_hash) FROM patient_workflow_mappings")
        unique_patients = cursor.fetchone()[0]

        # Recent entries
        cursor.execute("""
            SELECT patient_name, patient_hash, workflow_id, created_at
            FROM patient_workflow_mappings
            ORDER BY created_at DESC
            LIMIT 5
        """)
        recent = [dict(row) for row in cursor.fetchall()]

    return {"total_mappings": total_mappings, "unique_patients": unique_patients, "recent_entries": recent}
