"""Additional Temporal workflow query endpoints for patient-based access."""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
from datetime import datetime
from ..temporal.manager import temporal_manager
from ..logger import logger
from ..patients.mapping import get_workflows_by_patient_hash

router = APIRouter(prefix="/temporal")


@router.get("/patient/{patient_hash}/workflows", tags=["Temporal"])
async def get_patient_workflows(
    patient_hash: str,
    status: Optional[str] = Query(None, description="Filter by workflow status")
):
    """
    List all workflows for a specific patient.
    
    Uses SQLite database for instant results (no Temporal indexing delay).
    
    Args:
        patient_hash: 8-character patient hash from filename/workflow ID
        status: Optional workflow status filter (RUNNING, COMPLETED, FAILED)
        
    Returns:
        List of workflows for the patient
    """
    try:
        # Query SQLite database for workflows
        db_workflows = get_workflows_by_patient_hash(patient_hash)
        
        if not db_workflows:
            return {
                "patient_hash": patient_hash,
                "total_found": 0,
                "workflows": []
            }
        
        # Optionally enrich with Temporal status if needed
        client = await temporal_manager.get_client()
        workflows = []
        
        for db_wf in db_workflows:
            workflow_info = {
                "workflow_id": db_wf["workflow_id"],
                "file_path": db_wf["file_path"],
                "department": db_wf["department"],
                "created_at": db_wf["created_at"],
                "status": "UNKNOWN"  # Default
            }
            
            # Try to get status from Temporal
            if client:
                try:
                    handle = client.get_workflow_handle(db_wf["workflow_id"])
                    describe = await handle.describe()
                    workflow_info["status"] = describe.status.name
                    
                    # Filter by status if requested
                    if status and describe.status.name != status:
                        continue
                except Exception:
                    pass  # Workflow might be old/archived, keep it with UNKNOWN status
            
            workflows.append(workflow_info)
        
        return {
            "patient_hash": patient_hash,
            "total_found": len(workflows),
            "workflows": workflows
        }
        
    except Exception as e:
        logger.error(f"Failed to query patient workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query workflows: {str(e)}")
