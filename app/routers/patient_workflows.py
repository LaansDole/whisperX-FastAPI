"""Additional Temporal workflow query endpoints for patient-based access."""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from ..temporal.manager import temporal_manager
from ..logger import logger
from ..patients.mapping import get_workflows_by_patient_hash

router = APIRouter(prefix="/temporal")


@router.get("/patient/{patient_hash}/workflows", tags=["Temporal"])
async def get_patient_workflows(
    patient_hash: str,
    status: Optional[str] = Query(None, description="Filter by workflow status (RUNNING, COMPLETED, FAILED)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of workflows to return (1-100)"),
    offset: int = Query(0, ge=0, description="Number of workflows to skip for pagination"),
):
    """
    List workflows for a specific patient with pagination.

    Uses SQLite database for instant results (no Temporal indexing delay).
    Pagination limits the number of Temporal API calls for better performance.

    Args:
        patient_hash: 8-character patient hash from filename/workflow ID
        status: Optional workflow status filter (RUNNING, COMPLETED, FAILED)
        limit: Maximum workflows to return per request (default: 20, max: 100)
        offset: Number of workflows to skip (for pagination)

    Returns:
        Paginated list of workflows with metadata

    Example:
        GET /temporal/patient/abc12345/workflows?limit=10&offset=0  # First 10
        GET /temporal/patient/abc12345/workflows?limit=10&offset=10  # Next 10
    """
    try:
        # Query SQLite database for workflows
        db_workflows = get_workflows_by_patient_hash(patient_hash)

        if not db_workflows:
            return {
                "patient_hash": patient_hash,
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "workflows": [],
            }

        # Store total count before pagination
        total_count = len(db_workflows)

        # Apply pagination to limit Temporal API calls
        paginated_workflows = db_workflows[offset : offset + limit]

        # Optionally enrich with Temporal status if needed
        client = await temporal_manager.get_client()
        workflows = []

        for db_wf in paginated_workflows:
            workflow_info = {
                "workflow_id": db_wf["workflow_id"],
                "department": db_wf["department"],
                "created_at": db_wf["created_at"],
                "status": "UNKNOWN",  # Default
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
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "returned_count": len(workflows),
            "workflows": workflows,
        }

    except Exception as e:
        logger.error(f"Failed to query patient workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query workflows: {str(e)}")


@router.get("/patient/{patient_hash}/latest", tags=["Temporal"])
async def get_patient_latest_workflow(patient_hash: str):
    """
    Get the latest workflow for a specific patient.

    Returns the most recent workflow based on created_at timestamp.

    Args:
        patient_hash: 8-character patient hash from filename/workflow ID

    Returns:
        Latest workflow info or 404 if no workflows found
    """
    try:
        # Query SQLite database for workflows
        db_workflows = get_workflows_by_patient_hash(patient_hash)

        if not db_workflows:
            raise HTTPException(status_code=404, detail=f"No workflows found for patient hash: {patient_hash}")

        # Get the latest workflow (already sorted by created_at DESC)
        latest_wf = db_workflows[0]

        # Get Temporal status if available
        client = await temporal_manager.get_client()
        workflow_info = {
            "patient_hash": patient_hash,
            "workflow_id": latest_wf["workflow_id"],
            "department": latest_wf["department"],
            "created_at": latest_wf["created_at"],
            "status": "UNKNOWN",
        }

        if client:
            try:
                handle = client.get_workflow_handle(latest_wf["workflow_id"])
                describe = await handle.describe()
                workflow_info["status"] = describe.status.name
            except Exception as e:
                # Keep status as "UNKNOWN" but log the failure to retrieve Temporal status
                logger.warning(
                    "Failed to retrieve Temporal status for workflow_id=%s, patient_hash=%s: %s",
                    latest_wf.get("workflow_id"),
                    patient_hash,
                    str(e),
                )

        return workflow_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest workflow: {str(e)}")
