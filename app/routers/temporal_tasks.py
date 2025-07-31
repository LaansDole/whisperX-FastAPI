
from fastapi import APIRouter, Depends, HTTPException
from app.temporal_manager import temporal_manager
from app.temporal_workflows import WhisperXWorkflow
from app.temporal_config import config
from app.schemas import Response

temporal_router = APIRouter()

@temporal_router.post("/temporal/workflow", tags=["Temporal"])
async def start_workflow(audio_path: str, params: dict):
    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    handle = await client.start_workflow(
        WhisperXWorkflow.run,
        args=[audio_path, params],
        id=f"whisperx-workflow-{audio_path}",
        task_queue=config.TEMPORAL_TASK_QUEUE,
    )
    return Response(identifier=handle.id, message="Workflow started")

@temporal_router.get("/temporal/workflow/{workflow_id}", tags=["Temporal"])
async def get_workflow_status(workflow_id: str):
    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    handle = client.get_workflow_handle(workflow_id)
    try:
        description = await handle.describe()
        return {
            "workflow_id": handle.id,
            "status": description.status.name,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {e}")


@temporal_router.get("/temporal/workflow/{workflow_id}/result", tags=["Temporal"])
async def get_workflow_result(workflow_id: str):
    client = await temporal_manager.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Temporal service not available")
    handle = client.get_workflow_handle(workflow_id)
    try:
        result = await handle.result()
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found or not complete: {e}")
