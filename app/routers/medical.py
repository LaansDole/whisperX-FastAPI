"""Router for medical processing endpoints and health checks."""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..config import Config
from ..llm.lm_studio_client import LMStudioClient, LMStudioConfig
from ..llm.medical_llm_service import MedicalLLMService
from ..llm.chatbot_service import MedicalChatbotService
from ..hipaa.audit_logger import HIPAAAuditLogger
from ..hipaa.access_control import HIPAAAccessControl, Permission
from ..vector_store.medical_vector_store import MedicalDocumentVectorStore

router = APIRouter()
access_control = HIPAAAccessControl()
audit_logger = HIPAAAuditLogger()


@router.get("/health/lm-studio", tags=["Health"], summary="Check LM Studio service health")
async def lm_studio_health():
    """Check LM Studio service health and available models."""
    if not Config.LM_STUDIO_ENABLED:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "disabled",
                "message": "LM Studio integration is disabled",
                "url": Config.LM_STUDIO_BASE_URL
            }
        )

    try:
        client = LMStudioClient()

        # Check connection and list models
        status_info = await client.test_connection()

        await client.close()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=status_info
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error": str(e),
                "url": Config.LM_STUDIO_BASE_URL,
                "message": "LM Studio service is not available"
            }
        )


@router.get("/health/medical", tags=["Health"], summary="Check medical processing components")
async def medical_health():
    """Check health of all medical processing components."""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "medical_rag_enabled": Config.MEDICAL_RAG_ENABLED,
        "components": {}
    }

    # LM Studio health
    if Config.LM_STUDIO_ENABLED:
        try:
            client = LMStudioClient()
            lm_studio_status = await client.test_connection()
            health_status["components"]["lm_studio"] = {
                "status": "healthy" if lm_studio_status["status"] == "ok" else "unhealthy",
                "details": lm_studio_status
            }
            await client.close()
        except Exception as e:
            health_status["components"]["lm_studio"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    else:
        health_status["components"]["lm_studio"] = {
            "status": "disabled",
            "message": "LM Studio is disabled"
        }

    # Vector store health
    if Config.ENABLE_VECTOR_STORAGE:
        try:
            vector_store = MedicalDocumentVectorStore(
                storage_dir=Config.VECTOR_DB_PATH,
                embedding_dim=Config.EMBEDDING_DIMENSION
            )
            stats = vector_store.get_statistics()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "statistics": stats
            }
            vector_store.close()
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    else:
        health_status["components"]["vector_store"] = {
            "status": "disabled",
            "message": "Vector storage is disabled"
        }

    # HIPAA audit log health
    try:
        audit_stats = audit_logger.verify_audit_trail()
        health_status["components"]["audit_log"] = {
            "status": "healthy" if audit_stats["verified"] else "warning",
            "details": audit_stats
        }
    except Exception as e:
        health_status["components"]["audit_log"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Configuration validation
    config_issues = Config.validate_medical_config()
    if config_issues:
        health_status["components"]["configuration"] = {
            "status": "warning",
            "issues": config_issues
        }
    else:
        health_status["components"]["configuration"] = {
            "status": "healthy",
            "message": "All required settings configured"
        }

    # Overall status
    all_healthy = all(
        comp.get("status") == "healthy" for comp in health_status["components"].values()
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=health_status
    )


@router.get("/health/vector-store", tags=["Health"], summary="Vector store statistics")
async def vector_store_health():
    """Get vector store statistics and health."""
    if not Config.ENABLE_VECTOR_STORAGE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "disabled",
                "message": "Vector storage is disabled"
            }
        )

    try:
        vector_store = MedicalDocumentVectorStore(
            storage_dir=Config.VECTOR_DB_PATH,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )

        stats = vector_store.get_statistics()
        vector_store.close()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "statistics": stats
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error": str(e)
            }
        )


@router.post("/medical/phi-detect", tags=["Medical"], summary="Detect PHI in text")
@access_control.require_permission(Permission.READ_PHI)
async def detect_phi(
    text: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Detect Protected Health Information in provided text."""
    if not Config.is_medical_processing_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical processing is not enabled"
        )

    # Log PHI access attempt
    audit_logger.log_phi_access(
        user_id=current_user.get("user_id"),
        patient_id="direct_input",
        action="phi_detection",
        resource="api_endpoint",
        result="attempt"
    )

    try:
        client = LMStudioClient()
        service = MedicalLLMService(client)

        # Perform PHI detection
        phi_result = await service.detect_phi(text)

        await client.close()

        # Log successful PHI access
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id="direct_input",
            action="phi_detection",
            resource="api_endpoint",
            result="success",
            phi_detected=phi_result.get("phi_detected", False),
            entity_count=len(phi_result.get("entities", []))
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=phi_result
        )

    except Exception as e:
        # Log failed PHI access
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id="direct_input",
            action="phi_detection",
            resource="api_endpoint",
            result="failed",
            error=str(e)
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PHI detection failed: {str(e)}"
        )


@router.post("/medical/extract-entities", tags=["Medical"], summary="Extract medical entities")
@access_control.require_permission(Permission.READ_PHI)
async def extract_medical_entities(
    text: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Extract medical entities from provided text."""
    if not Config.is_medical_processing_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical processing is not enabled"
        )

    try:
        client = LMStudioClient()
        service = MedicalLLMService(client)

        # Perform entity extraction
        entities = await service.extract_medical_entities(text)

        await client.close()

        # Log medical data processing
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id="direct_input",
            action="entity_extraction",
            resource="api_endpoint",
            result="success",
            entity_count=len(entities)
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "entities": entities,
                "entity_count": len(entities),
                "processed_at": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}"
        )


@router.post("/medical/generate-soap", tags=["Medical"], summary="Generate SOAP note")
@access_control.require_permission(Permission.WRITE_PHI)
async def generate_soap_note(
    transcript: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Generate SOAP note from consultation transcript."""
    if not Config.is_medical_processing_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical processing is not enabled"
        )

    try:
        client = LMStudioClient()
        service = MedicalLLMService(client)

        # Generate SOAP note
        soap_note = await service.generate_soap_note(transcript)

        await client.close()

        # Log clinical documentation creation
        background_tasks.add_task(
            audit_logger.log_data_modification,
            user_id=current_user.get("user_id"),
            action="create",
            resource_type="soap_note",
            resource_id=f"soap_{int(time.time())}",
            changes={"sections_created": list(soap_note.keys())}
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "soap_note": soap_note,
                "generated_at": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SOAP note generation failed: {str(e)}"
        )


@router.get("/medical/search-similar", tags=["Medical"], summary="Search similar consultations")
@access_control.require_permission(Permission.SEARCH_PHI)
async def search_similar_consultations(
    query_text: str,
    background_tasks: BackgroundTasks,
    limit: int = 10,
    patient_id_encrypted: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Search for similar consultations using vector similarity."""
    if not Config.ENABLE_VECTOR_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage is not enabled"
        )

    try:
        # Generate embedding for query
        client = LMStudioClient()
        query_embedding = await client.generate_embedding(query_text, Config.EMBEDDING_MODEL)
        await client.close()

        # Search vector store
        import numpy as np
        vector_store = MedicalDocumentVectorStore(
            storage_dir=Config.VECTOR_DB_PATH,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )

        results = await vector_store.search_similar(
            query_embedding=np.array(query_embedding),
            patient_id_encrypted=patient_id_encrypted,
            limit=limit
        )

        vector_store.close()

        # Log search activity
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=patient_id_encrypted or "cross_patient_search",
            action="similarity_search",
            resource="vector_database",
            result="success",
            results_count=len(results)
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "query": query_text,
                "results": results,
                "result_count": len(results),
                "searched_at": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/medical/consultation/{consultation_id}", tags=["Medical"], summary="Get consultation details")
@access_control.require_permission(Permission.READ_PHI)
async def get_consultation_details(
    consultation_id: str,
    background_tasks: BackgroundTasks,
    include_entities: bool = True,
    include_phi: bool = False,
    include_structured: bool = True,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Get detailed information about a specific consultation."""
    if not Config.ENABLE_VECTOR_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage is not enabled"
        )

    try:
        vector_store = MedicalDocumentVectorStore(
            storage_dir=Config.VECTOR_DB_PATH,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )

        details = await vector_store.get_consultation_details(
            consultation_id=consultation_id,
            include_entities=include_entities,
            include_phi=include_phi,
            include_structured=include_structured
        )

        vector_store.close()

        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Consultation {consultation_id} not found"
            )

        # Log PHI access
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=details.get("patient_id_encrypted"),
            action="consultation_retrieval",
            resource=consultation_id,
            result="success"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=details
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve consultation: {str(e)}"
        )


@router.get("/medical/patient/{patient_id_encrypted}/consultations", tags=["Medical"], summary="Get patient consultations")
@access_control.require_permission(Permission.READ_PHI)
async def get_patient_consultations(
    patient_id_encrypted: str,
    background_tasks: BackgroundTasks,
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """Get all consultations for a specific patient."""
    if not Config.ENABLE_VECTOR_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage is not enabled"
        )

    try:
        vector_store = MedicalDocumentVectorStore(
            storage_dir=Config.VECTOR_DB_PATH,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )

        consultations = await vector_store.get_patient_consultations(
            patient_id_encrypted=patient_id_encrypted,
            limit=limit,
            offset=offset
        )

        vector_store.close()

        # Log PHI access
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=patient_id_encrypted,
            action="patient_history_access",
            resource="patient_consultations",
            result="success",
            record_count=len(consultations)
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "patient_id_encrypted": patient_id_encrypted,
                "consultations": consultations,
                "count": len(consultations),
                "limit": limit,
                "offset": offset
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient consultations: {str(e)}"
        )


@router.get("/medical/stats", tags=["Medical"], summary="Get medical processing statistics")
@access_control.require_permission(Permission.VIEW_AUDIT_LOG)
async def get_medical_statistics(current_user: Dict[str, Any] = Depends(lambda: {})):
    """Get statistics about medical processing and data."""
    if not Config.ENABLE_VECTOR_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage is not enabled"
        )

    try:
        vector_store = MedicalDocumentVectorStore(
            storage_dir=Config.VECTOR_DB_PATH,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )

        stats = vector_store.get_statistics()
        vector_store.close()

        # Add configuration info
        stats["configuration"] = {
            "medical_rag_enabled": Config.MEDICAL_RAG_ENABLED,
            "lm_studio_enabled": Config.LM_STUDIO_ENABLED,
            "vector_storage_enabled": Config.ENABLE_VECTOR_STORAGE,
            "embedding_model": Config.EMBEDDING_MODEL,
            "embedding_dimension": Config.EMBEDDING_DIMENSION
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# ============================================================================
# RAG Chatbot Endpoints
# ============================================================================

@router.post("/medical/chat", tags=["Medical"], summary="RAG-powered chatbot for patient queries")
@access_control.require_permission(Permission.READ_PHI)
async def medical_chat(
    query: str,
    patient_id_encrypted: str,
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """
    Query patient medical records using RAG-powered chatbot.
    
    This endpoint:
    1. Searches for relevant patient consultations in the vector store
    2. Uses retrieved context to generate informed responses
    3. Maintains conversation history per session
    
    Args:
        query: User's question about the patient
        patient_id_encrypted: Encrypted patient identifier
        session_id: Optional session ID for conversation continuity
    """
    if not Config.is_medical_processing_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical processing is not enabled. Set MEDICAL_RAG_ENABLED=true and LM_STUDIO_ENABLED=true"
        )
    
    if not Config.ENABLE_VECTOR_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage is not enabled. Set ENABLE_VECTOR_STORAGE=true"
        )

    # Log chatbot query
    audit_logger.log_phi_access(
        user_id=current_user.get("user_id"),
        patient_id=patient_id_encrypted,
        action="chatbot_query",
        resource="rag_chatbot",
        result="attempt"
    )

    try:
        chatbot = MedicalChatbotService()
        
        result = await chatbot.query(
            user_query=query,
            patient_id_encrypted=patient_id_encrypted,
            session_id=session_id
        )
        
        await chatbot.close()

        # Log successful query
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=patient_id_encrypted,
            action="chatbot_query",
            resource="rag_chatbot",
            result="success",
            sources_count=len(result.get("sources", []))
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )

    except Exception as e:
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=patient_id_encrypted,
            action="chatbot_query",
            resource="rag_chatbot",
            result="failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot query failed: {str(e)}"
        )


@router.post("/medical/process-transcript", tags=["Medical"], summary="Process transcript through RAG pipeline")
@access_control.require_permission(Permission.WRITE_PHI)
async def process_transcript(
    transcript: str,
    patient_id: str,
    provider_id: str,
    background_tasks: BackgroundTasks,
    encounter_date: Optional[str] = None,
    enable_phi_detection: bool = True,
    enable_entity_extraction: bool = True,
    enable_soap_generation: bool = True,
    enable_vector_storage: bool = True,
    current_user: Dict[str, Any] = Depends(lambda: {})
):
    """
    Process a medical transcript through the full RAG pipeline.
    
    This endpoint:
    1. Extracts medical entities (diagnoses, medications, etc.)
    2. Detects PHI (Protected Health Information)
    3. Generates SOAP note
    4. Creates embeddings and stores in vector database
    
    Use this after transcription to prepare patient data for the chatbot.
    
    Args:
        transcript: The medical consultation transcript text
        patient_id: Patient identifier (will be encrypted for storage)
        provider_id: Healthcare provider identifier
        encounter_date: Date of encounter (ISO format, defaults to today)
        enable_*: Flags to enable/disable specific processing steps (for debugging)
    """
    import hashlib
    import numpy as np
    import uuid
    
    if not Config.is_medical_processing_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical processing is not enabled. Set MEDICAL_RAG_ENABLED=true and LM_STUDIO_ENABLED=true"
        )

    # Generate consultation ID and encrypt patient ID
    consultation_id = f"cons_{uuid.uuid4().hex[:12]}"
    patient_id_encrypted = hashlib.sha256(
        f"{patient_id}{Config.HIPAA_SALT}".encode()
    ).hexdigest()[:32]
    encounter_date = encounter_date or datetime.now().date().isoformat()

    # Log processing start
    audit_logger.log_phi_access(
        user_id=current_user.get("user_id"),
        patient_id=patient_id_encrypted,
        action="transcript_processing",
        resource=consultation_id,
        result="started"
    )

    try:
        results = {
            "consultation_id": consultation_id,
            "patient_id_encrypted": patient_id_encrypted,
            "provider_id": provider_id,
            "encounter_date": encounter_date,
            "processing_started": datetime.now().isoformat(),
            "steps": {}
        }

        # Initialize LM Studio client
        config = LMStudioConfig(
            base_url=Config.LM_STUDIO_BASE_URL,
            timeout=Config.LM_STUDIO_TIMEOUT,
            temperature=Config.LM_STUDIO_TEMPERATURE,
            max_tokens=Config.LM_STUDIO_MAX_TOKENS,
            model=Config.LM_STUDIO_MODEL,
        )
        client = LMStudioClient(config)
        service = MedicalLLMService(client)

        # Step 1: PHI Detection
        if enable_phi_detection and Config.ENABLE_PHI_DETECTION:
            try:
                phi_result = await service.detect_phi(transcript)
                results["steps"]["phi_detection"] = {
                    "success": True,
                    "phi_detected": phi_result.get("phi_detected", False),
                    "entity_count": len(phi_result.get("entities", [])),
                    "entities": phi_result.get("entities", [])
                }
            except Exception as e:
                results["steps"]["phi_detection"] = {"success": False, "error": str(e)}
        else:
            results["steps"]["phi_detection"] = {"skipped": True}

        # Step 2: Medical Entity Extraction
        entities = []
        if enable_entity_extraction and Config.ENABLE_ENTITY_EXTRACTION:
            try:
                entities = await service.extract_medical_entities(transcript)
                results["steps"]["entity_extraction"] = {
                    "success": True,
                    "entity_count": len(entities),
                    "entities": entities
                }
            except Exception as e:
                results["steps"]["entity_extraction"] = {"success": False, "error": str(e)}
        else:
            results["steps"]["entity_extraction"] = {"skipped": True}

        # Step 3: SOAP Note Generation
        if enable_soap_generation and Config.ENABLE_SOAP_GENERATION:
            try:
                soap_note = await service.generate_soap_note(transcript)
                results["steps"]["soap_generation"] = {
                    "success": True,
                    "soap_note": soap_note
                }
            except Exception as e:
                results["steps"]["soap_generation"] = {"success": False, "error": str(e)}
        else:
            results["steps"]["soap_generation"] = {"skipped": True}

        # Step 4: Embedding Generation & Vector Storage
        if enable_vector_storage and Config.ENABLE_VECTOR_STORAGE:
            try:
                # Generate embedding
                embedding = await client.generate_embedding(transcript, Config.EMBEDDING_MODEL)
                results["steps"]["embedding_generation"] = {
                    "success": True,
                    "dimension": len(embedding)
                }

                # Store in vector database
                vector_store = MedicalDocumentVectorStore(
                    storage_dir=Config.VECTOR_DB_PATH,
                    embedding_dim=Config.EMBEDDING_DIMENSION
                )

                vector_id = await vector_store.store_consultation(
                    consultation_id=consultation_id,
                    patient_id_encrypted=patient_id_encrypted,
                    provider_id=provider_id,
                    encounter_date=encounter_date,
                    transcript=transcript,
                    embedding=np.array(embedding, dtype=np.float32),
                    metadata={"processed_at": datetime.now().isoformat()}
                )

                # Store medical entities
                if entities:
                    await vector_store.store_medical_entities(consultation_id, entities)

                # Store structured document
                soap_note = results["steps"].get("soap_generation", {}).get("soap_note", {})
                await vector_store.store_structured_document(
                    consultation_id=consultation_id,
                    structured_doc={"transcript_length": len(transcript)},
                    soap_note=soap_note if isinstance(soap_note, dict) else None,
                    clinical_summary=soap_note.get("assessment") if isinstance(soap_note, dict) else None
                )

                vector_store.save_index()
                vector_store.close()

                results["steps"]["vector_storage"] = {
                    "success": True,
                    "vector_id": vector_id
                }

            except Exception as e:
                results["steps"]["vector_storage"] = {"success": False, "error": str(e)}
        else:
            results["steps"]["vector_storage"] = {"skipped": True}

        await client.close()

        # Calculate summary
        successful_steps = sum(
            1 for step in results["steps"].values()
            if step.get("success", False)
        )
        total_steps = sum(
            1 for step in results["steps"].values()
            if not step.get("skipped", False)
        )

        results["processing_completed"] = datetime.now().isoformat()
        results["summary"] = {
            "successful_steps": successful_steps,
            "total_steps": total_steps,
            "all_successful": successful_steps == total_steps
        }

        # Log successful processing
        background_tasks.add_task(
            audit_logger.log_data_modification,
            user_id=current_user.get("user_id"),
            action="create",
            resource_type="medical_consultation",
            resource_id=consultation_id,
            changes={"steps_completed": list(results["steps"].keys())}
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=results
        )

    except Exception as e:
        background_tasks.add_task(
            audit_logger.log_phi_access,
            user_id=current_user.get("user_id"),
            patient_id=patient_id_encrypted,
            action="transcript_processing",
            resource=consultation_id,
            result="failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcript processing failed: {str(e)}"
        )


@router.delete("/medical/chat/session/{session_id}", tags=["Medical"], summary="Clear chat session")
async def clear_chat_session(session_id: str):
    """Clear conversation history for a specific session."""
    try:
        chatbot = MedicalChatbotService()
        chatbot.clear_session(session_id)
        await chatbot.close()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Session {session_id} cleared", "session_id": session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}"
        )