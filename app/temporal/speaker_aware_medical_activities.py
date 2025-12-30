"""Temporal activities for audio processing workflows."""

from temporalio import activity
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .error_handler import TemporalErrorHandler
from .monitoring import TemporalMetrics

logging.basicConfig(level=logging.INFO)


# ============================================================================
# PHASE 3: Speaker-Aware Medical Activities
# ============================================================================


@activity.defn
async def transform_to_dialogue_activity(
    whisperx_result: Dict[str, Any],
    workflow_id: Optional[str] = None,
    manual_speaker_mapping: Optional[Dict[str, str]] = None,
    consultation_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Transform WhisperX result to speaker-attributed dialogue.

    Args:
        whisperx_result: Complete WhisperX transcription result
        workflow_id: Optional workflow ID for context
        manual_speaker_mapping: Optional manual speaker role mapping
        consultation_metadata: Optional consultation metadata

    Returns:
        Speaker-attributed dialogue data with statistics

    Raises:
        ApplicationError: If transformation fails
    """
    from app.services.transcription_transformer import TranscriptionTransformer

    async with TemporalMetrics.activity_timer("transform_to_dialogue", workflow_id or "unknown"):
        try:
            transformer = TranscriptionTransformer()

            dialogue_data = transformer.transform_with_overrides(
                whisperx_result=whisperx_result,
                manual_speaker_mapping=manual_speaker_mapping or {},
                workflow_id=workflow_id,
                consultation_metadata=consultation_metadata,
            )

            logging.info(
                f"Dialogue transformation complete: {len(dialogue_data.get('dialogue', []))} segments, "
                f"{len(dialogue_data.get('speaker_mapping', {}))} speakers"
            )

            return dialogue_data

        except Exception as e:
            logging.error(f"Dialogue transformation failed: {e}")
            # Classify as non-retryable for invalid input data
            is_retryable = not isinstance(e, (ValueError, TypeError))
            raise TemporalErrorHandler.create_application_error(
                e,
                "Dialogue Transformation",
                retryable=is_retryable,
            )


@activity.defn
async def detect_phi_in_dialogue_activity(dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect PHI with speaker attribution from dialogue data.

    Args:
        dialogue_data: Speaker-attributed dialogue from transform activity

    Returns:
        PHI detection results with speaker attribution

    Raises:
        ApplicationError: If PHI detection fails or LM Studio unavailable
    """
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("phi_detection_dialogue"):
        try:
            # Initialize LM Studio client
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
                temperature=Config.LM_STUDIO_TEMPERATURE,
                max_tokens=Config.LM_STUDIO_MAX_TOKENS,
                model=Config.LM_STUDIO_MODEL,
            )

            async with LMStudioClient(config) as client:
                service = MedicalLLMService(client)

                # Check LM Studio availability - fail fast if unavailable
                if not await client.health_check():
                    raise TemporalErrorHandler.create_application_error(
                        Exception("LM Studio not available"), "PHI Detection (Dialogue)"
                    )

                # Perform speaker-aware PHI detection
                phi_result = await service.detect_phi_in_dialogue(dialogue_data)

            logging.info(
                f"PHI detection complete: {phi_result.get('phi_detected', False)}, "
                f"{len(phi_result.get('entities', []))} entities"
            )

            return {
                **phi_result,
                "processed_at": datetime.now(Config.TIMEZONE).isoformat(),
            }

        except Exception as e:
            logging.error(f"PHI detection in dialogue failed: {e}")
            # Classify as non-retryable for invalid input data
            is_retryable = not isinstance(e, (ValueError, TypeError))
            raise TemporalErrorHandler.create_application_error(
                e,
                "PHI Detection (Dialogue)",
                retryable=is_retryable,
            )


@activity.defn
async def extract_entities_with_speaker_activity(dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract medical entities with speaker attribution.

    Args:
        dialogue_data: Speaker-attributed dialogue from transform activity

    Returns:
        Medical entities with speaker roles

    Raises:
        ApplicationError: If entity extraction fails or LM Studio unavailable
    """
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("entity_extraction_speaker"):
        try:
            # Initialize LM Studio client
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
                temperature=Config.LM_STUDIO_TEMPERATURE,
                max_tokens=Config.LM_STUDIO_MAX_TOKENS,
                model=Config.LM_STUDIO_MODEL,
            )

            async with LMStudioClient(config) as client:
                service = MedicalLLMService(client)

                # Check LM Studio availability
                if not await client.health_check():
                    raise TemporalErrorHandler.create_application_error(
                        Exception("LM Studio not available"), "Entity Extraction (Speaker)"
                    )

                # Extract entities with speaker attribution
                entities = await service.extract_entities_with_speaker(dialogue_data)

            # Count entities by speaker
            doctor_entities = sum(1 for e in entities if e.get("speaker_role") == "doctor")
            patient_entities = sum(1 for e in entities if e.get("speaker_role") == "patient")

            logging.info(
                f"Entity extraction complete: {len(entities)} total "
                f"(doctor: {doctor_entities}, patient: {patient_entities})"
            )

            return {
                "entities": entities,
                "entity_count": len(entities),
                "speaker_breakdown": {
                    "doctor": doctor_entities,
                    "patient": patient_entities,
                    "unknown": len(entities) - doctor_entities - patient_entities,
                },
                "processed_at": datetime.now(Config.TIMEZONE).isoformat(),
            }

        except Exception as e:
            logging.error(f"Entity extraction with speaker failed: {e}")
            # Classify as non-retryable for invalid input data
            is_retryable = not isinstance(e, (ValueError, TypeError))
            raise TemporalErrorHandler.create_application_error(
                e,
                "Entity Extraction (Speaker)",
                retryable=is_retryable,
            )


@activity.defn
async def generate_soap_from_dialogue_activity(dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate SOAP note from speaker-attributed dialogue.

    Args:
        dialogue_data: Speaker-attributed dialogue from transform activity

    Returns:
        SOAP note with proper speaker-based categorization

    Raises:
        ApplicationError: If SOAP generation fails or LM Studio unavailable
    """
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("soap_generation_dialogue"):
        try:
            # Initialize LM Studio client
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
                temperature=Config.LM_STUDIO_TEMPERATURE,
                max_tokens=Config.LM_STUDIO_MAX_TOKENS,
                model=Config.LM_STUDIO_MODEL,
            )

            async with LMStudioClient(config) as client:
                service = MedicalLLMService(client)

                # Check LM Studio availability
                if not await client.health_check():
                    raise TemporalErrorHandler.create_application_error(
                        Exception("LM Studio not available"), "SOAP Generation (Dialogue)"
                    )

                # Generate SOAP note from dialogue
                soap_note = await service.generate_soap_from_dialogue(dialogue_data)

            # Count words in each section
            section_lengths = {
                section: len(content.split()) if content else 0 for section, content in soap_note.items()
            }

            logging.info(
                f"SOAP generation complete: "
                f"S:{section_lengths.get('subjective', 0)}w, "
                f"O:{section_lengths.get('objective', 0)}w, "
                f"A:{section_lengths.get('assessment', 0)}w, "
                f"P:{section_lengths.get('plan', 0)}w"
            )

            return {
                "soap_note": soap_note,
                "section_lengths": section_lengths,
                "processed_at": datetime.now(Config.TIMEZONE).isoformat(),
            }

        except Exception as e:
            logging.error(f"SOAP generation from dialogue failed: {e}")
            # Classify as non-retryable for invalid input data
            is_retryable = not isinstance(e, (ValueError, TypeError))
            raise TemporalErrorHandler.create_application_error(
                e,
                "SOAP Generation (Dialogue)",
                retryable=is_retryable,
            )


@activity.defn
async def store_consultation_with_speaker_data_activity(
    consultation_id: str,
    patient_id_encrypted: str,
    provider_id: str,
    encounter_date: str,
    dialogue_data: Dict[str, Any],
    phi_result: Optional[Dict[str, Any]] = None,
    entities_result: Optional[Dict[str, Any]] = None,
    soap_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Store consultation in vector database with all speaker-attributed data.

    Args:
        consultation_id: Unique consultation identifier
        patient_id_encrypted: Encrypted patient ID
        provider_id: Healthcare provider ID
        encounter_date: Date of encounter (ISO format)
        dialogue_data: Speaker-attributed dialogue
        phi_result: Optional PHI detection results
        entities_result: Optional entity extraction results
        soap_result: Optional SOAP note results

    Returns:
        Storage confirmation with vector ID

    Raises:
        ApplicationError: If vector storage fails (retryable)
    """
    from app.config import Config
    from app.vector_store.medical_vector_store import MedicalDocumentVectorStore
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    import numpy as np

    if not Config.ENABLE_VECTOR_STORAGE:
        return {"skipped": True, "reason": "Vector storage not enabled"}

    async with TemporalMetrics.activity_timer("vector_storage_speaker", consultation_id):
        try:
            # Generate embedding for the dialogue
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
            )

            async with LMStudioClient(config) as client:
                # Format dialogue for embedding
                dialogue_text = "\n".join(
                    [
                        f"{seg.get('speaker', 'Unknown')}: {seg.get('text', '')}"
                        for seg in dialogue_data.get("dialogue", [])
                    ]
                )

                # Generate embedding
                embedding = await client.generate_embedding(dialogue_text, Config.EMBEDDING_MODEL)

            # Initialize vector store
            vector_store = MedicalDocumentVectorStore(
                storage_dir=Config.VECTOR_DB_PATH,
                embedding_dim=Config.EMBEDDING_DIMENSION,
                index_type=Config.VECTOR_INDEX_TYPE,
            )

            try:
                # Prepare metadata
                metadata = {
                    "processing_timestamp": datetime.now(Config.TIMEZONE).isoformat(),
                    "embedding_model": Config.EMBEDDING_MODEL,
                    "speaker_mapping": dialogue_data.get("speaker_mapping", {}),
                    "statistics": dialogue_data.get("statistics", {}),
                    "has_phi": phi_result.get("phi_detected", False) if phi_result else False,
                    "entity_count": entities_result.get("entity_count", 0) if entities_result else 0,
                    "has_soap_note": bool(soap_result and soap_result.get("soap_note")),
                }

                # Store consultation
                embedding_array = np.array(embedding, dtype=np.float32)
                vector_id = await vector_store.store_consultation(
                    consultation_id=consultation_id,
                    patient_id_encrypted=patient_id_encrypted,
                    provider_id=provider_id,
                    encounter_date=encounter_date,
                    transcript=dialogue_text,
                    embedding=embedding_array,
                    metadata=metadata,
                )

                # Store medical entities if available
                if entities_result and not entities_result.get("skipped"):
                    await vector_store.store_medical_entities(consultation_id, entities_result.get("entities", []))

                # Store PHI detections if available
                if phi_result and phi_result.get("phi_detected"):
                    await vector_store.store_phi_detections(consultation_id, phi_result.get("entities", []))

                # Store SOAP note if available
                if soap_result and not soap_result.get("skipped"):
                    soap_note = soap_result.get("soap_note", {})
                    await vector_store.store_structured_document(
                        consultation_id,
                        {"soap_note": soap_note},
                        soap_note,
                        None,  # clinical_summary
                    )

                # Save index
                vector_store.save_index()

                logging.info(f"Vector storage complete: {vector_id}")

                return {
                    "consultation_id": consultation_id,
                    "vector_id": vector_id,
                    "stored_at": datetime.now(Config.TIMEZONE).isoformat(),
                    "metadata": metadata,
                }
            finally:
                # Ensure vector_store is always closed, even on exceptions
                vector_store.close()

        except Exception as e:
            logging.error("Vector storage with speaker data failed", exc_info=e)
            # Classify error as retryable (e.g., network/IO) vs non-retryable (e.g., invalid data)
            is_retryable = not isinstance(e, (ValueError, TypeError))
            raise TemporalErrorHandler.create_application_error(
                e,
                "Vector storage with speaker data failed",
                retryable=is_retryable,
            )
