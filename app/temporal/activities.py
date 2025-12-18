"""Temporal activities for audio processing workflows."""

from temporalio import activity
import logging
import asyncio
from datetime import datetime

from .error_handler import TemporalErrorHandler
from .monitoring import TemporalMetrics

logging.basicConfig(level=logging.INFO)

@activity.defn
async def transcribe_activity(
    audio_path: str,
    model_params: dict,
    asr_options: dict,
    vad_options: dict,
) -> dict:
    """Activity to transcribe audio."""
    from app.audio import process_audio_file
    from app.whisperx_services import transcribe_with_whisper
    from app.schemas import WhisperModelParams, ASROptions, VADOptions

    audio = process_audio_file(audio_path)
    model_params_obj = WhisperModelParams(**model_params)
    asr_options_obj = ASROptions(**asr_options)
    vad_options_obj = VADOptions(**vad_options)
    
    async with TemporalMetrics.activity_timer("transcription", audio_path):
        try:
            result = transcribe_with_whisper(
                audio=audio,
                task=model_params_obj.task.value,
                asr_options=asr_options_obj.model_dump(),
                vad_options=vad_options_obj.model_dump(),
                language=model_params_obj.language,
                batch_size=model_params_obj.batch_size,
                chunk_size=model_params_obj.chunk_size,
                model=model_params_obj.model,
                device=model_params_obj.device,
                device_index=model_params_obj.device_index,
                compute_type=model_params_obj.compute_type,
                threads=model_params_obj.threads,
            )
            return result
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "Transcription")

@activity.defn
async def align_activity(
    transcript: dict, audio_path: str, align_params: dict
) -> dict:
    """Activity to align transcript."""
    from app.audio import process_audio_file
    from app.whisperx_services import align_whisper_output
    from app.schemas import AlignmentParams
    
    audio = process_audio_file(audio_path)
    align_params_obj = AlignmentParams(**align_params)
    
    async with TemporalMetrics.activity_timer("alignment", audio_path):
        try:
            result = align_whisper_output(
                transcript=transcript["segments"],
                audio=audio,
                language_code=transcript["language"],
                device=align_params_obj.device,
                align_model=align_params_obj.align_model,
                interpolate_method=align_params_obj.interpolate_method,
                return_char_alignments=align_params_obj.return_char_alignments,
            )
            return result
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "Alignment")

@activity.defn
async def diarize_activity(audio_path: str, diarize_params: dict) -> dict:
    """Activity to diarize audio."""
    from app.audio import process_audio_file
    from app.whisperx_services import diarize
    from app.schemas import DiarizationParams

    audio = process_audio_file(audio_path)
    diarize_params_obj = DiarizationParams(**diarize_params)
    
    async with TemporalMetrics.activity_timer("diarization", audio_path):
        try:
            result = diarize(
                audio,
                device=diarize_params_obj.device,
                min_speakers=diarize_params_obj.min_speakers,
                max_speakers=diarize_params_obj.max_speakers,
            )
            # Convert DataFrame to a serializable format that preserves the data structure
            # Use orient="index" or a custom format that assign_word_speakers can handle
            return {
                "segments": result.to_dict(orient="records"),
                "metadata": {
                    "min_speakers": diarize_params_obj.min_speakers,
                    "max_speakers": diarize_params_obj.max_speakers,
                }
            }
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "Diarization")

@activity.defn
async def assign_speakers_activity(
    diarization_segments: dict, transcript: dict
) -> dict:
    """Activity to assign speakers."""
    import whisperx
    import pandas as pd
    
    async with TemporalMetrics.activity_timer("speaker_assignment"):
        try:
            # Extract the segments list from the diarization result
            segments_list = diarization_segments.get("segments", [])
            
            # Convert back to DataFrame format that whisperx.assign_word_speakers expects
            segments_df = pd.DataFrame(segments_list)
            
            result = whisperx.assign_word_speakers(segments_df, transcript)
            return result
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "Speaker assignment")

@activity.defn
async def phi_detection_activity(transcript: str, consultation_id: str) -> dict:
    """Activity to detect Protected Health Information."""
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("phi_detection", consultation_id):
        try:
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

            # Check LM Studio availability
            if not await client.health_check():
                raise Exception("LM Studio server not available for PHI detection")

            # Perform PHI detection
            phi_result = await service.detect_phi(transcript)

            await client.close()
            return {
                "consultation_id": consultation_id,
                "phi_detected": phi_result.get("phi_detected", False),
                "entities": phi_result.get("entities", []),
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"PHI detection failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "PHI Detection")

@activity.defn
async def medical_entity_extraction_activity(transcript: str, consultation_id: str) -> dict:
    """Activity to extract medical entities from transcript."""
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("medical_entity_extraction", consultation_id):
        try:
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

            # Check LM Studio availability
            if not await client.health_check():
                raise Exception("LM Studio server not available for entity extraction")

            # Perform entity extraction
            entities = await service.extract_medical_entities(transcript)

            await client.close()
            return {
                "consultation_id": consultation_id,
                "entities": entities,
                "entity_count": len(entities),
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Medical entity extraction failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "Medical Entity Extraction")

@activity.defn
async def soap_generation_activity(transcript: str, consultation_id: str) -> dict:
    """Activity to generate SOAP note from transcript."""
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("soap_generation", consultation_id):
        try:
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

            # Check LM Studio availability
            if not await client.health_check():
                raise Exception("LM Studio server not available for SOAP generation")

            # Generate SOAP note
            soap_note = await service.generate_soap_note(transcript)

            await client.close()
            return {
                "consultation_id": consultation_id,
                "soap_note": soap_note,
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"SOAP generation failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "SOAP Generation")

@activity.defn
async def document_structuring_activity(
    transcript: str,
    consultation_id: str,
    phi_data: dict = None,
    entities: list = None
) -> dict:
    """Activity to structure medical document from components."""
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig
    from app.llm.medical_llm_service import MedicalLLMService

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("document_structuring", consultation_id):
        try:
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

            # Check LM Studio availability
            if not await client.health_check():
                raise Exception("LM Studio server not available for document structuring")

            # Structure document
            structured_doc = await service.structure_medical_document(
                transcript=transcript,
                phi_data=phi_data or {},
                entities=entities or []
            )

            # Generate clinical summary
            if structured_doc and "error" not in structured_doc:
                clinical_summary = await service.generate_clinical_summary(structured_doc)
                structured_doc["clinical_summary"] = clinical_summary

            await client.close()
            return {
                "consultation_id": consultation_id,
                "structured_document": structured_doc,
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Document structuring failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "Document Structuring")

@activity.defn
async def embedding_generation_activity(transcript: str, consultation_id: str) -> dict:
    """Activity to generate embeddings for transcript."""
    from app.config import Config
    from app.llm.lm_studio_client import LMStudioClient, LMStudioConfig

    if not Config.MEDICAL_RAG_ENABLED or not Config.LM_STUDIO_ENABLED:
        return {"skipped": True, "reason": "Medical RAG or LM Studio not enabled"}

    async with TemporalMetrics.activity_timer("embedding_generation", consultation_id):
        try:
            # Initialize LM Studio client
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
            )

            client = LMStudioClient(config)

            # Check LM Studio availability
            if not await client.health_check():
                raise Exception("LM Studio server not available for embedding generation")

            # Generate embedding
            embedding = await client.generate_embedding(transcript, Config.EMBEDDING_MODEL)

            await client.close()
            return {
                "consultation_id": consultation_id,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "model": Config.EMBEDDING_MODEL,
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "Embedding Generation")

@activity.defn
async def vector_storage_activity(
    consultation_id: str,
    patient_id_encrypted: str,
    provider_id: str,
    encounter_date: str,
    transcript: str,
    embedding: list,
    phi_data: dict = None,
    medical_entities: list = None,
    structured_document: dict = None
) -> dict:
    """Activity to store consultation in vector database."""
    from app.config import Config
    import numpy as np
    from app.vector_store.medical_vector_store import MedicalDocumentVectorStore

    if not Config.ENABLE_VECTOR_STORAGE:
        return {"skipped": True, "reason": "Vector storage not enabled"}

    async with TemporalMetrics.activity_timer("vector_storage", consultation_id):
        try:
            # Initialize vector store
            vector_store = MedicalDocumentVectorStore(
                storage_dir=Config.VECTOR_DB_PATH,
                embedding_dim=Config.EMBEDDING_DIMENSION,
                index_type=Config.VECTOR_INDEX_TYPE
            )

            # Convert embedding to numpy array
            embedding_array = np.array(embedding, dtype=np.float32)

            # Store consultation
            vector_id = await vector_store.store_consultation(
                consultation_id=consultation_id,
                patient_id_encrypted=patient_id_encrypted,
                provider_id=provider_id,
                encounter_date=encounter_date,
                transcript=transcript,
                embedding=embedding_array,
                metadata={
                    "processing_timestamp": datetime.now().isoformat(),
                    "embedding_model": Config.EMBEDDING_MODEL
                }
            )

            # Store medical entities if available
            if medical_entities:
                await vector_store.store_medical_entities(consultation_id, medical_entities)

            # Store PHI detections if available
            if phi_data and phi_data.get("phi_detected"):
                await vector_store.store_phi_detections(consultation_id, phi_data.get("entities", []))

            # Store structured document if available
            if structured_document:
                soap_note = structured_document.get("soap_note", {})
                clinical_summary = structured_document.get("clinical_summary")
                await vector_store.store_structured_document(
                    consultation_id, structured_document, soap_note, clinical_summary
                )

            # Save index
            vector_store.save_index()

            result = {
                "consultation_id": consultation_id,
                "vector_id": vector_id,
                "stored_at": datetime.now().isoformat()
            }

            vector_store.close()
            return result

        except Exception as e:
            logging.error(f"Vector storage failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "Vector Storage")

@activity.defn
async def comprehensive_medical_processing_activity(
    transcript: str,
    consultation_id: str,
    patient_id_encrypted: str = None,
    provider_id: str = None,
    encounter_date: str = None,
    processing_options: dict = None
) -> dict:
    """Activity for comprehensive medical processing with parallel execution."""
    from app.config import Config

    if not Config.is_medical_processing_enabled():
        return {"skipped": True, "reason": "Medical processing not enabled"}

    processing_options = processing_options or {}
    enable_parallel = processing_options.get("parallel", Config.ENABLE_PARALLEL_MEDICAL_PROCESSING)

    async with TemporalMetrics.activity_timer("comprehensive_medical_processing", consultation_id):
        try:
            results = {
                "consultation_id": consultation_id,
                "processing_started": datetime.now().isoformat(),
                "parallel_processing": enable_parallel
            }

            if enable_parallel:
                # Execute PHI detection and entity extraction in parallel
                phi_task = asyncio.create_task(
                    phi_detection_activity(transcript, consultation_id)
                )
                entity_task = asyncio.create_task(
                    medical_entity_extraction_activity(transcript, consultation_id)
                )
                embedding_task = asyncio.create_task(
                    embedding_generation_activity(transcript, consultation_id)
                )

                # Wait for parallel tasks
                phi_result, entity_result, embedding_result = await asyncio.gather(
                    phi_task, entity_task, embedding_task, return_exceptions=True
                )

                results["phi_detection"] = phi_result if not isinstance(phi_result, Exception) else {"error": str(phi_result)}
                results["medical_entities"] = entity_result if not isinstance(entity_result, Exception) else {"error": str(entity_result)}
                results["embedding"] = embedding_result if not isinstance(embedding_result, Exception) else {"error": str(embedding_result)}

            else:
                # Sequential processing
                results["phi_detection"] = await phi_detection_activity(transcript, consultation_id)
                results["medical_entities"] = await medical_entity_extraction_activity(transcript, consultation_id)
                results["embedding"] = await embedding_generation_activity(transcript, consultation_id)

            # Document structuring (depends on previous results)
            phi_data = results["phi_detection"] if results["phi_detection"].get("phi_detected", False) else None
            entities = results["medical_entities"].get("entities", []) if not results["medical_entities"].get("skipped") else []

            results["document_structuring"] = await document_structuring_activity(
                transcript, consultation_id, phi_data, entities
            )

            # Generate SOAP note
            results["soap_note"] = await soap_generation_activity(transcript, consultation_id)

            # Vector storage (if enabled and we have all required data)
            if Config.ENABLE_VECTOR_STORAGE and all([
                patient_id_encrypted, provider_id, encounter_date,
                not results["embedding"].get("skipped")
            ]):
                embedding_data = results["embedding"].get("embedding", [])
                results["vector_storage"] = await vector_storage_activity(
                    consultation_id=consultation_id,
                    patient_id_encrypted=patient_id_encrypted,
                    provider_id=provider_id,
                    encounter_date=encounter_date,
                    transcript=transcript,
                    embedding=embedding_data,
                    phi_data=results["phi_detection"],
                    medical_entities=entities,
                    structured_document=results["document_structuring"].get("structured_document")
                )

            results["processing_completed"] = datetime.now().isoformat()

            # Calculate processing summary
            successful_tasks = [
                task for task, result in results.items()
                if isinstance(result, dict) and "error" not in result and not result.get("skipped")
            ]
            results["summary"] = {
                "total_tasks": len([k for k in results.keys() if k not in ["consultation_id", "processing_started", "parallel_processing", "processing_completed", "summary"]]),
                "successful_tasks": len(successful_tasks),
                "task_list": successful_tasks
            }

            return results

        except Exception as e:
            logging.error(f"Comprehensive medical processing failed: {e}")
            raise TemporalErrorHandler.create_application_error(e, "Comprehensive Medical Processing")
