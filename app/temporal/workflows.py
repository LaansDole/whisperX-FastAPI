"""Temporal workflows for audio processing."""

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from datetime import timedelta
import logging
import uuid
from datetime import datetime

from .config import TemporalConfig
from .monitoring import TemporalMetrics

logging.basicConfig(level=logging.INFO)


@workflow.defn
class WhisperXWorkflow:
    @workflow.run
    async def run(self, audio_path: str, params: dict) -> dict:
        TemporalMetrics.log_workflow_progress("started", audio_path)

        try:
            # Transcription step with extended timeout and retry
            transcription_result = await workflow.execute_activity(
                "transcribe_activity",
                args=[
                    audio_path,
                    params["whisper_model_params"],
                    params["asr_options"],
                    params["vad_options"],
                ],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.TRANSCRIPTION_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=10),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=5),
                    maximum_attempts=3,
                ),
            )
            TemporalMetrics.log_workflow_progress("transcription_complete", audio_path)

            # Alignment step
            aligned_result = await workflow.execute_activity(
                "align_activity",
                args=[
                    transcription_result,
                    audio_path,
                    params["alignment_params"],
                ],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.ALIGNMENT_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=5),
                    backoff_coefficient=1.5,
                    maximum_interval=timedelta(minutes=2),
                    maximum_attempts=2,
                ),
            )
            TemporalMetrics.log_workflow_progress("alignment_complete", audio_path)

            # Diarization step
            diarization_result = await workflow.execute_activity(
                "diarize_activity",
                args=[audio_path, params["diarization_params"]],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.DIARIZATION_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=15),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=5),
                    maximum_attempts=3,
                ),
            )
            TemporalMetrics.log_workflow_progress("diarization_complete", audio_path)

            # Speaker assignment step
            final_result = await workflow.execute_activity(
                "assign_speakers_activity",
                args=[diarization_result, aligned_result],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.SPEAKER_ASSIGNMENT_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=2),
                    backoff_coefficient=1.2,
                    maximum_interval=timedelta(seconds=30),
                    maximum_attempts=2,
                ),
            )
            TemporalMetrics.log_workflow_progress("speaker_assignment_complete", audio_path)

            TemporalMetrics.log_workflow_progress("completed", audio_path)
            return final_result

        except ActivityError as e:
            logging.error(f"Activity failed in workflow for {audio_path}: {e}")
            # Check if it's a non-retryable error
            if isinstance(e.cause, ApplicationError) and e.cause.non_retryable:
                logging.error(f"Non-retryable error encountered: {e.cause}")
                raise e
            else:
                # Let temporal handle the retry at workflow level if needed
                raise e
        except Exception as e:
            logging.error(f"Unexpected error in workflow for {audio_path}: {e}")
            raise e


@workflow.defn
class MedicalRAGWorkflow:
    """Workflow for medical RAG processing with optional audio transcription."""

    @workflow.run
    async def run(self, input_params: dict) -> dict:
        """
        Run medical RAG workflow.

        Args:
            input_params: Dictionary containing:
                - transcript: Optional pre-transcribed text
                - audio_path: Optional audio file path (if transcript not provided)
                - consultation_id: Unique consultation identifier
                - patient_id_encrypted: Encrypted patient identifier
                - provider_id: Provider identifier
                - encounter_date: Date of encounter
                - medical_options: Medical processing options
                - transcription_options: Options for audio transcription (if needed)
        """
        from app.config import Config

        consultation_id = input_params.get("consultation_id", f"med_{uuid.uuid4().hex[:8]}")
        TemporalMetrics.log_workflow_progress("started", consultation_id)

        try:
            results = {
                "consultation_id": consultation_id,
                "workflow_type": "medical_rag",
                "started_at": datetime.now(Config.TIMEZONE).isoformat(),
                "transcription_performed": False,
            }

            # Step 1: Get transcript (either provided or transcribe audio)
            transcript = input_params.get("transcript")
            if not transcript and input_params.get("audio_path"):
                # Need to transcribe audio first
                TemporalMetrics.log_workflow_progress("transcription_started", consultation_id)

                transcription_result = await workflow.execute_activity(
                    "transcribe_activity",
                    args=[
                        input_params["audio_path"],
                        input_params.get("transcription_options", {}).get("whisper_model_params", {}),
                        input_params.get("transcription_options", {}).get("asr_options", {}),
                        input_params.get("transcription_options", {}).get("vad_options", {}),
                    ],
                    start_to_close_timeout=timedelta(minutes=TemporalConfig.TRANSCRIPTION_TIMEOUT_MINUTES),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=10),
                        backoff_coefficient=2.0,
                        maximum_interval=timedelta(minutes=5),
                        maximum_attempts=3,
                    ),
                )

                transcript = transcription_result.get("text", "")
                results["transcription_result"] = transcription_result
                results["transcription_performed"] = True

                TemporalMetrics.log_workflow_progress("transcription_complete", consultation_id)

            elif not transcript:
                raise ValueError("Either transcript or audio_path must be provided")

            # Step 2: Comprehensive medical processing
            TemporalMetrics.log_workflow_progress("medical_processing_started", consultation_id)

            medical_options = input_params.get("medical_options", {})
            medical_result = await workflow.execute_activity(
                "comprehensive_medical_processing_activity",
                args=[
                    transcript,
                    consultation_id,
                    input_params.get("patient_id_encrypted"),
                    input_params.get("provider_id"),
                    input_params.get("encounter_date", datetime.now(Config.TIMEZONE).date().isoformat()),
                    medical_options,
                ],
                start_to_close_timeout=timedelta(minutes=Config.MEDICAL_WORKFLOW_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=5),
                    backoff_coefficient=1.5,
                    maximum_interval=timedelta(minutes=2),
                    maximum_attempts=2,
                ),
            )

            results.update(medical_result)
            TemporalMetrics.log_workflow_progress("medical_processing_complete", consultation_id)

            # Step 3: Final assembly and summary
            results["workflow_completed"] = datetime.now(Config.TIMEZONE).isoformat()

            # Generate processing summary
            summary = self._generate_processing_summary(results)
            results["processing_summary"] = summary

            TemporalMetrics.log_workflow_progress("completed", consultation_id)
            return results

        except ActivityError as e:
            logging.error(f"Activity failed in medical workflow for {consultation_id}: {e}")
            if isinstance(e.cause, ApplicationError) and e.cause.non_retryable:
                logging.error(f"Non-retryable error encountered: {e.cause}")
                raise e
            else:
                raise e
        except Exception as e:
            logging.error(f"Unexpected error in medical workflow for {consultation_id}: {e}")
            raise e

    def _generate_processing_summary(self, results: dict) -> dict:
        """Generate summary of medical processing results."""
        summary = {
            "consultation_id": results["consultation_id"],
            "workflow_type": results["workflow_type"],
            "success": True,
            "processing_stages": {},
            "key_outputs": {},
        }

        # Analyze each processing stage
        if "transcription_performed" in results:
            summary["processing_stages"]["transcription"] = {
                "performed": results["transcription_performed"],
                "success": "transcription_result" in results,
            }

        # Medical processing stages
        if "phi_detection" in results:
            phi_result = results["phi_detection"]
            summary["processing_stages"]["phi_detection"] = {
                "performed": not phi_result.get("skipped", False),
                "phi_detected": phi_result.get("phi_detected", False),
                "entities_found": len(phi_result.get("entities", [])),
                "success": "error" not in phi_result,
            }

        if "medical_entities" in results:
            entity_result = results["medical_entities"]
            summary["processing_stages"]["entity_extraction"] = {
                "performed": not entity_result.get("skipped", False),
                "entities_extracted": entity_result.get("entity_count", 0),
                "success": "error" not in entity_result,
            }

        if "soap_note" in results:
            soap_result = results["soap_note"]
            summary["processing_stages"]["soap_generation"] = {
                "performed": not soap_result.get("skipped", False),
                "success": "error" not in soap_result,
            }

        if "vector_storage" in results:
            vector_result = results["vector_storage"]
            summary["processing_stages"]["vector_storage"] = {
                "performed": not vector_result.get("skipped", False),
                "success": "error" not in vector_result,
            }

        # Key outputs for easy access
        if results.get("medical_entities", {}).get("entities"):
            summary["key_outputs"]["diagnoses"] = [
                e for e in results["medical_entities"]["entities"] if e.get("type") == "diagnosis"
            ]

        if results.get("medical_entities", {}).get("entities"):
            summary["key_outputs"]["medications"] = [
                e for e in results["medical_entities"]["entities"] if e.get("type") == "medication"
            ]

        if results.get("soap_note", {}).get("soap_note"):
            summary["key_outputs"]["clinical_summary"] = results["soap_note"]["soap_note"].get("assessment", "")

        return summary


@workflow.defn
class HybridAudioMedicalWorkflow:
    """Combined workflow that processes audio through WhisperX and then through medical RAG."""

    @workflow.run
    async def run(self, audio_path: str, params: dict) -> dict:
        """
        Run hybrid workflow: WhisperX processing + Medical RAG.

        Args:
            audio_path: Path to audio/video file
            params: Dictionary containing both WhisperX and medical parameters
        """
        consultation_id = params.get("consultation_id", f"hybrid_{uuid.uuid4().hex[:8]}")
        TemporalMetrics.log_workflow_progress("hybrid_workflow_started", consultation_id)

        from app.config import Config

        try:
            results = {
                "consultation_id": consultation_id,
                "workflow_type": "hybrid_audio_medical",
                "started_at": datetime.now(Config.TIMEZONE).isoformat(),
                "audio_path": audio_path,
            }

            # Stage 1: WhisperX processing (transcription, alignment, diarization)
            TemporalMetrics.log_workflow_progress("whisperx_processing_started", consultation_id)

            whisperx_result = await workflow.execute_activity(
                "transcribe_activity",
                args=[
                    audio_path,
                    params["whisper_model_params"],
                    params["asr_options"],
                    params["vad_options"],
                ],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.TRANSCRIPTION_TIMEOUT_MINUTES),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=10),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=5),
                    maximum_attempts=3,
                ),
            )

            results["whisperx_transcription"] = whisperx_result
            TemporalMetrics.log_workflow_progress("whisperx_transcription_complete", consultation_id)

            # Optional: Alignment and diarization if enabled
            if params.get("enable_alignment", True):
                aligned_result = await workflow.execute_activity(
                    "align_activity",
                    args=[
                        whisperx_result,
                        audio_path,
                        params["alignment_params"],
                    ],
                    start_to_close_timeout=timedelta(minutes=TemporalConfig.ALIGNMENT_TIMEOUT_MINUTES),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=1.5,
                        maximum_interval=timedelta(minutes=2),
                        maximum_attempts=2,
                    ),
                )
                results["whisperx_alignment"] = aligned_result
                TemporalMetrics.log_workflow_progress("whisperx_alignment_complete", consultation_id)

            if params.get("enable_diarization", True):
                diarization_result = await workflow.execute_activity(
                    "diarize_activity",
                    args=[audio_path, params["diarize_params"]],
                    start_to_close_timeout=timedelta(minutes=TemporalConfig.DIARIZATION_TIMEOUT_MINUTES),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=15),
                        backoff_coefficient=2.0,
                        maximum_interval=timedelta(minutes=5),
                        maximum_attempts=3,
                    ),
                )
                results["whisperx_diarization"] = diarization_result
                TemporalMetrics.log_workflow_progress("whisperx_diarization_complete", consultation_id)

                # Speaker assignment if diarization was performed
                if "whisperx_alignment" in results:
                    speaker_result = await workflow.execute_activity(
                        "assign_speakers_activity",
                        args=[diarization_result, results["whisperx_alignment"]],
                        start_to_close_timeout=timedelta(minutes=TemporalConfig.SPEAKER_ASSIGNMENT_TIMEOUT_MINUTES),
                        retry_policy=RetryPolicy(
                            initial_interval=timedelta(seconds=2),
                            backoff_coefficient=1.2,
                            maximum_interval=timedelta(seconds=30),
                            maximum_attempts=2,
                        ),
                    )
                    results["whisperx_speakers"] = speaker_result
                    TemporalMetrics.log_workflow_progress("whisperx_speakers_complete", consultation_id)

            # Stage 2: Medical RAG processing

            if Config.is_medical_processing_enabled():
                TemporalMetrics.log_workflow_progress("medical_processing_started", consultation_id)

                # Extract transcript for medical processing
                transcript = whisperx_result.get("text", "")

                medical_params = {
                    "transcript": transcript,
                    "consultation_id": consultation_id,
                    "patient_id_encrypted": params.get("patient_id_encrypted"),
                    "provider_id": params.get("provider_id"),
                    "encounter_date": params.get("encounter_date", datetime.now(Config.TIMEZONE).date().isoformat()),
                    "medical_options": params.get("medical_options", {}),
                }

                medical_result = await workflow.execute_activity(
                    "comprehensive_medical_processing_activity",
                    args=[
                        medical_params["transcript"],
                        medical_params["consultation_id"],
                        medical_params["patient_id_encrypted"],
                        medical_params["provider_id"],
                        medical_params["encounter_date"],
                        medical_params["medical_options"],
                    ],
                    start_to_close_timeout=timedelta(minutes=Config.MEDICAL_WORKFLOW_TIMEOUT_MINUTES),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=1.5,
                        maximum_interval=timedelta(minutes=2),
                        maximum_attempts=2,
                    ),
                )

                results["medical_processing"] = medical_result
                TemporalMetrics.log_workflow_progress("medical_processing_complete", consultation_id)

            # Final summary
            results["workflow_completed"] = datetime.now(Config.TIMEZONE).isoformat()

            # Generate comprehensive summary
            summary = self._generate_hybrid_summary(results)
            results["comprehensive_summary"] = summary

            TemporalMetrics.log_workflow_progress("completed", consultation_id)
            return results

        except Exception as e:
            logging.error(f"Hybrid workflow failed for {consultation_id}: {e}")
            raise

    def _generate_hybrid_summary(self, results: dict) -> dict:
        """Generate comprehensive summary of hybrid processing."""
        summary = {
            "consultation_id": results["consultation_id"],
            "workflow_type": results["workflow_type"],
            "success": True,
            "stages_completed": [],
            "final_outputs": {},
        }

        # WhisperX stage
        if "whisperx_transcription" in results:
            summary["stages_completed"].append("transcription")
            summary["final_outputs"]["raw_transcript"] = results["whisperx_transcription"].get("text", "")

        if "whisperx_alignment" in results:
            summary["stages_completed"].append("alignment")

        if "whisperx_diarization" in results:
            summary["stages_completed"].append("diarization")
            # Count speakers
            segments = results["whisperx_diarization"].get("segments", [])
            speakers = set(seg.get("speaker", "UNKNOWN") for seg in segments)
            summary["final_outputs"]["speakers_detected"] = list(speakers)

        # Medical stage
        if "medical_processing" in results:
            summary["stages_completed"].append("medical_rag")
            medical = results["medical_processing"]

            if medical.get("soap_note", {}).get("soap_note"):
                summary["final_outputs"]["soap_note"] = medical["soap_note"]["soap_note"]

            if medical.get("medical_entities", {}).get("entities"):
                summary["final_outputs"]["medical_entities"] = medical["medical_entities"]["entities"]

            if medical.get("phi_detection", {}).get("phi_detected"):
                summary["final_outputs"]["phi_detected"] = True
                summary["final_outputs"]["phi_entities"] = medical["phi_detection"]["entities"]

        return summary
