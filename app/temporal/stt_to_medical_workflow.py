"""Unified STT-to-Medical Workflow with speaker-aware processing."""

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from datetime import timedelta
import logging
from typing import Dict, Any, Optional

from .config import TemporalConfig
from .monitoring import TemporalMetrics

logging.basicConfig(level=logging.INFO)


@workflow.defn
class STTToMedicalWorkflow:
    """
    Unified workflow: Audio → WhisperX → Speaker-Aware Medical Processing.

    This workflow combines transcription and medical processing with full
    speaker attribution and observability in Temporal UI.
    """

    @workflow.run
    async def run(
        self,
        audio_path: str,
        params: dict,
        enable_medical_processing: bool = False,
        medical_params: Optional[dict] = None,
    ) -> dict:
        """
        Execute complete STT → Medical pipeline.

        Args:
            audio_path: Path to audio/video file
            params: WhisperX processing parameters (model, ASR options, VAD, etc.)
            enable_medical_processing: Whether to run medical processing after WhisperX
            medical_params: Medical processing parameters (patient_id, provider_id, etc.)

        Returns:
            Combined results from WhisperX and medical processing

        The workflow executes:
        1. WhisperX: Transcribe → Align → Diarize → Assign Speakers
        2. If enable_medical_processing:
           - Transform to speaker-attributed dialogue
           - PHI detection (speaker-aware)
           - Entity extraction (speaker-aware)
           - SOAP generation (speaker-aware)
           - Vector storage (with speaker metadata)
        """
        from app.config import Config

        consultation_id = (medical_params or {}).get("consultation_id") or f"stt_{workflow.uuid4().hex[:8]}"

        TemporalMetrics.log_workflow_progress("stt_medical_workflow_started", consultation_id)

        try:
            results = {
                "consultation_id": consultation_id,
                "workflow_type": "stt_to_medical",
                "started_at": workflow.now().isoformat(),
                "audio_path": audio_path,
                "medical_processing_enabled": enable_medical_processing,
            }

            # ================================================================
            # STAGE 1: WhisperX Processing
            # ================================================================
            TemporalMetrics.log_workflow_progress("whisperx_started", consultation_id)

            # 1.1: Transcription
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
            results["whisperx_transcription"] = transcription_result
            TemporalMetrics.log_workflow_progress("transcription_complete", consultation_id)

            # 1.2: Alignment
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
            results["whisperx_alignment"] = aligned_result
            TemporalMetrics.log_workflow_progress("alignment_complete", consultation_id)

            # 1.3: Diarization
            diarization_result = await workflow.execute_activity(
                "diarize_activity",
                args=[audio_path, params["diarization_params"]],
                start_to_close_timeout=timedelta(minutes=TemporalConfig.DIARIZATION_TIMEOUT_MINUTES),
                heartbeat_timeout=timedelta(minutes=30),  # Allow long audio processing
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=15),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=5),
                    maximum_attempts=3,
                ),
            )
            results["whisperx_diarization"] = diarization_result
            TemporalMetrics.log_workflow_progress("diarization_complete", consultation_id)

            # 1.4: Speaker Assignment
            speaker_result = await workflow.execute_activity(
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
            results["whisperx_final"] = speaker_result
            TemporalMetrics.log_workflow_progress("whisperx_complete", consultation_id)

            # ================================================================
            # STAGE 2: Medical Processing (if enabled)
            # ================================================================
            if enable_medical_processing:
                if not medical_params:
                    raise ValueError("medical_params required when enable_medical_processing=True")

                TemporalMetrics.log_workflow_progress("medical_processing_started", consultation_id)

                # 2.1: Transform to Dialogue
                dialogue_data = await workflow.execute_activity(
                    "transform_to_dialogue_activity",
                    args=[
                        speaker_result,  # WhisperX final result with speakers
                        medical_params.get("workflow_id"),
                        medical_params.get("manual_speaker_mapping"),
                        {
                            "patient_id": medical_params.get("patient_id"),
                            "provider_id": medical_params.get("provider_id"),
                            "encounter_date": medical_params.get("encounter_date"),
                        },
                    ],
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=2),
                        maximum_attempts=2,
                    ),
                )
                results["dialogue_transformation"] = dialogue_data
                TemporalMetrics.log_workflow_progress("dialogue_transformation_complete", consultation_id)

                # 2.2: PHI Detection (speaker-aware)
                phi_result = await workflow.execute_activity(
                    "detect_phi_in_dialogue_activity",
                    args=[dialogue_data],
                    start_to_close_timeout=timedelta(minutes=3),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=1.5,
                        maximum_attempts=2,
                    ),
                )
                results["phi_detection"] = phi_result
                TemporalMetrics.log_workflow_progress("phi_detection_complete", consultation_id)

                # 2.3: Entity Extraction (speaker-aware)
                entities_result = await workflow.execute_activity(
                    "extract_entities_with_speaker_activity",
                    args=[dialogue_data],
                    start_to_close_timeout=timedelta(minutes=3),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=1.5,
                        maximum_attempts=2,
                    ),
                )
                results["entity_extraction"] = entities_result
                TemporalMetrics.log_workflow_progress("entity_extraction_complete", consultation_id)

                # 2.4: SOAP Generation (speaker-aware)
                soap_result = await workflow.execute_activity(
                    "generate_soap_from_dialogue_activity",
                    args=[dialogue_data],
                    start_to_close_timeout=timedelta(minutes=3),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=1.5,
                        maximum_attempts=2,
                    ),
                )
                results["soap_generation"] = soap_result
                TemporalMetrics.log_workflow_progress("soap_generation_complete", consultation_id)

                # 2.5: Vector Storage (with all speaker data)
                if Config.ENABLE_VECTOR_STORAGE:
                    storage_result = await workflow.execute_activity(
                        "store_consultation_with_speaker_data_activity",
                        args=[
                            consultation_id,
                            medical_params.get("patient_id_encrypted"),
                            medical_params.get("provider_id"),
                            medical_params.get("encounter_date", workflow.now().date().isoformat()),
                            dialogue_data,
                            phi_result,
                            entities_result,
                            soap_result,
                        ],
                        start_to_close_timeout=timedelta(minutes=2),
                        retry_policy=RetryPolicy(
                            initial_interval=timedelta(seconds=3),
                            maximum_attempts=3,  # Retryable for DB issues
                        ),
                    )
                    results["vector_storage"] = storage_result
                    TemporalMetrics.log_workflow_progress("vector_storage_complete", consultation_id)

                TemporalMetrics.log_workflow_progress("medical_processing_complete", consultation_id)

            # ================================================================
            # STAGE 3: Summary
            # ================================================================
            results["workflow_completed"] = workflow.now().isoformat()
            results["summary"] = self._generate_summary(results)

            TemporalMetrics.log_workflow_progress("completed", consultation_id)
            return results

        except ActivityError as e:
            logging.error(f"Activity failed in STT-to-Medical workflow for {consultation_id}: {e}")
            if isinstance(e.cause, ApplicationError) and e.cause.non_retryable:
                logging.error(f"Non-retryable error: {e.cause}")
                raise e
            raise e
        except Exception as e:
            logging.error(f"Unexpected error in STT-to-Medical workflow for {consultation_id}: {e}")
            raise e

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow execution summary."""
        summary = {
            "consultation_id": results["consultation_id"],
            "workflow_type": results["workflow_type"],
            "medical_processing_enabled": results["medical_processing_enabled"],
            "stages_completed": [],
            "outputs": {},
        }

        # WhisperX stages
        if "whisperx_transcription" in results:
            summary["stages_completed"].append("transcription")
            summary["outputs"]["transcript_length"] = len(results["whisperx_transcription"].get("text", ""))

        if "whisperx_alignment" in results:
            summary["stages_completed"].append("alignment")

        if "whisperx_diarization" in results:
            summary["stages_completed"].append("diarization")
            segments = results["whisperx_diarization"].get("segments", [])
            speakers = set(seg.get("speaker", "UNKNOWN") for seg in segments)
            summary["outputs"]["speakers_detected"] = len(speakers)

        if "whisperx_final" in results:
            summary["stages_completed"].append("speaker_assignment")

        # Medical stages
        if results["medical_processing_enabled"]:
            if "dialogue_transformation" in results:
                summary["stages_completed"].append("dialogue_transformation")
                dialogue = results["dialogue_transformation"]
                summary["outputs"]["dialogue_segments"] = len(dialogue.get("dialogue", []))
                summary["outputs"]["speaker_mapping"] = dialogue.get("speaker_mapping", {})

            if "phi_detection" in results:
                summary["stages_completed"].append("phi_detection")
                phi = results["phi_detection"]
                if not phi.get("skipped"):
                    summary["outputs"]["phi_detected"] = phi.get("phi_detected", False)
                    summary["outputs"]["phi_entity_count"] = len(phi.get("entities", []))

            if "entity_extraction" in results:
                summary["stages_completed"].append("entity_extraction")
                entities = results["entity_extraction"]
                if not entities.get("skipped"):
                    summary["outputs"]["medical_entity_count"] = entities.get("entity_count", 0)
                    summary["outputs"]["speaker_breakdown"] = entities.get("speaker_breakdown", {})

            if "soap_generation" in results:
                summary["stages_completed"].append("soap_generation")
                soap = results["soap_generation"]
                if not soap.get("skipped"):
                    summary["outputs"]["soap_sections"] = list(soap.get("soap_note", {}).keys())

            if "vector_storage" in results:
                summary["stages_completed"].append("vector_storage")
                summary["outputs"]["vector_id"] = results["vector_storage"].get("vector_id")

        summary["total_stages"] = len(summary["stages_completed"])
        return summary
