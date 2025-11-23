"""Temporal workflows for audio processing."""

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from datetime import timedelta
import logging

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
