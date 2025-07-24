
from temporalio import workflow
from datetime import timedelta
from app.temporal_activities import (
    transcribe_activity,
    align_activity,
    diarize_activity,
    assign_speakers_activity,
)

@workflow.defn
class WhisperXWorkflow:
    @workflow.run
    async def run(self, audio_path: str, params: dict) -> dict:
        transcription_result = await workflow.execute_activity(
            transcribe_activity,
            args=[
                audio_path,
                params["whisper_model_params"],
                params["asr_options"],
                params["vad_options"],
            ],
            start_to_close_timeout=timedelta(minutes=15),
        )

        aligned_result = await workflow.execute_activity(
            align_activity,
            args=[
                transcription_result,
                audio_path,
                params["alignment_params"],
            ],
            start_to_close_timeout=timedelta(minutes=5),
        )

        diarization_result = await workflow.execute_activity(
            diarize_activity,
            args=[audio_path, params["diarization_params"]],
            start_to_close_timeout=timedelta(minutes=5),
        )

        final_result = await workflow.execute_activity(
            assign_speakers_activity,
            args=[diarization_result, aligned_result],
            start_to_close_timeout=timedelta(minutes=2),
        )

        return final_result
