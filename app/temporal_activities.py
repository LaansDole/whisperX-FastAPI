
from temporalio import activity
from app.whisperx_services import (
    transcribe_with_whisper,
    diarize,
    align_whisper_output,
)
import whisperx
from app.schemas import (
    WhisperModelParams,
    ASROptions,
    VADOptions,
    AlignmentParams,
    DiarizationParams,
)

@activity.defn
async def transcribe_activity(
    audio_path: str,
    model_params: dict,
    asr_options: dict,
    vad_options: dict,
) -> dict:
    """Activity to transcribe audio."""
    from app.audio import process_audio_file

    audio = process_audio_file(audio_path)
    model_params_obj = WhisperModelParams(**model_params)
    asr_options_obj = ASROptions(**asr_options)
    vad_options_obj = VADOptions(**vad_options)

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

@activity.defn
async def align_activity(
    transcript: dict, audio_path: str, align_params: dict
) -> dict:
    """Activity to align transcript."""
    from app.audio import process_audio_file

    audio = process_audio_file(audio_path)
    align_params_obj = AlignmentParams(**align_params)
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

@activity.defn
async def diarize_activity(audio_path: str, diarize_params: dict) -> dict:
    """Activity to diarize audio."""
    from app.audio import process_audio_file

    audio = process_audio_file(audio_path)
    diarize_params_obj = DiarizationParams(**diarize_params)
    result = diarize(
        audio,
        device=diarize_params_obj.device,
        min_speakers=diarize_params_obj.min_speakers,
        max_speakers=diarize_params_obj.max_speakers,
    )
    return result.to_dict(orient="records")

@activity.defn
async def assign_speakers_activity(
    diarization_segments: dict, transcript: dict
) -> dict:
    """Activity to assign speakers."""
    result = whisperx.assign_word_speakers(diarization_segments, transcript)
    return result
