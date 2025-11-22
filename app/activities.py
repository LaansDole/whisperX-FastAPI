
from temporalio import activity
from app.temporal_error_handler import TemporalErrorHandler
from app.temporal_monitoring import TemporalMetrics
from app.whisperx_services import (
    transcribe_with_whisper,
    transcribe_with_meralion_fallback,
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
import logging

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
    async with TemporalMetrics.activity_timer("speaker_assignment"):
        try:
            # Extract the segments list from the diarization result
            segments_list = diarization_segments.get("segments", [])
            
            # Convert back to DataFrame format that whisperx.assign_word_speakers expects
            import pandas as pd
            segments_df = pd.DataFrame(segments_list)
            
            result = whisperx.assign_word_speakers(segments_df, transcript)
            return result
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "Speaker assignment")




@activity.defn
async def transcribe_meralion_activity(
    audio_path: str,
    model_params: dict,
    asr_options: dict,
    vad_options: dict,
) -> dict:
    """Activity to transcribe audio using MERaLiON with fallback."""
    from app.audio import process_audio_file
    
    async with TemporalMetrics.activity_timer("transcription_meralion"):
        try:
            # Process audio file
            audio = process_audio_file(audio_path)
            
            # Extract parameters
            language = model_params.get("language", "en")
            task = model_params.get("task", "transcribe")
            device = model_params.get("device", "auto")
            device_index = model_params.get("device_index", 0)
            batch_size = model_params.get("batch_size", 8)
            threads = model_params.get("threads", 0)
            use_meralion = model_params.get("use_meralion", True)
            meralion_fallback_enabled = model_params.get("meralion_fallback_enabled", True)
            meralion_max_new_tokens = model_params.get("meralion_max_new_tokens", 256)
            override_model = model_params.get("model")
            
            # Use MERaLiON transcription with fallback
            result = transcribe_with_meralion_fallback(
                audio=audio,
                task=task,
                language=language,
                device=device,
                device_index=device_index,
                asr_options=asr_options,
                vad_options=vad_options,
                batch_size=batch_size,
                threads=threads,
                use_meralion=use_meralion,
                meralion_fallback_enabled=meralion_fallback_enabled,
                meralion_max_new_tokens=meralion_max_new_tokens,
                override_model=override_model,
            )
            
            return result
        except Exception as e:
            raise TemporalErrorHandler.create_application_error(e, "MERaLiON transcription")


