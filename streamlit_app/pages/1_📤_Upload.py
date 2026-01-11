"""
Upload Audio File Page

Upload audio/video files for transcription with optional medical processing.
Supports both file upload and live voice recording from microphone.
"""

import streamlit as st
from datetime import date, datetime
import httpx
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client

# Page config
st.set_page_config(page_title="Upload Audio - whisperX", page_icon="📤", layout="wide")

# Initialize API client
api_client = get_api_client()

with st.sidebar:
    st.header("⚙️ Transcription Settings")

    # Model selection
    model_options = {
        "tiny": "Tiny (fastest)",
        "base": "Base (recommended)",
        "small": "Small",
        "medium": "Medium",
        "large-v3": "Large-v3 (best)",
        "large-v3-turbo": "Large-v3 Turbo",
    }
    selected_model = st.selectbox(
        "Whisper Model",
        options=list(model_options.keys()),
        index=1,  # Default to 'base'
        format_func=lambda x: model_options[x],
        help="Larger = more accurate but slower",
        key="whisper_model",
    )

    # Language selection
    language_options = {
        "en": "🇬🇧 English",
        "vi": "🇻🇳 Vietnamese",
        "zh": "🇨🇳 Chinese",
        "yue": "🇭🇰 Cantonese",
    }
    selected_language = st.selectbox(
        "Language",
        options=list(language_options.keys()),
        index=0,
        format_func=lambda x: language_options[x],
        key="whisper_language",
    )

    st.divider()
    st.subheader("🎤 Speaker Diarization")

    col1, col2 = st.columns(2)
    with col1:
        min_speakers = st.number_input(
            "Min",
            min_value=0,
            max_value=10,
            value=0,
            help="0 = auto",
            key="min_speakers",
        )

    with col2:
        max_speakers = st.number_input(
            "Max",
            min_value=0,
            max_value=10,
            value=0,
            help="0 = auto",
            key="max_speakers",
        )

    st.divider()
    initial_prompt = st.text_area(
        "Initial Prompt",
        placeholder="Medical terms, names...",
        help="Vocabulary hints for better accuracy",
        key="initial_prompt",
        height=80,
    )


# Helper function to get current settings
def get_transcription_settings():
    """Get current transcription settings from session state."""
    return {
        "model": st.session_state.get("whisper_model", "base"),
        "language": st.session_state.get("whisper_language", "en"),
        "min_speakers": st.session_state.get("min_speakers", 0) or None,
        "max_speakers": st.session_state.get("max_speakers", 0) or None,
        "initial_prompt": st.session_state.get("initial_prompt", "").strip() or None,
    }


st.title("📤 Upload Audio")
st.markdown("Upload an audio file or record directly from your microphone for transcription.")

# Tab-based input selection
upload_tab, record_tab = st.tabs(["📁 Upload File", "🎙️ Record Audio"])

with upload_tab:
    with st.form("upload_form", clear_on_submit=True):
        st.subheader("File Upload")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=["mp3", "wav", "m4a", "flac", "ogg", "opus", "webm", "mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM, MP4, AVI, MOV, MKV",
        )

        if uploaded_file:
            file_details = {
                "Filename": uploaded_file.name,
                "File type": uploaded_file.type,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
            }
            st.json(file_details)

        # Patient information
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            patient_name_upload = st.text_input(
                "Patient Full Name *",
                help="Required for HIPAA-compliant identification",
                placeholder="John Michael Smith",
                key="patient_name_upload",
            )

        with col2:
            encounter_date_upload = st.date_input(
                "Encounter Date",
                value=date.today(),
                help="Date of consultation",
                key="encounter_date_upload",
            )

        # Medical processing options
        st.subheader("Processing Options")

        enable_medical_upload = st.checkbox(
            "Enable Medical Processing",
            help="Enable PHI detection, medical entity extraction, and SOAP note generation",
            value=False,
            key="enable_medical_upload",
        )

        provider_id_upload = st.text_input(
            "Provider ID" + (" *" if enable_medical_upload else ""),
            value="DR001",
            help="Healthcare provider identifier (required when medical processing is enabled)",
            placeholder="e.g., DR001, PROV123",
            key="provider_id_upload",
        )

        # Submit button
        submitted_upload = st.form_submit_button("Upload & Process", type="primary", use_container_width=True)

        # Form validation and submission
        if submitted_upload:
            if not uploaded_file:
                st.error("❌ Please select a file to upload")
            elif not patient_name_upload or len(patient_name_upload.strip()) == 0:
                st.error("❌ Patient name is required")
            elif enable_medical_upload and (not provider_id_upload or len(provider_id_upload.strip()) == 0):
                st.error("❌ Provider ID is required when medical processing is enabled")
            else:
                with st.spinner("Uploading file and starting workflow..."):
                    try:
                        file_bytes = uploaded_file.read()
                        encounter_date_str = encounter_date_upload.isoformat()
                        settings = get_transcription_settings()

                        response = api_client.upload_audio(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            patient_name=patient_name_upload.strip(),
                            enable_medical=enable_medical_upload,
                            provider_id=provider_id_upload.strip() if provider_id_upload else None,
                            encounter_date=encounter_date_str,
                            model=settings["model"],
                            language=settings["language"],
                            min_speakers=settings["min_speakers"],
                            max_speakers=settings["max_speakers"],
                            initial_prompt=settings["initial_prompt"],
                        )

                        st.success("✅ Upload successful!")
                        workflow_id = response.get("identifier", "N/A")
                        message = response.get("message", "Workflow started")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Model**: {settings["model"]} | **Language**: {settings["language"]}
                        """)

                        st.info("ℹ️ Go to **Workflows** page to track progress.")

                        if "recent_uploads" not in st.session_state:
                            st.session_state.recent_uploads = []

                        st.session_state.recent_uploads.insert(
                            0,
                            {
                                "workflow_id": workflow_id,
                                "patient_name": patient_name_upload,
                                "filename": uploaded_file.name,
                                "medical_enabled": enable_medical_upload,
                                "source": "upload",
                                "model": settings["model"],
                            },
                        )
                        st.session_state.recent_uploads = st.session_state.recent_uploads[:10]

                    except httpx.HTTPError as e:
                        st.error(f"❌ Upload failed: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

with record_tab:
    st.subheader("🎙️ Record from Microphone")
    st.markdown("Click the microphone button to start recording.")

    recorded_audio = st.audio_input(
        "Record your audio",
        sample_rate=16000,
        help="Audio captured at 16kHz (optimal for speech recognition)",
    )

    if recorded_audio:
        st.success("✅ Recording captured!")
        st.audio(recorded_audio)
        st.caption(f"Size: {recorded_audio.size / 1024:.2f} KB")

    with st.form("recording_form", clear_on_submit=False):
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            patient_name_record = st.text_input(
                "Patient Full Name *",
                placeholder="John Michael Smith",
                key="patient_name_record",
            )

        with col2:
            encounter_date_record = st.date_input(
                "Encounter Date",
                value=date.today(),
                key="encounter_date_record",
            )

        st.subheader("Processing Options")

        enable_medical_record = st.checkbox(
            "Enable Medical Processing",
            value=False,
            key="enable_medical_record",
        )

        provider_id_record = st.text_input(
            "Provider ID" + (" *" if enable_medical_record else ""),
            value="DR001",
            help="Healthcare provider identifier (required when medical processing is enabled)",
            placeholder="e.g., DR001, PROV123",
            key="provider_id_record",
        )

        submitted_record = st.form_submit_button("Submit Recording", type="primary", use_container_width=True)

        if submitted_record:
            if not recorded_audio:
                st.error("❌ Please record audio before submitting")
            elif not patient_name_record or len(patient_name_record.strip()) == 0:
                st.error("❌ Patient name is required")
            elif enable_medical_record and (not provider_id_record or len(provider_id_record.strip()) == 0):
                st.error("❌ Provider ID is required")
            else:
                with st.spinner("Uploading recording..."):
                    try:
                        audio_bytes = recorded_audio.read()
                        filename = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        settings = get_transcription_settings()

                        response = api_client.upload_audio(
                            file_bytes=audio_bytes,
                            filename=filename,
                            patient_name=patient_name_record.strip(),
                            enable_medical=enable_medical_record,
                            provider_id=provider_id_record.strip() if provider_id_record else None,
                            encounter_date=encounter_date_record.isoformat(),
                            model=settings["model"],
                            language=settings["language"],
                            min_speakers=settings["min_speakers"],
                            max_speakers=settings["max_speakers"],
                            initial_prompt=settings["initial_prompt"],
                        )

                        st.success("✅ Recording submitted!")
                        workflow_id = response.get("identifier", "N/A")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Model**: {settings["model"]} | **Language**: {settings["language"]}
                        """)

                        st.info("ℹ️ Go to **Workflows** page to track progress.")

                        if "recent_uploads" not in st.session_state:
                            st.session_state.recent_uploads = []

                        st.session_state.recent_uploads.insert(
                            0,
                            {
                                "workflow_id": workflow_id,
                                "patient_name": patient_name_record,
                                "filename": filename,
                                "medical_enabled": enable_medical_record,
                                "source": "recording",
                                "model": settings["model"],
                            },
                        )
                        st.session_state.recent_uploads = st.session_state.recent_uploads[:10]

                    except httpx.HTTPError as e:
                        st.error(f"❌ Upload failed: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

if "recent_uploads" in st.session_state and st.session_state.recent_uploads:
    st.divider()
    st.subheader("📝 Recent Uploads")

    for idx, upload in enumerate(st.session_state.recent_uploads):
        source_icon = "🎙️" if upload.get("source") == "recording" else "📁"
        with st.expander(f"{source_icon} {upload['filename']} [{upload.get('model', 'base')}]"):
            st.markdown(f"""
            - **Workflow ID**: `{upload["workflow_id"]}`
            - **Patient**: {upload["patient_name"]}
            - **Model**: {upload.get("model", "base")}
            - **Medical**: {"✅" if upload["medical_enabled"] else "❌"}
            """)

# Footer
st.caption("⚠️ **HIPAA**: Patient names are stored as cryptographic hashes.")
