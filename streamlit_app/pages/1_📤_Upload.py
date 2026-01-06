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

st.title("📤 Upload Audio")
st.markdown("Upload an audio file or record directly from your microphone for transcription.")

# Initialize API client
api_client = get_api_client()

# ============================================================================
# SHARED: Transcription Settings (applies to both tabs)
# ============================================================================
with st.expander("⚙️ Transcription Settings", expanded=False):
    st.markdown("Configure WhisperX model and processing options.")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Model selection with descriptions
        model_options = {
            "tiny": "Tiny - Fastest, lowest accuracy",
            "base": "Base - Fast, good accuracy (recommended)",
            "small": "Small - Balanced speed/accuracy",
            "medium": "Medium - Higher accuracy, slower",
            "large-v3": "Large-v3 - Best accuracy, slowest",
            "large-v3-turbo": "Large-v3 Turbo - Fast + high accuracy",
        }
        selected_model = st.selectbox(
            "Whisper Model",
            options=list(model_options.keys()),
            index=1,  # Default to 'base'
            format_func=lambda x: model_options[x],
            help="Larger models are more accurate but slower. Use 'base' for most cases.",
            key="whisper_model",
        )

    with col2:
        # Language selection
        language_options = {
            "en": "🇬🇧 English",
            "vi": "🇻🇳 Vietnamese",
            "zh": "🇨🇳 Chinese (Mandarin)",
            "yue": "🇭🇰 Cantonese",
        }
        selected_language = st.selectbox(
            "Language",
            options=list(language_options.keys()),
            index=0,  # Default to English
            format_func=lambda x: language_options[x],
            help="Select the primary language spoken in the audio.",
            key="whisper_language",
        )

    with col3:
        # Compute type
        compute_options = {
            "int8": "int8 - Fast (recommended for CPU)",
            "float16": "float16 - Balanced (GPU recommended)",
            "float32": "float32 - Most accurate, slowest",
        }
        selected_compute = st.selectbox(
            "Compute Type",
            options=list(compute_options.keys()),
            index=0,  # Default to int8
            format_func=lambda x: compute_options[x],
            help="Computation precision. Use int8 for CPU, float16 for GPU.",
            key="compute_type",
        )

    # Speaker diarization settings
    st.markdown("---")
    st.markdown("**Speaker Settings** (for diarization)")

    col1, col2 = st.columns(2)

    with col1:
        min_speakers = st.number_input(
            "Minimum Speakers",
            min_value=0,
            max_value=10,
            value=0,
            help="Minimum expected speakers. 0 = auto-detect.",
            key="min_speakers",
        )
        # Convert 0 to None for API
        min_speakers_val = min_speakers if min_speakers > 0 else None

    with col2:
        max_speakers = st.number_input(
            "Maximum Speakers",
            min_value=0,
            max_value=10,
            value=0,
            help="Maximum expected speakers. 0 = auto-detect.",
            key="max_speakers",
        )
        # Convert 0 to None for API
        max_speakers_val = max_speakers if max_speakers > 0 else None

    # Initial prompt for domain-specific vocabulary
    st.markdown("---")
    initial_prompt = st.text_area(
        "Initial Prompt (Optional)",
        placeholder="e.g., Medical terms: hypertension, metformin, diabetes mellitus...",
        help="Provide context or vocabulary hints to improve transcription accuracy for domain-specific terms.",
        key="initial_prompt",
        height=80,
    )
    initial_prompt_val = initial_prompt.strip() if initial_prompt else None

# Tab-based input selection
upload_tab, record_tab = st.tabs(["📁 Upload File", "🎙️ Record Audio"])


# Helper function to get current settings
def get_transcription_settings():
    """Get current transcription settings from session state."""
    return {
        "model": st.session_state.get("whisper_model", "base"),
        "language": st.session_state.get("whisper_language", "en"),
        "compute_type": st.session_state.get("compute_type", "int8"),
        "min_speakers": st.session_state.get("min_speakers", 0) or None,
        "max_speakers": st.session_state.get("max_speakers", 0) or None,
        "initial_prompt": st.session_state.get("initial_prompt", "").strip() or None,
    }


# ============================================================================
# TAB 1: File Upload
# ============================================================================
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
                help="Required for HIPAA-compliant identification (submitted securely in request body)",
                placeholder="John Michael Smith",
                key="patient_name_upload",
            )

        with col2:
            encounter_date_upload = st.date_input(
                "Encounter Date", value=date.today(), help="Date of consultation (defaults to today)", key="encounter_date_upload"
            )

        # Medical processing options
        st.subheader("Processing Options")

        enable_medical_upload = st.checkbox(
            "Enable Medical Processing",
            help="Enable PHI detection, medical entity extraction, and SOAP note generation",
            value=False,
            key="enable_medical_upload",
        )

        provider_id_upload = None
        if enable_medical_upload:
            st.info("ℹ️ Medical processing requires a Provider ID")
            provider_id_upload = st.text_input(
                "Provider ID *",
                help="Healthcare provider identifier (required for medical processing)",
                placeholder="DR001",
                key="provider_id_upload",
            )

        # Submit button
        submitted_upload = st.form_submit_button("Upload & Process", type="primary", use_container_width=True)

        # Form validation and submission
        if submitted_upload:
            # Validate required fields
            if not uploaded_file:
                st.error("❌ Please select a file to upload")
            elif not patient_name_upload or len(patient_name_upload.strip()) == 0:
                st.error("❌ Patient name is required")
            elif enable_medical_upload and (not provider_id_upload or len(provider_id_upload.strip()) == 0):
                st.error("❌ Provider ID is required when medical processing is enabled")
            else:
                # All validations passed, proceed with upload
                with st.spinner("Uploading file and starting workflow..."):
                    try:
                        # Read file bytes
                        file_bytes = uploaded_file.read()

                        # Format encounter date
                        encounter_date_str = encounter_date_upload.isoformat()

                        # Get transcription settings
                        settings = get_transcription_settings()

                        # Upload to backend
                        response = api_client.upload_audio(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            patient_name=patient_name_upload.strip(),
                            enable_medical=enable_medical_upload,
                            provider_id=provider_id_upload.strip() if provider_id_upload else None,
                            encounter_date=encounter_date_str,
                            # Transcription settings
                            model=settings["model"],
                            language=settings["language"],
                            compute_type=settings["compute_type"],
                            min_speakers=settings["min_speakers"],
                            max_speakers=settings["max_speakers"],
                            initial_prompt=settings["initial_prompt"],
                        )

                        # Success!
                        st.success("✅ Upload successful!")

                        workflow_id = response.get("identifier", "N/A")
                        message = response.get("message", "Workflow started")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Status**: {message}
                        **Model**: {settings["model"]} | **Language**: {settings["language"]}
                        **Medical Processing**: {"Enabled" if enable_medical_upload else "Disabled"}
                        """)

                        # Show workflow tracking link
                        st.info("ℹ️ Your workflow is being processed. Go to the **Workflows** page to track progress.")

                        # Store workflow ID in session state for easy access
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

                        # Keep only last 10
                        st.session_state.recent_uploads = st.session_state.recent_uploads[:10]

                    except httpx.HTTPError as e:
                        st.error(f"❌ Upload failed: {str(e)}")
                        st.caption("Please ensure the backend is running and accessible")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

# ============================================================================
# TAB 2: Voice Recording
# ============================================================================
with record_tab:
    st.subheader("🎙️ Record from Microphone")
    st.markdown("Click the microphone button to start recording. Click again to stop.")

    # Audio input widget
    recorded_audio = st.audio_input(
        "Record your audio",
        sample_rate=16000,  # Optimal for speech recognition
        help="Click to start recording. Audio will be captured at 16kHz (optimal for speech recognition).",
    )

    if recorded_audio:
        st.success("✅ Recording captured!")
        st.audio(recorded_audio)

        # Recording details
        recording_size = recorded_audio.size
        st.caption(f"Recording size: {recording_size / 1024:.2f} KB")

    with st.form("recording_form", clear_on_submit=False):
        # Patient information
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            patient_name_record = st.text_input(
                "Patient Full Name *",
                help="Required for HIPAA-compliant identification (submitted securely in request body)",
                placeholder="John Michael Smith",
                key="patient_name_record",
            )

        with col2:
            encounter_date_record = st.date_input(
                "Encounter Date", value=date.today(), help="Date of consultation (defaults to today)", key="encounter_date_record"
            )

        # Medical processing options
        st.subheader("Processing Options")

        enable_medical_record = st.checkbox(
            "Enable Medical Processing",
            help="Enable PHI detection, medical entity extraction, and SOAP note generation",
            value=False,
            key="enable_medical_record",
        )

        provider_id_record = None
        if enable_medical_record:
            st.info("ℹ️ Medical processing requires a Provider ID")
            provider_id_record = st.text_input(
                "Provider ID *",
                help="Healthcare provider identifier (required for medical processing)",
                placeholder="DR001",
                key="provider_id_record",
            )

        # Submit button
        submitted_record = st.form_submit_button("Submit Recording", type="primary", use_container_width=True)

        # Form validation and submission
        if submitted_record:
            # Validate required fields
            if not recorded_audio:
                st.error("❌ Please record audio before submitting")
            elif not patient_name_record or len(patient_name_record.strip()) == 0:
                st.error("❌ Patient name is required")
            elif enable_medical_record and (not provider_id_record or len(provider_id_record.strip()) == 0):
                st.error("❌ Provider ID is required when medical processing is enabled")
            else:
                # All validations passed, proceed with upload
                with st.spinner("Uploading recording and starting workflow..."):
                    try:
                        # Read audio bytes
                        audio_bytes = recorded_audio.read()

                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"recorded_{timestamp}.wav"

                        # Format encounter date
                        encounter_date_str = encounter_date_record.isoformat()

                        # Get transcription settings
                        settings = get_transcription_settings()

                        # Upload to backend
                        response = api_client.upload_audio(
                            file_bytes=audio_bytes,
                            filename=filename,
                            patient_name=patient_name_record.strip(),
                            enable_medical=enable_medical_record,
                            provider_id=provider_id_record.strip() if provider_id_record else None,
                            encounter_date=encounter_date_str,
                            # Transcription settings
                            model=settings["model"],
                            language=settings["language"],
                            compute_type=settings["compute_type"],
                            min_speakers=settings["min_speakers"],
                            max_speakers=settings["max_speakers"],
                            initial_prompt=settings["initial_prompt"],
                        )

                        # Success!
                        st.success("✅ Recording submitted successfully!")

                        workflow_id = response.get("identifier", "N/A")
                        message = response.get("message", "Workflow started")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Status**: {message}
                        **Model**: {settings["model"]} | **Language**: {settings["language"]}
                        **Medical Processing**: {"Enabled" if enable_medical_record else "Disabled"}
                        """)

                        # Show workflow tracking link
                        st.info("ℹ️ Your workflow is being processed. Go to the **Workflows** page to track progress.")

                        # Store workflow ID in session state for easy access
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

                        # Keep only last 10
                        st.session_state.recent_uploads = st.session_state.recent_uploads[:10]

                    except httpx.HTTPError as e:
                        st.error(f"❌ Upload failed: {str(e)}")
                        st.caption("Please ensure the backend is running and accessible")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

# ============================================================================
# Recent uploads section (shared between tabs)
# ============================================================================
if "recent_uploads" in st.session_state and st.session_state.recent_uploads:
    st.divider()
    st.subheader("📝 Recent Uploads (This Session)")

    for idx, upload in enumerate(st.session_state.recent_uploads):
        source_icon = "🎙️" if upload.get("source") == "recording" else "📁"
        model_badge = f"[{upload.get('model', 'base')}]" if upload.get("model") else ""
        with st.expander(f"{source_icon} {upload['filename']} {model_badge}"):
            st.markdown(f"""
            - **Workflow ID**: `{upload["workflow_id"]}`
            - **Patient**: {upload["patient_name"]}
            - **Source**: {"Recording" if upload.get("source") == "recording" else "File Upload"}
            - **Model**: {upload.get("model", "base")}
            - **Medical Processing**: {"✅ Enabled" if upload["medical_enabled"] else "❌ Disabled"}
            """)

            if st.button("View in Workflows Page", key=f"view_{idx}"):
                st.switch_page("pages/2_📊_Workflows.py")

# Help section
with st.expander("ℹ️ Help & Tips"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📁 File Upload:**
        - **Audio**: MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM
        - **Video**: MP4, AVI, MOV, MKV (audio will be extracted)
        - Maximum upload size: 500 MB

        **🎙️ Voice Recording:**
        - Click the microphone button to start recording
        - Click again to stop recording
        - Audio is captured at 16kHz (optimal for speech recognition)
        """)

    with col2:
        st.markdown("""
        **⚙️ Model Selection Tips:**
        - **base**: Best for most use cases (fast + accurate)
        - **large-v3**: Best accuracy for difficult audio
        - **large-v3-turbo**: Fast + high accuracy (GPU recommended)

        **🗣️ Speaker Settings:**
        - Set min/max speakers if you know how many
        - Leave at 0 for auto-detection
        """)

    st.markdown("""
    **Initial Prompt:**
    Use the initial prompt to provide domain-specific vocabulary hints. For example:
    - Medical: "hypertension, metformin, diabetes mellitus, cardiovascular"
    - Legal: "plaintiff, defendant, deposition, litigation"
    """)

# Footer
st.caption(
    "⚠️ **HIPAA Compliance**: Patient names are transmitted securely and stored as cryptographic hashes. Ensure proper authentication is enabled for production use."
)
