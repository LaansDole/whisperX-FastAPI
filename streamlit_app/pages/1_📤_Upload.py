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

# Tab-based input selection
upload_tab, record_tab = st.tabs(["📁 Upload File", "🎙️ Record Audio"])

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

                        # Upload to backend
                        response = api_client.upload_audio(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            patient_name=patient_name_upload.strip(),
                            enable_medical=enable_medical_upload,
                            provider_id=provider_id_upload.strip() if provider_id_upload else None,
                            encounter_date=encounter_date_str,
                        )

                        # Success!
                        st.success("✅ Upload successful!")

                        workflow_id = response.get("identifier", "N/A")
                        message = response.get("message", "Workflow started")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Status**: {message}
                        **Patient**: {patient_name_upload} (stored as hash)
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

                        # Upload to backend
                        response = api_client.upload_audio(
                            file_bytes=audio_bytes,
                            filename=filename,
                            patient_name=patient_name_record.strip(),
                            enable_medical=enable_medical_record,
                            provider_id=provider_id_record.strip() if provider_id_record else None,
                            encounter_date=encounter_date_str,
                        )

                        # Success!
                        st.success("✅ Recording submitted successfully!")

                        workflow_id = response.get("identifier", "N/A")
                        message = response.get("message", "Workflow started")

                        st.markdown(f"""
                        **Workflow ID**: `{workflow_id}`
                        **Status**: {message}
                        **Patient**: {patient_name_record} (stored as hash)
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
        with st.expander(f"{source_icon} {upload['filename']} - {upload['workflow_id'][:40]}..."):
            st.markdown(f"""
            - **Workflow ID**: `{upload["workflow_id"]}`
            - **Patient**: {upload["patient_name"]}
            - **Source**: {"Recording" if upload.get("source") == "recording" else "File Upload"}
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

        **Patient Name Security:**
        - Names are sent securely in the request body
        - Names are hashed for filename and workflow ID
        - Only the hash is used in logs and tracking
        """)

    with col2:
        st.markdown("""
        **🎙️ Voice Recording:**
        - Click the microphone button to start recording
        - Click again to stop recording
        - Audio is captured at 16kHz (optimal for speech recognition)
        - Recorded as WAV format

        **Browser Requirements:**
        - Microphone permission must be granted
        - Works best in Chrome, Firefox, or Edge
        - HTTPS required for microphone access in production
        """)

    st.markdown("""
    **Medical Processing:**
    - Detects and flags Protected Health Information (PHI)
    - Extracts medical entities (diagnoses, medications, symptoms)
    - Generates structured SOAP notes
    - Requires LM Studio to be running with a medical model

    **Processing Time:**
    - Transcription: ~1-2 minutes per 10 minutes of audio
    - Medical processing: Additional 15-30 seconds
    """)

# Footer
st.caption(
    "⚠️ **HIPAA Compliance**: Patient names are transmitted securely and stored as cryptographic hashes. Ensure proper authentication is enabled for production use."
)
