"""
Upload Audio File Page

Upload audio/video files for transcription with optional medical processing.
"""

import streamlit as st
from datetime import date
import httpx
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, get_status_emoji

# Page config
st.set_page_config(
    page_title="Upload Audio - whisperX",
    page_icon="📤",
    layout="wide"
)

st.title("📤 Upload Audio File")
st.markdown("Upload an audio or video file for transcription and optional medical processing.")



# Initialize API client
api_client = get_api_client()

# Upload form
with st.form("upload_form", clear_on_submit=True):
    st.subheader("File Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'webm', 'mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM, MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
    
    
    
    # Patient information
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input(
            "Patient Full Name *",
            help="Required for HIPAA-compliant identification (submitted securely in request body)",
            placeholder="John Michael Smith"
        )
    
    with col2:
        encounter_date_input = st.date_input(
            "Encounter Date",
            value=date.today(),
            help="Date of consultation (defaults to today)"
        )
    
    
    
    # Medical processing options
    st.subheader("Processing Options")
    
    enable_medical = st.checkbox(
        "Enable Medical Processing",
        help="Enable PHI detection, medical entity extraction, and SOAP note generation",
        value=False
    )
    
    provider_id = None
    if enable_medical:
        st.info("ℹ️ Medical processing requires a Provider ID")
        provider_id = st.text_input(
            "Provider ID *",
            help="Healthcare provider identifier (required for medical processing)",
            placeholder="DR001"
        )
    
    
    
    # Submit button
    submitted = st.form_submit_button(
        "Upload & Process",
        type="primary",
        use_container_width=True
    )
    
    # Form validation and submission
    if submitted:
        # Validate required fields
        if not uploaded_file:
            st.error("❌ Please select a file to upload")
        elif not patient_name or len(patient_name.strip()) == 0:
            st.error("❌ Patient name is required")
        elif enable_medical and (not provider_id or len(provider_id.strip()) == 0):
            st.error("❌ Provider ID is required when medical processing is enabled")
        else:
            # All validations passed, proceed with upload
            with st.spinner("Uploading file and starting workflow..."):
                try:
                    # Read file bytes
                    file_bytes = uploaded_file.read()
                    
                    # Format encounter date
                    encounter_date_str = encounter_date_input.isoformat()
                    
                    # Upload to backend
                    response = api_client.upload_audio(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        patient_name=patient_name.strip(),
                        enable_medical=enable_medical,
                        provider_id=provider_id.strip() if provider_id else None,
                        encounter_date=encounter_date_str
                    )
                    
                    # Success!
                    st.success("✅ Upload successful!")
                    
                    workflow_id = response.get("identifier", "N/A")
                    message = response.get("message", "Workflow started")
                    
                    st.markdown(f"""
                    **Workflow ID**: `{workflow_id}`  
                    **Status**: {message}  
                    **Patient**: {patient_name} (stored as hash)  
                    **Medical Processing**: {"Enabled" if enable_medical else "Disabled"}
                    """)
                    
                    # Show workflow tracking link
                    st.info("ℹ️ Your workflow is being processed. Go to the **Workflows** page to track progress.")
                    
                    # Store workflow ID in session state for easy access
                    if 'recent_uploads' not in st.session_state:
                        st.session_state.recent_uploads = []
                    
                    st.session_state.recent_uploads.insert(0, {
                        'workflow_id': workflow_id,
                        'patient_name': patient_name,
                        'filename': uploaded_file.name,
                        'medical_enabled': enable_medical
                    })
                    
                    # Keep only last 10
                    st.session_state.recent_uploads = st.session_state.recent_uploads[:10]
                    
                except httpx.HTTPError as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    st.caption("Please ensure the backend is running and accessible")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

# Recent uploads section
if 'recent_uploads' in st.session_state and st.session_state.recent_uploads:
    st.subheader("📝 Recent Uploads (This Session)")
    
    for idx, upload in enumerate(st.session_state.recent_uploads):
        with st.expander(f"{upload['filename']} - {upload['workflow_id'][:40]}..."):
            st.markdown(f"""
            - **Workflow ID**: `{upload['workflow_id']}`
            - **Patient**: {upload['patient_name']}
            - **Medical Processing**: {"✅ Enabled" if upload['medical_enabled'] else "❌ Disabled"}
            """)
            
            if st.button("View in Workflows Page", key=f"view_{idx}"):
                st.switch_page("pages/2_📊_Workflows.py")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **Supported File Formats:**
    - **Audio**: MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM
    - **Video**: MP4, AVI, MOV, MKV (audio will be extracted)
    
    **File Size Limits:**
    - Maximum upload size: 500 MB
    - For larger files, consider splitting or compressing
    
    **Patient Name Security:**
    - Patient names are sent securely in the request body (not in URL)
    - Names are hashed for filename and workflow ID generation
    - Only the hash is used in logs and tracking systems
    
    **Medical Processing:**
    - Detects and flags Protected Health Information (PHI)
    - Extracts medical entities (diagnoses, medications, symptoms)
    - Generates structured SOAP notes
    - Requires LM Studio to be running with a medical model
    
    **Processing Time:**
    - Transcription: ~1-2 minutes per 10 minutes of audio
    - Medical processing: Additional 15-30 seconds
    - Actual time depends on file length and system resources
    """)

# Footer

st.caption("⚠️ **HIPAA Compliance**: Patient names are transmitted securely and stored as cryptographic hashes. Ensure proper authentication is enabled for production use.")
