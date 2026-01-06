"""
Home Page

Dashboard with system stats and getting started guide.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_time_ago
import httpx

# Page configuration
st.set_page_config(page_title="whisperX Medical Transcription", page_icon="💊", layout="wide")

st.markdown('<h1 style="background: linear-gradient(90deg, #2196F3, #4CAF50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">whisperX Medical Transcription</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome to the **whisperX Medical Transcription System**. This application provides
HIPAA-compliant audio transcription with optional medical processing capabilities.

**Features:**
- 🎙️ Audio/video transcription with WhisperX
- 🔴 Live voice recording from microphone
- 👥 Speaker diarization and identification
- 🏥 Medical entity extraction
- 📋 Automated SOAP note generation
- 🔍 Vector-based consultation search
""")

st.divider()

# Quick Stats
st.subheader("📊 Quick Stats")

# Initialize API client
api_client = get_api_client()

try:
    stats = api_client.get_database_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Workflows",
            value=stats.get("total_mappings", 0),
            help="Total number of processed consultations",
        )

    with col2:
        st.metric(
            label="Unique Patients",
            value=stats.get("unique_patients", 0),
            help="Number of distinct patients in the system",
        )

    with col3:
        avg_workflows = stats.get("total_mappings", 0) / max(stats.get("unique_patients", 1), 1)
        st.metric(label="Avg Workflows/Patient", value=f"{avg_workflows:.1f}", help="Average consultations per patient")

    st.divider()

    # Recent activity
    st.subheader("🕐 Recent Activity")

    recent = stats.get("recent_entries", [])

    if recent:
        for entry in recent[:5]:
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                # Truncate workflow ID for display
                workflow_id = entry.get("workflow_id", "N/A")
                if len(workflow_id) > 40:
                    workflow_id = f"{workflow_id[:37]}..."
                st.markdown(f"**{workflow_id}**")

            with col2:
                patient_hash = entry.get("patient_hash", "N/A")
                st.caption(f"Patient: `{patient_hash}`")

            with col3:
                created_at = entry.get("created_at", "N/A")
                st.caption(format_time_ago(created_at))
    else:
        st.info("No recent activity. Upload an audio file to get started!")

except httpx.HTTPError as e:
    st.error("Unable to fetch statistics from backend")
    st.caption(f"Error: {str(e)}")

st.divider()

# Getting Started
st.subheader("🚀 Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **1. Upload Audio**
    - Navigate to the **Upload** page
    - Select an audio or video file
    - Enter patient information
    - Enable medical processing if needed

    **2. Track Progress**
    - View workflow status in **Workflows** page
    - Auto-refresh shows real-time updates
    - Expand completed workflows to see results
    """)

with col2:
    st.markdown("""
    **3. View Results**
    - Speaker-attributed transcripts
    - Extracted medical entities
    - Generated SOAP notes
    - Consultation statistics

    **4. Search History**
    - Use **Patients** page to search by hash
    - View all consultations for a patient
    - Filter and paginate results
    """)

# Footer
st.divider()
st.caption(
    "⚠️ **HIPAA Notice**: This system handles Protected Health Information (PHI). Ensure proper authentication and access controls are in place for production use."
)
