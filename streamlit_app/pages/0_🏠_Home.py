"""
Home Page - Admin Dashboard

Comprehensive dashboard with system stats, recent activity, and quick actions.
Enhanced with patient management features and navigation.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_time_ago, get_status_emoji
from components.protected_name import render_protected_name
import httpx

# Page configuration
st.set_page_config(page_title="MedVoice System", page_icon="💊", layout="wide")

st.markdown(
    '<h1 style="background: linear-gradient(90deg, #2196F3, #4CAF50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MedVoice System</h1>',
    unsafe_allow_html=True,
)

st.markdown("""
This application provides
HIPAA-compliant audio transcription with optional medical processing capabilities.
""")

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

st.divider()

# Quick Stats
st.subheader("📊 Quick Stats")

# Initialize API client
api_client = get_api_client()

try:
    stats = api_client.get_database_stats()

    # System Overview Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_patients = stats.get("unique_patients", 0)
        st.metric(
            label="Total Patients",
            value=total_patients,
            help="Number of distinct patients in the system",
        )

    with col2:
        total_workflows = stats.get("total_mappings", 0)
        st.metric(
            label="Total Workflows",
            value=total_workflows,
            help="Total number of processed consultations",
        )

    with col3:
        # Calculate average
        avg_workflows = total_workflows / max(total_patients, 1)
        st.metric(label="Avg Workflows/Patient", value=f"{avg_workflows:.1f}", help="Average consultations per patient")

    with col4:
        # Placeholder for active workflows (can be enhanced later)
        st.metric(label="System Status", value="🟢 Online", help="System health status")

    st.divider()

    # Enhanced Recent Activity (20 entries instead of 5)
    st.subheader("🕐 Recent Activity")

    # Auto-refresh enabled by default (every 30 seconds)
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=30000, key="home_refresh")

    recent = stats.get("recent_entries", [])

    if recent:
        # Display up to 20 recent entries
        display_count = min(20, len(recent))

        for idx, entry in enumerate(recent[:display_count]):
            workflow_id = entry.get("workflow_id", "N/A")
            patient_hash = entry.get("patient_hash", "N/A")
            patient_name = entry.get("patient_name", "Unknown")
            created_at = entry.get("created_at", "N/A")

            # Try to get workflow status (best effort, don't fail if unavailable)
            status_response = None
            try:
                status_response = api_client.get_workflow_status(workflow_id)
                status = status_response.get("status", "UNKNOWN")
            except Exception:
                status = "UNKNOWN"

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Status emoji + clickable workflow ID
                    status_emoji = get_status_emoji(status)

                    # Truncate workflow ID for display
                    display_id = workflow_id if len(workflow_id) <= 50 else f"{workflow_id[:47]}..."
                    st.markdown(f"{status_emoji} **{display_id}**")

                    # Protected patient name
                    render_protected_name(patient_hash, patient_name, inline=True)

                    # Timestamp
                    st.caption(f"Created: {format_time_ago(created_at)}")

                with col2:
                    # Action button based on status - navigates to Workflows page with pre-loaded details
                    if status == "COMPLETED":
                        if st.button("📄 View Results", key=f"view_{idx}", use_container_width=True):
                            st.session_state.selected_workflow = workflow_id
                            st.switch_page("pages/3_🔍_Workflows.py")
                    elif status == "RUNNING":
                        if st.button("⏳ Monitor", key=f"monitor_{idx}", use_container_width=True):
                            st.session_state.selected_workflow = workflow_id
                            st.switch_page("pages/3_🔍_Workflows.py")
                    elif status == "FAILED":
                        if st.button("❌ View Error", key=f"error_{idx}", use_container_width=True):
                            st.session_state.selected_workflow = workflow_id
                            st.switch_page("pages/3_🔍_Workflows.py")
                    else:
                        if st.button("🔍 View Details", key=f"details_{idx}", use_container_width=True):
                            st.session_state.selected_workflow = workflow_id
                            st.switch_page("pages/3_🔍_Workflows.py")

        # Show count
        if len(recent) > display_count:
            st.caption(
                f"Showing {display_count} of {len(recent)} recent workflows. Use **All Workflows** page to view more."
            )
    else:
        st.info("No recent activity. Upload an audio file to get started!")

except httpx.HTTPError as e:
    st.error("Unable to fetch statistics from backend")
    st.caption(f"Error: {str(e)}")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")

# Footer
st.divider()
st.caption(
    "⚠️ **HIPAA Notice**: This system handles Protected Health Information (PHI). Ensure proper authentication and access controls are in place for production use."
)
