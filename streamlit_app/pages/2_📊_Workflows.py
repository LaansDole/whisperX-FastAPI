"""
Workflows Page

Track workflow status and view transcription results.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import httpx
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_workflow_id, extract_patient_hash, get_status_emoji

# Page config
st.set_page_config(page_title="Workflows - whisperX", page_icon="📊", layout="wide")

st.title("📊 Workflow Tracking")
st.markdown("Monitor workflow status and view transcription results in real-time.")


# Auto-refresh configuration
refresh_col, search_col = st.columns([1, 3])

with refresh_col:
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Refresh every 5 seconds")

if auto_refresh:
    # Auto-refresh every 5 seconds (5000 milliseconds)
    st_autorefresh(interval=5000, key="workflow_refresh")

with search_col:
    search_workflow_id = st.text_input(
        "Search by Workflow ID", placeholder="whisperx-wf-pt_abc12345-...", help="Enter full or partial workflow ID"
    )

# Initialize API client
api_client = get_api_client()


#  Workflow display function
def display_workflow(workflow_id: str):
    """Display workflow status and results"""
    try:
        # Get workflow status
        status_response = api_client.get_workflow_status(workflow_id)

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**Workflow ID**: `{format_workflow_id(workflow_id)}`")
            patient_hash = extract_patient_hash(workflow_id)
            if patient_hash:
                st.caption(f"Patient Hash: `{patient_hash}`")

        with col2:
            status = status_response.get("status", "UNKNOWN")
            emoji = get_status_emoji(status)
            st.markdown(f"{emoji} **{status}**")

        with col3:
            # Get workflow creation time from response if available
            st.caption(status_response.get("run_time", "N/A"))

        # If completed, show results
        if status == "COMPLETED":
            with st.expander("📄 View Results", expanded=False):
                try:
                    result = api_client.get_workflow_result(workflow_id)

                    # Tabs for different result views
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["📝 Transcript", "🏥 Medical Entities", "📋 SOAP Note", "📊 Raw JSON"]
                    )

                    with tab1:
                        # Display transcript
                        dialogue = result.get("dialogue", [])

                        if dialogue:
                            st.markdown("#### Speaker-Attributed Dialogue")

                            for segment in dialogue:
                                speaker = segment.get("speaker", "Unknown")
                                role = segment.get("role", "")
                                text = segment.get("text", "")

                                # Color-code by role
                                if role.lower() == "doctor":
                                    st.markdown(f"**🩺 {speaker}** ({role}): {text}")
                                elif role.lower() == "patient":
                                    st.markdown(f"**👤 {speaker}** ({role}): {text}")
                                else:
                                    st.markdown(f"**{speaker}**: {text}")
                        else:
                            st.info("No dialogue data available")

                        # Statistics
                        stats = result.get("statistics", {})
                        if stats:
                            st.markdown("#### Statistics")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Segments", stats.get("total_segments", 0))
                            with col2:
                                st.metric("Total Duration", f"{stats.get('total_duration_seconds', 0):.1f}s")
                            with col3:
                                st.metric("Speakers", stats.get("num_speakers", 0))

                    with tab2:
                        # Medical entities
                        entities = result.get("medical_entities", [])

                        if entities:
                            st.markdown("#### Extracted Medical Entities")

                            # Group entities by type
                            entity_types = {}
                            for entity in entities:
                                entity_type = entity.get("type", "Unknown")
                                if entity_type not in entity_types:
                                    entity_types[entity_type] = []
                                entity_types[entity_type].append(entity)

                            for entity_type, type_entities in entity_types.items():
                                with st.expander(f"{entity_type} ({len(type_entities)})", expanded=True):
                                    for entity in type_entities:
                                        st.markdown(f"- **{entity.get('entity', 'N/A')}**")
                                        if "speaker" in entity:
                                            st.caption(f"  Mentioned by: {entity['speaker']}")
                        else:
                            st.info("No medical entities extracted. Enable medical processing during upload.")

                    with tab3:
                        # SOAP note
                        soap_note = result.get("soap_note", {})

                        if soap_note:
                            st.markdown("#### SOAP Note")

                            for section, content in soap_note.items():
                                st.markdown(f"**{section.upper()}**")
                                st.markdown(content)

                        else:
                            st.info("No SOAP note generated. Enable medical processing during upload.")

                    with tab4:
                        # Raw JSON
                        st.json(result)

                        # Download button
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            label="Download Results (JSON)",
                            data=json_str,
                            file_name=f"{workflow_id}_results.json",
                            mime="application/json",
                        )

                except httpx.HTTPError as e:
                    st.error(f"Failed to fetch results: {str(e)}")

        elif status == "RUNNING":
            st.info("⏳ Workflow is still processing. Results will appear when completed.")

        elif status == "FAILED":
            st.error("❌ Workflow failed. Check backend logs for details.")

            # Try to get error details
            error_msg = status_response.get("error", "No error details available")
            with st.expander("Error Details"):
                st.code(error_msg)

    except httpx.HTTPError as e:
        st.error(f"❌ Failed to fetch workflow status: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


# Main content
if search_workflow_id and len(search_workflow_id.strip()) > 0:
    # Search mode - show specific workflow
    st.subheader(f"Workflow: {search_workflow_id}")
    display_workflow(search_workflow_id.strip())

else:
    # Show recent uploads from session state
    if "recent_uploads" in st.session_state and st.session_state.recent_uploads:
        st.subheader("📝 Your Recent Uploads")

        for upload in st.session_state.recent_uploads[:5]:
            display_workflow(upload["workflow_id"])
    else:
        st.info("""
        **No workflows to display.**

        - Use the search box above to look up a specific workflow ID
        - Or go to the **Upload** page to start a new transcription
        """)

        if st.button("Go to Upload Page", type="primary"):
            st.switch_page("pages/1_📤_Upload.py")

# Help section

with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **Workflow Status:**
    - ⏳ **RUNNING**: Transcription in progress
    - ✅ **COMPLETED**: Processing finished, results available
    - ❌ **FAILED**: An error occurred during processing
    - 🕐 **PENDING**: Workflow queued, not started yet

    **Auto-Refresh:**
    - Enable auto-refresh to automatically check for status updates
    - Refreshes every 5 seconds
    - Useful for monitoring running workflows

    **Viewing Results:**
    - Results only available for COMPLETED workflows
    - Click "View Results" to expand and see transcription
    - Use tabs to navigate between Transcript, Entities, SOAP, and Raw JSON
    - Download results as JSON for external use

    **Performance:**
    - Processing time depends on audio length and medical processing options
    - Typical: 1-2 minutes per 10 minutes of audio
    - Medical processing adds 15-30 seconds

    **Troubleshooting:**
    - If workflow stays in RUNNING for too long, check backend logs
    - FAILED workflows may indicate model loading issues or invalid audio
    - Contact system administrator if issues persist
    """)

# Footer

st.caption("💡 **Tip**: Keep this page open with auto-refresh enabled to monitor running workflows in real-time.")
