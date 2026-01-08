"""
Query Results Page

Search and view completed workflow results with full transcriptions and medical outputs.
"""

import streamlit as st
import httpx
import json
from datetime import datetime, date
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_workflow_id, format_timestamp, get_status_emoji

# Page config
st.set_page_config(page_title="Query Results - whisperX", page_icon="🔍", layout="wide")

st.title("🔍 Query Results")
st.markdown("Search for completed workflows and view transcription results.")

# Initialize API client
api_client = get_api_client()

# Search options
search_type = st.radio(
    "Search by:",
    options=["Workflow ID", "Patient Hash"],
    horizontal=True,
    help="Search for workflows using their unique ID or patient hash",
)

col1, col2 = st.columns([3, 1])

with col1:
    if search_type == "Workflow ID":
        search_input = st.text_input(
            "Workflow ID",
            placeholder="whisperx-wf-pt_abc12345-20260108_...",
            help="Enter the complete workflow ID",
        )
    else:
        search_input = st.text_input(
            "Patient Hash",
            placeholder="abc12345",
            help="Enter the 8-character patient hash",
            max_chars=8,
        )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Results section
if search_input and search_button:
    try:
        if search_type == "Workflow ID":
            # Search by workflow ID
            with st.spinner("Fetching workflow results..."):
                # Get workflow status first
                status_response = api_client.get_workflow_status(search_input)
                status = status_response.get("status", "UNKNOWN")

                if status == "COMPLETED":
                    # Fetch full results
                    result = api_client.get_workflow_result(search_input)

                    # Display workflow info
                    st.success(f"✅ Workflow **{format_workflow_id(search_input)}** - {status}")

                    # Tabs for different result sections
                    tab1, tab2, tab3 = st.tabs(["📄 Transcription", "🏥 Medical Results", "📊 Raw Data"])

                    with tab1:
                        st.subheader("Transcription")

                        # Check if we have transcript data
                        if "whisperx_transcription" in result:
                            transcription = result["whisperx_transcription"]
                            
                            # Display speaker-attributed dialogue
                            if "dialogue" in transcription and transcription["dialogue"]:
                                st.markdown("### Speaker Dialogue")
                                for entry in transcription["dialogue"]:
                                    speaker = entry.get("speaker", "Unknown")
                                    text = entry.get("text", "")
                                    st.markdown(f"**{speaker}**: {text}")
                            
                            # Show full text
                            if "text" in transcription:
                                with st.expander("📝 Full Text"):
                                    st.text_area(
                                        "Full Transcription",
                                        value=transcription["text"],
                                        height=300,
                                        disabled=True,
                                    )
                            
                            # Speaker mapping
                            if "speaker_mapping" in transcription:
                                with st.expander("👥 Speaker Mapping"):
                                    speaker_map = transcription["speaker_mapping"]
                                    for speaker_id, label in speaker_map.items():
                                        st.markdown(f"- **{speaker_id}**: {label}")
                        else:
                            st.info("No transcription data available for this workflow.")

                    with tab2:
                        st.subheader("Medical Processing Results")

                        # Check if medical processing was enabled
                        workflow_type = result.get("workflow_type", "")
                        
                        if "medical" in workflow_type.lower():
                            # Medical entities
                            if "medical_entities" in result:
                                st.markdown("### 🔬 Medical Entities")
                                entities = result["medical_entities"]
                                
                                if entities and isinstance(entities, dict):
                                    for entity_type, items in entities.items():
                                        if items:
                                            with st.expander(f"{entity_type.replace('_', ' ').title()} ({len(items)})"):
                                                for item in items:
                                                    st.markdown(f"- {item}")
                                else:
                                    st.info("No medical entities extracted.")
                            
                            # SOAP note
                            if "soap_note" in result:
                                st.markdown("### 📋 SOAP Note")
                                soap = result["soap_note"]
                                
                                if soap and isinstance(soap, dict):
                                    for section, content in soap.items():
                                        with st.expander(section.upper()):
                                            st.markdown(content)
                                else:
                                    st.info("No SOAP note generated.")
                        else:
                            st.info("Medical processing was not enabled for this workflow.")

                    with tab3:
                        st.subheader("Raw JSON Data")
                        
                        # Download button
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"{search_input}_results.json",
                            mime="application/json",
                        )
                        
                        # Display JSON
                        st.json(result)

                elif status == "RUNNING":
                    st.warning(f"⏳ Workflow is still running. Status: {status}")
                    st.info("Come back later when the workflow is complete.")
                
                elif status == "FAILED":
                    st.error(f"❌ Workflow failed. Status: {status}")
                    
                    # Try to get error details
                    if "error" in status_response:
                        with st.expander("Error Details"):
                            st.code(status_response["error"])
                
                else:
                    st.warning(f"⚠️ Unknown workflow status: {status}")

        else:
            # Search by patient hash - show completed workflows
            with st.spinner("Searching patient workflows..."):
                response = api_client.get_patient_workflows(
                    patient_hash=search_input,
                    status="COMPLETED",
                    limit=10,
                    offset=0,
                )

                workflows = response.get("workflows", [])
                filtered_count = response.get("filtered_count", 0)

                if workflows:
                    st.success(f"✅ Found {filtered_count} completed workflow(s) for patient: `{search_input}`")

                    for idx, workflow in enumerate(workflows):
                        workflow_id = workflow.get("workflow_id", "N/A")
                        status = workflow.get("status", "UNKNOWN")

                        with st.expander(f"{get_status_emoji(status)} {format_workflow_id(workflow_id)}"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(f"**Status**: {status}")
                                st.markdown(f"**Created**: {format_timestamp(workflow.get('created_at', 'N/A'))}")

                            with col2:
                                if st.button("View Full Results", key=f"view_{idx}"):
                                    # Update search to this workflow ID
                                    st.session_state.search_workflow = workflow_id
                                    st.rerun()

                            # Quick preview of results
                            try:
                                result = api_client.get_workflow_result(workflow_id)
                                
                                # Show transcription preview
                                if "whisperx_transcription" in result:
                                    transcription = result["whisperx_transcription"]
                                    if "text" in transcription:
                                        preview_text = transcription["text"][:200] + "..." if len(transcription["text"]) > 200 else transcription["text"]
                                        st.markdown("**Preview:**")
                                        st.caption(preview_text)
                            except Exception:
                                st.caption("Unable to load preview")
                else:
                    st.warning(f"No completed workflows found for patient: `{search_input}`")
                    st.info("Note: Only COMPLETED workflows are shown. Check the Patients page for all workflows.")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.error("❌ Workflow not found")
            st.info("Please check the workflow ID or patient hash and try again.")
        else:
            st.error(f"❌ HTTP Error: {e.response.status_code}")
            with st.expander("Error Details"):
                st.code(str(e))

    except httpx.HTTPError as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.caption("Please ensure the backend is running and accessible")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))

# Check if we should auto-load a workflow from session state
if "search_workflow" in st.session_state and not search_input:
    st.session_state.search_input = st.session_state.search_workflow
    st.session_state.search_type = "Workflow ID"
    del st.session_state.search_workflow
    st.rerun()

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **Search by Workflow ID:**
    - Paste the complete workflow ID (e.g., `whisperx-wf-pt_abc12345-20260108_...`)
    - View full transcription with speaker attribution
    - Access medical entities and SOAP notes (if applicable)
    - Download results as JSON

    **Search by Patient Hash:**
    - Enter the 8-character patient hash
    - See list of all COMPLETED workflows for that patient
    - Click "View Full Results" to see detailed transcription
    - Only shows workflows that have finished processing

    **Result Sections:**
    - **Transcription**: Speaker-attributed dialogue and full text
    - **Medical Results**: Extracted entities and SOAP notes
    - **Raw Data**: Complete JSON response for download

    **Status Indicators:**
    - ✅ COMPLETED: Results available for viewing
    - ⏳ RUNNING: Still processing, check back later
    - ❌ FAILED: Workflow encountered an error
    """)

# Footer
st.caption("💡 **Tip**: Use the Workflows page to track real-time progress of running workflows.")
