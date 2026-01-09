"""
Workflows Page - Unified Search and Results

Search workflows by ID or patient hash, track status, and view detailed results.
Consolidates functionality from Workflows, Patients, and Results pages.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import httpx
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_workflow_id, format_timestamp, get_status_emoji

# Page config
st.set_page_config(page_title="Workflows - whisperX", page_icon="🔍", layout="wide")

st.title("🔍 Workflow Tracking & Results")
st.markdown("Search workflows by ID or patient hash, track status, and view detailed transcription results.")

# Initialize API client
api_client = get_api_client()

# Search mode selection
search_mode = st.radio(
    "Search by:",
    options=["Workflow ID", "Patient Hash", "Recent Uploads"],
    horizontal=True,
    help="Choose how to search for workflows",
)

# Auto-refresh configuration (only for Recent Uploads)
if search_mode == "Recent Uploads":
    auto_refresh_col, _ = st.columns([1, 3])
    with auto_refresh_col:
        auto_refresh = st.toggle("Auto-refresh", value=True, help="Refresh every 5 seconds")

    if auto_refresh:
        st_autorefresh(interval=5000, key="workflow_refresh")

st.divider()

# Search inputs based on mode
if search_mode == "Workflow ID":
    col1, col2 = st.columns([3, 1])

    with col1:
        workflow_id_input = st.text_input(
            "Workflow ID",
            placeholder="whisperx-wf-pt_abc12345-20260108_...",
            help="Enter the complete workflow ID",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

elif search_mode == "Patient Hash":
    col1, col2 = st.columns([3, 1])

    with col1:
        patient_hash_input = st.text_input(
            "Patient Hash",
            placeholder="abc12345",
            help="Enter the 8-character patient hash",
            max_chars=8,
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Filters
    if patient_hash_input and len(patient_hash_input.strip()) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "RUNNING", "COMPLETED", "FAILED"],
                help="Filter workflows by their current status",
            )

        with col2:
            limit = st.selectbox("Results per page", options=[10, 20, 50], index=0)

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(f"Page size: {limit}")

        # Initialize pagination
        if "page_offset" not in st.session_state:
            st.session_state.page_offset = 0


# ============================================================================
# Display Functions
# ============================================================================

def display_workflow_detail(workflow_id: str):
    """Display detailed workflow results with tabs."""
    try:
        # Get workflow status first
        status_response = api_client.get_workflow_status(workflow_id)
        status = status_response.get("status", "UNKNOWN")

        st.success(f"✅ Workflow **{format_workflow_id(workflow_id)}** - {status}")

        if status == "COMPLETED":
            # Fetch full results
            result = api_client.get_workflow_result(workflow_id)

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
                    # Display all medical sections from Results page
                    display_medical_results(result)
                else:
                    st.info("Medical processing was not enabled for this workflow.")

            with tab3:
                st.subheader("Raw JSON Data")

                # Download button
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{workflow_id}_results.json",
                    mime="application/json",
                )

                # Display JSON
                st.json(result)

        elif status == "RUNNING":
            st.warning("⏳ Workflow is still running.")
            st.info("Results will appear when processing is complete. Enable auto-refresh to monitor progress.")

        elif status == "FAILED":
            st.error("❌ Workflow failed.")
            if "error" in status_response:
                with st.expander("Error Details"):
                    st.code(status_response["error"])

        else:
            st.warning(f"⚠️ Unknown workflow status: {status}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.error("❌ Workflow not found")
        else:
            st.error(f"❌ HTTP Error: {e.response.status_code}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def display_medical_results(result: dict):
    """Display comprehensive medical processing results."""
    # 1. Dialogue Transformation
    if "dialogue_transformation" in result:
        st.markdown("### 🗣️ Dialogue")
        dialogue_data = result["dialogue_transformation"]

        if dialogue_data and "dialogue" in dialogue_data:
            dialogue = dialogue_data["dialogue"]
            speaker_mapping = dialogue_data.get("speaker_mapping", {})

            for entry in dialogue:
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")

                # Use different icons for different speakers
                if "provider" in speaker.lower() or "doctor" in speaker.lower():
                    icon = "👨‍⚕️"
                elif "patient" in speaker.lower():
                    icon = "👤"
                else:
                    icon = "🗣️"

                st.markdown(f"{icon} **{speaker}**: {text}")

            # Show speaker mapping
            if speaker_mapping:
                with st.expander("👥 Speaker Mapping"):
                    for speaker_id, label in speaker_mapping.items():
                        st.markdown(f"- **{speaker_id}** → {label}")
        else:
            st.info("No dialogue data available.")

        st.divider()

    # 2. PHI Detection
    if "phi_detection" in result:
        st.markdown("### 🔒 PHI Detection")
        phi_data = result["phi_detection"]

        if phi_data and isinstance(phi_data, dict):
            phi_counts = {}
            total_phi = 0

            for phi_type, instances in phi_data.items():
                if isinstance(instances, list):
                    count = len(instances)
                    if count > 0:
                        phi_counts[phi_type] = count
                        total_phi += count

            if phi_counts:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Total PHI Detected", total_phi)

                with col2:
                    st.markdown("**PHI Types:**")
                    for phi_type, count in phi_counts.items():
                        st.markdown(f"- **{phi_type.replace('_', ' ').title()}**: {count} instance(s)")

                with st.expander("🔍 View Details"):
                    for phi_type, instances in phi_data.items():
                        if isinstance(instances, list) and instances:
                            st.markdown(f"**{phi_type.replace('_', ' ').title()}:**")
                            for item in instances:
                                st.caption(f"• {item}")
            else:
                st.success("✅ No PHI detected")
        else:
            st.info("No PHI detection data available.")

        st.divider()

    # 3. Entity Extraction
    if "entity_extraction" in result:
        st.markdown("### 🔬 Medical Entities")
        entities_data = result["entity_extraction"]

        if entities_data and isinstance(entities_data, dict):
            if "entities_by_speaker" in entities_data:
                # Speaker-aware entities
                for speaker, speaker_entities in entities_data["entities_by_speaker"].items():
                    if speaker_entities:
                        with st.expander(f"🗣️ {speaker}"):
                            for entity_type, items in speaker_entities.items():
                                if items:
                                    st.markdown(f"**{entity_type.replace('_', ' ').title()}**:")
                                    for item in items:
                                        st.markdown(f"  • {item}")
            else:
                # Flat entity structure
                for entity_type, items in entities_data.items():
                    if items and isinstance(items, list):
                        with st.expander(f"{entity_type.replace('_', ' ').title()} ({len(items)})"):
                            for item in items:
                                st.markdown(f"• {item}")
        else:
            st.info("No medical entities extracted.")

        st.divider()

    # 4. SOAP Note
    if "soap_generation" in result:
        st.markdown("### 📋 SOAP Note")
        soap_data = result["soap_generation"]

        if soap_data and isinstance(soap_data, dict):
            soap_sections = {
                "subjective": ("🩺", "Subjective", "Patient's reported symptoms and history"),
                "objective": ("🔍", "Objective", "Clinical findings and observations"),
                "assessment": ("💡", "Assessment", "Diagnosis and clinical impression"),
                "plan": ("📝", "Plan", "Treatment plan and follow-up"),
            }

            for section_key, (icon, title, description) in soap_sections.items():
                if section_key in soap_data:
                    content = soap_data[section_key]
                    with st.expander(f"{icon} {title}", expanded=(section_key == "subjective")):
                        st.caption(description)
                        st.markdown(content)
        else:
            st.info("No SOAP note generated.")

        st.divider()

    # 5. Vector Storage
    if "vector_storage" in result:
        st.markdown("### 💾 Vector Storage")
        storage_data = result["vector_storage"]

        if storage_data:
            if isinstance(storage_data, dict):
                success = storage_data.get("success", False)
                if success:
                    st.success("✅ Consultation data stored in vector database")

                    col1, col2 = st.columns(2)
                    with col1:
                        if "document_id" in storage_data:
                            st.markdown(f"**Document ID**: `{storage_data['document_id']}`")

                    with col2:
                        if "indexed_count" in storage_data:
                            st.markdown(f"**Indexed Items**: {storage_data['indexed_count']}")
                else:
                    st.warning("⚠️ Vector storage encountered an issue")
                    if "error" in storage_data:
                        st.caption(storage_data["error"])
            else:
                st.success("✅ Data stored successfully")
        else:
            st.info("Vector storage not configured or disabled.")


def display_workflow_list(workflows: list):
    """Display list of workflows with expandable details."""
    for idx, workflow in enumerate(workflows):
        workflow_id = workflow.get("workflow_id", "N/A")
        status = workflow.get("status", "UNKNOWN")

        with st.expander(f"{get_status_emoji(status)} {format_workflow_id(workflow_id)}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Status**: {status}")
                st.markdown(f"**Created**: {format_timestamp(workflow.get('created_at', 'N/A'))}")

            with col2:
                if st.button("📄 View Full Results", key=f"view_{idx}"):
                    st.session_state.switch_to_workflow = workflow_id
                    st.rerun()

            # Quick preview
            if status == "COMPLETED":
                try:
                    result = api_client.get_workflow_result(workflow_id)
                    if "whisperx_transcription" in result:
                        transcription = result["whisperx_transcription"]
                        if "text" in transcription:
                            preview_text = (
                                transcription["text"][:200] + "..."
                                if len(transcription["text"]) > 200
                                else transcription["text"]
                            )
                            st.markdown("**Preview:**")
                            st.caption(preview_text)
                except Exception:
                    st.caption("Unable to load preview")


# ============================================================================
# Main Logic
# ============================================================================

# Handle workflow switch from list view
if "switch_to_workflow" in st.session_state:
    workflow_id_input = st.session_state.switch_to_workflow
    search_mode = "Workflow ID"
    search_button = True
    del st.session_state.switch_to_workflow

# Execute search based on mode
if search_mode == "Workflow ID":
    if workflow_id_input and search_button:
        display_workflow_detail(workflow_id_input.strip())

elif search_mode == "Patient Hash":
    if patient_hash_input and search_button:
        if len(patient_hash_input.strip()) >= 8:
            try:
                with st.spinner("Searching patient workflows..."):
                    status_param = None if status_filter == "All" else status_filter

                    response = api_client.get_patient_workflows(
                        patient_hash=patient_hash_input.strip(),
                        status=status_param,
                        limit=limit,
                        offset=st.session_state.page_offset,
                    )

                    workflows = response.get("workflows", [])
                    total_count = response.get("total_count", 0)
                    filtered_count = response.get("filtered_count", 0)

                    st.success(f"✅ Found {filtered_count} workflow(s) for patient: `{patient_hash_input.strip()}`")

                    if total_count != filtered_count:
                        st.info(f"ℹ️ Showing {filtered_count} out of {total_count} total workflows")

                    if workflows:
                        display_workflow_list(workflows)

                        # Pagination controls
                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col1:
                            if st.session_state.page_offset > 0:
                                if st.button("← Previous Page"):
                                    st.session_state.page_offset = max(0, st.session_state.page_offset - limit)
                                    st.rerun()

                        with col2:
                            current_page = (st.session_state.page_offset // limit) + 1
                            total_pages = (filtered_count + limit - 1) // limit
                            st.markdown(f"**Page {current_page} of {total_pages}**")

                        with col3:
                            if st.session_state.page_offset + limit < filtered_count:
                                if st.button("Next Page →"):
                                    st.session_state.page_offset += limit
                                    st.rerun()
                    else:
                        st.warning(f"No workflows found for patient: `{patient_hash_input.strip()}`")
                        if status_filter != "All":
                            st.info("Try removing the status filter to see all workflows.")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    st.error(f"❌ No patient found with hash: `{patient_hash_input.strip()}`")
                else:
                    st.error(f"❌ HTTP Error: {e.response.status_code}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.warning("⚠️ Patient hash must be exactly 8 characters")

elif search_mode == "Recent Uploads":
    if "recent_uploads" in st.session_state and st.session_state.recent_uploads:
        st.subheader("📝 Your Recent Uploads")

        for upload in st.session_state.recent_uploads[:5]:
            display_workflow_detail(upload["workflow_id"])
    else:
        st.info("""
        **No recent uploads to display.**

        - Use Workflow ID or Patient Hash search above
        - Or go to the **Upload** page to start a new transcription
        """)

        if st.button("Go to Upload Page", type="primary"):
            st.switch_page("pages/1_📤_Upload.py")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **Search by Workflow ID:**
    - View detailed results for a specific workflow
    - See transcription with speaker attribution
    - Access medical entities and SOAP notes (if applicable)
    - Download results as JSON

    **Search by Patient Hash:**
    - Find all workflows for a patient
    - Filter by status (RUNNING/COMPLETED/FAILED)
    - Paginate through results
    - Click "View Full Results" for detailed view

    **Recent Uploads:**
    - See workflows from your current session
    - Auto-refresh to monitor progress
    - Limited to 5 most recent uploads

    **Status Indicators:**
    - ✅ COMPLETED: Results available
    - ⏳ RUNNING: Still processing
    - ❌ FAILED: Encountered an error

    **Performance:**
    - Processing: ~1-2 min per 10 min of audio
    - Medical processing: +15-30 seconds
    - Auto-refresh recommended for tracking
    """)

# Footer
st.caption("💡 **Tip**: Enable auto-refresh in Recent Uploads mode to monitor running workflows in real-time.")
