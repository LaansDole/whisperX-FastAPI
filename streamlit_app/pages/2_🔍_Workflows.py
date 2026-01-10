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

            # Check if medical processing was enabled
            workflow_type = result.get("workflow_type", "")
            has_medical = "medical" in workflow_type.lower() or "dialogue_transformation" in result

            if has_medical:
                # Full tabs with medical results
                tab1, tab2, tab3 = st.tabs(["📄 Transcription", "🏥 Medical Results", "📊 Raw Data"])

                with tab1:
                    st.subheader("Transcription")
                    # Use dialogue_transformation for properly attributed dialogue
                    if "dialogue_transformation" in result:
                        dialogue_data = result["dialogue_transformation"]

                        # Display speaker-attributed dialogue using full_transcript_markdown
                        if "full_transcript_markdown" in dialogue_data:
                            st.markdown(dialogue_data["full_transcript_markdown"])
                        elif "dialogue" in dialogue_data:
                            # Fallback to dialogue array
                            for entry in dialogue_data["dialogue"]:
                                speaker_role = entry.get("speaker_role", "Unknown").title()
                                text = entry.get("text", "")
                                st.markdown(f"**{speaker_role}:** {text}")

                        # Show full text in expander
                        if "full_transcript" in dialogue_data:
                            with st.expander("📝 Full Text"):
                                st.text_area(
                                    "Full Transcription",
                                    value=dialogue_data["full_transcript"],
                                    height=300,
                                    disabled=True,
                                )

                        # Speaker mapping
                        if "speaker_mapping" in dialogue_data:
                            with st.expander("👥 Speaker Mapping"):
                                speaker_map = dialogue_data["speaker_mapping"]
                                for speaker_id, mapping_data in speaker_map.items():
                                    if isinstance(mapping_data, dict):
                                        role = mapping_data.get("role", "Unknown").title()
                                        confidence = mapping_data.get("confidence", 0) * 100
                                        st.markdown(f"- **{speaker_id}**: {role} ({confidence:.0f}% confidence)")
                                    else:
                                        st.markdown(f"- **{speaker_id}**: {mapping_data}")
                    else:
                        st.info("No transcription data available.")

                with tab2:
                    st.subheader("Medical Processing Results")
                    display_medical_results(result)

                with tab3:
                    st.subheader("Raw Data")
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{workflow_id}_results.json",
                        mime="application/json",
                    )
                    st.json(result)

            else:
                # Simple transcription-only view
                tab1, tab2 = st.tabs(["📄 Transcription", "📊 Raw Data"])

                with tab1:
                    st.subheader("Transcription")
                    # Use whisperx_final for non-medical workflows
                    if "whisperx_final" in result:
                        final_result = result["whisperx_final"]
                        # Extract full text from segments
                        if "segments" in final_result:
                            full_text = " ".join(seg.get("text", "").strip() for seg in final_result["segments"])
                            st.text_area(
                                "Full Transcription",
                                value=full_text,
                                height=400,
                                disabled=True,
                            )
                        else:
                            st.info("No transcription segments available.")
                    else:
                        st.info("No transcription data available.")

                with tab2:
                    st.subheader("Raw Data")
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{workflow_id}_results.json",
                        mime="application/json",
                    )
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
    # 1. Dialogue - use full_transcript_markdown from dialogue_transformation
    st.markdown("### Dialogue")

    # Get dialogue_transformation data
    if "dialogue_transformation" in result:
        dialogue_data = result["dialogue_transformation"]

        # Use full_transcript_markdown for proper speaker attribution
        if "full_transcript_markdown" in dialogue_data:
            st.markdown(dialogue_data["full_transcript_markdown"])
            st.divider()
        elif "dialogue" in dialogue_data:
            # Fallback to dialogue array
            dialogue = dialogue_data["dialogue"]
            speaker_mapping = dialogue_data.get("speaker_mapping", {})

            for entry in dialogue:
                speaker_role = entry.get("speaker_role", "Unknown").title()
                text = entry.get("text", "")
                st.markdown(f"**{speaker_role}:** {text}")

            # Show speaker mapping
            if speaker_mapping:
                with st.expander("👥 Speaker Mapping"):
                    for speaker_id, mapping_data in speaker_mapping.items():
                        if isinstance(mapping_data, dict):
                            role = mapping_data.get("role", "Unknown").title()
                            confidence = mapping_data.get("confidence", 0) * 100
                            st.markdown(f"- **{speaker_id}**: {role} ({confidence:.0f}% confidence)")
                        else:
                            st.markdown(f"- **{speaker_id}**: {mapping_data}")
            st.divider()
    else:
        st.info("No dialogue data available.")
        st.divider()

    # 2. PHI Detection
    if "phi_detection" in result:
        st.markdown("### PHI Detection")
        phi_data = result["phi_detection"]

        if phi_data and isinstance(phi_data, dict):
            # Check for phi_detected flag first
            phi_detected = phi_data.get("phi_detected", False)
            phi_entities = phi_data.get("entities", [])

            if phi_detected and phi_entities:
                st.warning(f"⚠️ {len(phi_entities)} PHI entities detected")

                # Group by type
                phi_by_type = {}
                for entity in phi_entities:
                    phi_type = entity.get("type", "unknown")
                    if phi_type not in phi_by_type:
                        phi_by_type[phi_type] = []
                    phi_by_type[phi_type].append(entity)

                # Display grouped PHI
                for phi_type, items in phi_by_type.items():
                    with st.expander(f"{phi_type.replace('_', ' ').title()} ({len(items)})"):
                        for entity in items:
                            text = entity.get("text", "Unknown")
                            confidence = entity.get("confidence", 0) * 100
                            st.markdown(f"**{text}**")
                            st.caption(f"Confidence: {confidence:.0f}%")
            else:
                st.success("✅ No PHI detected")
        else:
            st.info("No PHI detection data available.")

        st.divider()

    # 3. Entity Extraction
    if "entity_extraction" in result:
        st.markdown("### Medical Entities")
        entities_data = result["entity_extraction"]

        if entities_data and isinstance(entities_data, dict):
            # Check for entities array (new format)
            if "entities" in entities_data and entities_data["entities"]:
                entities = entities_data["entities"]
                entity_count = entities_data.get("entity_count", len(entities))

                st.markdown(f"**Found {entity_count} medical entities**")

                # Group by type
                entities_by_type = {}
                for entity in entities:
                    entity_type = entity.get("type", "unknown")
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)

                # Display grouped entities
                for entity_type, items in entities_by_type.items():
                    with st.expander(f"{entity_type.replace('_', ' ').title()} ({len(items)})"):
                        for entity in items:
                            text = entity.get("text", "Unknown")
                            normalized = entity.get("normalized", text)
                            speaker_role = entity.get("speaker_role", "Unknown").title()
                            confidence = entity.get("confidence", 0) * 100

                            st.markdown(f"**{normalized}**")
                            st.caption(f"Original: {text} | Speaker: {speaker_role} | Confidence: {confidence:.0f}%")
                            if "details" in entity:
                                st.caption(f"Details: {entity['details']}")
            elif "entities_by_speaker" in entities_data:
                # Speaker-aware entities (old format)
                for speaker, speaker_entities in entities_data["entities_by_speaker"].items():
                    if speaker_entities:
                        with st.expander(f"{speaker}"):
                            for entity_type, items in speaker_entities.items():
                                if items:
                                    st.markdown(f"**{entity_type.replace('_', ' ').title()}**:")
                                    for item in items:
                                        st.markdown(f"  • {item}")
            else:
                st.info("No medical entities extracted.")
        else:
            st.info("No medical entities data available.")

        st.divider()

    # 4. SOAP Note
    if "soap_generation" in result:
        st.markdown("### SOAP Note")
        soap_data = result["soap_generation"]

        if soap_data and isinstance(soap_data, dict):
            # Check for soap_note sub-object first
            soap_note = soap_data.get("soap_note", soap_data)

            soap_sections = {
                "subjective": ("S", "Subjective", "Patient's reported symptoms and history"),
                "objective": ("O", "Objective", "Clinical findings and observations"),
                "assessment": ("A", "Assessment", "Diagnosis and clinical impression"),
                "plan": ("P", "Plan", "Treatment plan and follow-up"),
            }

            has_content = any(soap_note.get(key, "").strip() for key in soap_sections.keys())

            if has_content:
                for section_key, (icon, title, description) in soap_sections.items():
                    content = soap_note.get(section_key, "")
                    if content and content.strip():
                        with st.expander(f"{icon} {title}", expanded=(section_key == "subjective")):
                            st.caption(description)
                            st.markdown(content)
            else:
                st.info("SOAP note sections are empty (expected for non-medical conversations).")
        else:
            st.info("No SOAP note data available.")

        st.divider()

    # 5. Vector Storage
    if "vector_storage" in result:
        st.markdown("### Vector Storage")
        storage_data = result["vector_storage"]

        if storage_data and isinstance(storage_data, dict):
            # Check if vector_id exists (new format)
            if "vector_id" in storage_data:
                st.success("✅ Consultation data stored in vector database")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if "consultation_id" in storage_data:
                        st.markdown(f"**Consultation ID**: `{storage_data['consultation_id']}`")

                with col2:
                    if "vector_id" in storage_data:
                        st.markdown(f"**Vector ID**: {storage_data['vector_id']}")

                with col3:
                    if "stored_at" in storage_data:
                        st.markdown(f"**Stored At**: {storage_data['stored_at'][:19]}")

                # Show metadata if available
                if "metadata" in storage_data:
                    metadata = storage_data["metadata"]
                    with st.expander("📋 Storage Metadata"):
                        if "entity_count" in metadata:
                            st.markdown(f"- **Entities**: {metadata['entity_count']}")
                        if "has_soap_note" in metadata:
                            st.markdown(f"- **SOAP Note**: {'Yes' if metadata['has_soap_note'] else 'No'}")
                        if "has_phi" in metadata:
                            st.markdown(f"- **PHI Detected**: {'Yes' if metadata['has_phi'] else 'No'}")
            else:
                # Old format with success flag
                success = storage_data.get("success", False)
                if success:
                    st.success("✅ Consultation data stored in vector database")
                    if "document_id" in storage_data:
                        st.markdown(f"**Document ID**: `{storage_data['document_id']}`")
                else:
                    st.warning("⚠️ Vector storage encountered an issue")
                    if "error" in storage_data:
                        st.caption(storage_data["error"])
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

            # Quick preview from whisperx_final
            if status == "COMPLETED":
                try:
                    result = api_client.get_workflow_result(workflow_id)
                    if "whisperx_final" in result:
                        final_result = result["whisperx_final"]
                        if "segments" in final_result and final_result["segments"]:
                            # Build text from segments
                            full_text = " ".join(seg.get("text", "").strip() for seg in final_result["segments"])
                            preview_text = full_text[:200] + "..." if len(full_text) > 200 else full_text
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
