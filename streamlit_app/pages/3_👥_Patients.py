"""
Patients Page

Search patient workflows by hash and view consultation history.
"""

import streamlit as st
import httpx
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client, format_workflow_id, format_timestamp, get_status_emoji

# Page config
st.set_page_config(page_title="Patients - whisperX", page_icon="👥", layout="wide")

st.title("👥 Patient Workflow Search")
st.markdown("Search for patient consultations using their unique hash identifier.")


# Initialize API client
api_client = get_api_client()

# Search form
col1, col2 = st.columns([3, 1])

with col1:
    patient_hash = st.text_input(
        "Patient Hash",
        placeholder="abc12345",
        help="Enter the 8-character patient hash (found in workflow IDs or filenames)",
        max_chars=8,
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Filter options
if patient_hash and len(patient_hash.strip()) > 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "RUNNING", "COMPLETED", "FAILED"],
            help="Filter workflows by their current status",
        )

    with col2:
        limit = st.number_input(
            "Results per page",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Number of workflows to display per page",
        )

    with col3:
        # Pagination offset (will be managed by session state)
        if "page_offset" not in st.session_state:
            st.session_state.page_offset = 0

# Main search logic
if (
    patient_hash
    and len(patient_hash.strip()) >= 8
    and (search_button or st.session_state.get("last_search") == patient_hash)
):
    st.session_state.last_search = patient_hash.strip()

    try:
        # Get workflows for patient
        status_param = None if status_filter == "All" else status_filter

        response = api_client.get_patient_workflows(
            patient_hash=patient_hash.strip(), status=status_param, limit=limit, offset=st.session_state.page_offset
        )

        total_count = response.get("total_count", 0)
        filtered_count = response.get("filtered_count", 0)
        workflows = response.get("workflows", [])

        # Display results header
        st.success(f"✅ Found {filtered_count} workflow(s) for patient hash: `{patient_hash}`")

        if total_count != filtered_count:
            st.info(
                f"ℹ️ Showing {filtered_count} out of {total_count} total workflows (filtered by status: {status_filter})"
            )

        if workflows:
            # Convert to DataFrame for better display
            df_data = []
            for workflow in workflows:
                df_data.append(
                    {
                        "Workflow ID": format_workflow_id(workflow.get("workflow_id", "N/A")),
                        "Status": f"{get_status_emoji(workflow.get('status', 'UNKNOWN'))} {workflow.get('status', 'UNKNOWN')}",
                        "Department": workflow.get("department", "N/A") or "N/A",
                        "Created": format_timestamp(workflow.get("created_at", "N/A")),
                    }
                )

            df = pd.DataFrame(df_data)

            # Display as data frame (interactive table)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Detailed view for each workflow

            st.subheader("📋 Workflow Details")

            for idx, workflow in enumerate(workflows):
                workflow_id = workflow.get("workflow_id", "N/A")
                status = workflow.get("status", "UNKNOWN")

                with st.expander(f"{get_status_emoji(status)} {format_workflow_id(workflow_id)}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Status**: {status}")
                        st.markdown(f"**Department**: {workflow.get('department', 'N/A') or 'N/A'}")

                    with col2:
                        st.markdown(f"**Created**: {format_timestamp(workflow.get('created_at', 'N/A'))}")

                    with col3:
                        st.markdown(f"**Patient Hash**: `{patient_hash}`")

                    # Action buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("View in Workflows Page", key=f"view_{idx}"):
                            # Store workflow ID in session and navigate
                            st.session_state.selected_workflow = workflow_id
                            st.switch_page("pages/2_📊_Workflows.py")

                    with col2:
                        # Copy workflow ID
                        st.code(workflow_id, language=None)

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
                st.markdown(f"<center>Page {current_page} of {total_pages}</center>", unsafe_allow_html=True)

            with col3:
                if st.session_state.page_offset + limit < filtered_count:
                    if st.button("Next Page →"):
                        st.session_state.page_offset += limit
                        st.rerun()

        else:
            st.warning(f"No workflows found for patient hash: `{patient_hash}`")

            if status_filter != "All":
                st.info("Try removing the status filter to see all workflows for this patient.")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.error(f"❌ No patient found with hash: `{patient_hash}`")
            st.info("""
            **Possible reasons:**
            - The patient hash is incorrect
            - No workflows have been created for this patient yet
            - The hash was changed due to patient name modification
            """)
        else:
            st.error(f"❌ HTTP Error: {e.response.status_code} - {str(e)}")

    except httpx.HTTPError as e:
        st.error(f"❌ Failed to fetch patient workflows: {str(e)}")
        st.caption("Please ensure the backend is running and accessible")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

elif patient_hash and len(patient_hash.strip()) > 0 and len(patient_hash.strip()) < 8:
    st.warning("⚠️ Patient hash must be exactly 8 characters")

# Help section

with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    **What is a Patient Hash?**
    - An 8-character identifier derived from the patient's name
    - Used to protect patient privacy (HIPAA-compliant)
    - Appears in workflow IDs and filenames
    - Example: `abc12345` from workflow ID `whisperx-wf-pt_abc12345-20260102_...`

    **Finding the Patient Hash:**
    1. From a workflow ID: Look for the 8 characters after `pt_`
    2. From a filename: First 8 characters after `audio_`
    3. From the Upload confirmation: Displayed in the success message

    **Status Filtering:**
    - **All**: Show all workflows regardless of status
    - **RUNNING**: Only show workflows currently being processed
    - **COMPLETED**: Only show finished workflows with results
    - **FAILED**: Only show workflows that encountered errors

    **Pagination:**
    - Results are paginated to improve performance
    - Default: 20 workflows per page
    - Use "Previous" and "Next" buttons to navigate
    - Adjust "Results per page" to show more/fewer at once

    **Viewing Details:**
    - Click on a workflow expander to see full details
    - Use "View in Workflows Page" to see complete results
    - Copy workflow ID to share or reference elsewhere

    **Performance Note:**
    - Searching with status filter may be slow for patients with many workflows
    - This is because each workflow must be checked individually with Temporal
    - For better performance, search without status filter first
    """)

# Example section
if not patient_hash or len(patient_hash.strip()) == 0:
    st.subheader("📚 Example")

    st.markdown("""
    If you uploaded a file and received this workflow ID:

    ```
    whisperx-wf-pt_abc12345-20260102_095830123456-a1b2
    ```

    The patient hash is: **`abc12345`** (the 8 characters after `pt_`)

    Enter this hash in the search box above to find all workflows for this patient.
    """)

# Footer

st.caption(
    "🔒 **Privacy**: Patient hashes are one-way cryptographic hashes. The original patient name cannot be recovered from the hash alone."
)
