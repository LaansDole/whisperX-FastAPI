"""
Patient Browser Page

Browse and search all patients with filtering, sorting, and pagination.
Provides quick access to patient workflows and consultations.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_api_client
from components.patient_card import render_patient_card
import httpx

# Page config
st.set_page_config(page_title="Patients - whisperX", page_icon="👥", layout="wide")

st.title("👥 Patient Browser")
st.markdown("Browse and search all patients in the system. View workflow statistics and access patient records.")

# Initialize API client
api_client = get_api_client()

# Initialize pagination state
if "patient_page" not in st.session_state:
    st.session_state.patient_page = 0

# Search and Filter Controls
st.subheader("🔍 Search & Filter")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    search_query = st.text_input(
        "Search by patient hash",
        placeholder="e.g., 154c26a1",
        help="Enter 8-character patient hash to search",
    )

with col2:
    filter_option = st.selectbox(
        "Filter",
        ["All Patients", "Active (last 7 days)", "Inactive (>30 days)", "Has Failed Workflows"],
        help="Filter patients by activity",
    )

with col3:
    sort_option = st.selectbox(
        "Sort by",
        ["Last Activity (newest first)", "Last Activity (oldest first)", "Most Workflows", "Patient Hash"],
        help="Sort patients",
    )

st.divider()

# Fetch patients
try:
    with st.spinner("Loading patients..."):
        # Get all patients from API (summary list)
        response = api_client.get_all_patients()
        patient_summaries = response.get("patients", [])
        total_patients = len(patient_summaries)

        # Fetch detailed info for each patient (includes workflows)
        all_patients = []
        for summary in patient_summaries:
            try:
                # Get full patient info with workflows
                patient_hash = summary.get("patient_hash")
                detailed_info = api_client.get_patient_info_by_hash(patient_hash)
                all_patients.append(detailed_info)
            except Exception as e:
                # If detail fetch fails, use summary with minimal info
                st.warning(f"Could not fetch details for patient {summary.get('patient_hash')}: {str(e)}")
                all_patients.append(
                    {
                        "patient_hash": summary.get("patient_hash"),
                        "patient_name": summary.get("patient_name"),
                        "total_workflows": summary.get("workflow_count", 0),
                        "workflows": [],
                    }
                )

    if total_patients == 0:
        st.info("No patients found in the system. Upload an audio file to create the first patient record.")
    else:
        # Apply search filter
        filtered_patients = all_patients
        if search_query and len(search_query.strip()) > 0:
            query = search_query.strip().lower()
            filtered_patients = [p for p in all_patients if query in p.get("patient_hash", "").lower()]

        # Apply activity filter
        from datetime import datetime, timedelta

        if filter_option == "Active (last 7 days)":
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            filtered_patients = [
                p
                for p in filtered_patients
                if p.get("workflows") and any(w.get("created_at", "") > seven_days_ago for w in p.get("workflows", []))
            ]

        elif filter_option == "Inactive (>30 days)":
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            filtered_patients = [
                p
                for p in filtered_patients
                if p.get("workflows")
                and all(w.get("created_at", "") <= thirty_days_ago for w in p.get("workflows", []))
            ]

        elif filter_option == "Has Failed Workflows":
            filtered_patients = [
                p
                for p in filtered_patients
                if p.get("workflows") and any(w.get("status") == "FAILED" for w in p.get("workflows", []))
            ]

        # Apply sorting
        if sort_option == "Last Activity (newest first)":
            filtered_patients.sort(
                key=lambda p: max([w.get("created_at", "") for w in p.get("workflows", [])], default=""),
                reverse=True,
            )
        elif sort_option == "Last Activity (oldest first)":
            filtered_patients.sort(
                key=lambda p: max([w.get("created_at", "") for w in p.get("workflows", [])], default=""),
                reverse=False,
            )
        elif sort_option == "Most Workflows":
            filtered_patients.sort(key=lambda p: p.get("total_workflows", 0), reverse=True)
        elif sort_option == "Patient Hash":
            filtered_patients.sort(key=lambda p: p.get("patient_hash", ""))

        # Pagination
        PAGE_SIZE = 20
        total_filtered = len(filtered_patients)
        total_pages = (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE

        # Reset page if out of bounds
        if st.session_state.patient_page >= total_pages:
            st.session_state.patient_page = max(0, total_pages - 1)

        start_idx = st.session_state.patient_page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_filtered)
        page_patients = filtered_patients[start_idx:end_idx]

        # Display summary
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**Showing {len(page_patients)} of {total_filtered} patients** (Total in system: {total_patients})"
            )
        with col2:
            if total_pages > 1:
                st.caption(f"Page {st.session_state.patient_page + 1} of {total_pages}")

        # Display patient cards
        if page_patients:
            for idx, patient in enumerate(page_patients):
                render_patient_card(patient, start_idx + idx, api_client)
        else:
            st.warning("No patients match your search criteria. Try adjusting your filters.")

        # Pagination controls
        if total_pages > 1:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.session_state.patient_page > 0:
                    if st.button("← Previous Page", use_container_width=True):
                        st.session_state.patient_page -= 1
                        st.rerun()

            with col2:
                st.markdown(
                    f"<center><b>Page {st.session_state.patient_page + 1} of {total_pages}</b></center>",
                    unsafe_allow_html=True,
                )

            with col3:
                if st.session_state.patient_page < total_pages - 1:
                    if st.button("Next Page →", use_container_width=True):
                        st.session_state.patient_page += 1
                        st.rerun()

except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        st.info("No patients found. The database may be empty.")
    else:
        st.error(f"HTTP Error {e.response.status_code}: Unable to fetch patients")
        st.caption(str(e))

except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.caption("Please check that the backend server is running and accessible.")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown(
        """
    **Patient Browser Features:**

    - **Search**: Enter patient hash to find specific patients
    - **Filter**: Show only active, inactive, or patients with failed workflows
    - **Sort**: Organize by activity, workflow count, or hash
    - **Pagination**: Navigate through large patient lists (20 per page)

    **Protected Patient Names:**

    - Patient names (PHI) are hidden by default
    - Click "Show Name" to reveal (until page refresh)
    - Complies with HIPAA privacy requirements

    **Quick Actions:**

    - **View All Workflows**: See all consultations for a patient
    - **Latest Consultation**: Jump to the most recent workflow
    """
    )

# Footer
st.caption("💡 **Tip**: Use the search feature to quickly find patients by their hash.")
