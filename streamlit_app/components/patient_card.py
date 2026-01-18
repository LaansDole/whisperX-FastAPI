"""
Patient Card Component

Reusable component for displaying patient summary information with statistics and actions.
"""

import streamlit as st
from components.protected_name import render_protected_name


def render_patient_card(patient_data: dict, index: int, api_client):
    """
    Render a patient summary card with stats and navigation actions.

    Args:
        patient_data: Dictionary with patient information
            - patient_hash: 8-character hash
            - patient_name: Plain text name (PHI)
            - total_workflows: Total workflow count
            - workflows: List of workflow dictionaries
        index: Unique index for widget keys
        api_client: API client instance for fetching additional data

    Displays:
        - Protected patient name
        - Workflow statistics (total, completed, failed, running)
        - Last activity timestamp
        - Action buttons (View Workflows, Latest Consultation)
    """
    patient_hash = patient_data.get("patient_hash", "Unknown")
    patient_name = patient_data.get("patient_name", "Unknown")
    total_workflows = patient_data.get("total_workflows", 0)
    workflows = patient_data.get("workflows", [])

    # Calculate statistics
    # Note: DB status is "active" for completed workflows, "pending" for in-progress
    # We'll treat "active" as completed for now
    completed_count = sum(1 for w in workflows if w.get("status") == "active")
    failed_count = 0  # Not tracked in DB, would need to query Temporal
    running_count = sum(1 for w in workflows if w.get("status") == "pending")

    # Find latest workflow
    latest_workflow = workflows[0] if workflows else None
    last_activity = latest_workflow.get("created_at", "N/A") if latest_workflow else "N/A"

    # Determine medical processing adoption
    medical_count = sum(1 for w in workflows if "medical" in str(w.get("workflow_id", "")).lower())
    medical_percentage = (medical_count / max(total_workflows, 1)) * 100 if total_workflows > 0 else 0

    with st.expander(f"Patient {patient_hash}", expanded=False):
        # Protected patient name
        render_protected_name(patient_hash, patient_name, str(index), inline=False)

        st.divider()

        # Statistics
        st.markdown("**📊 Statistics**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Workflows", total_workflows)

        with col2:
            st.metric("Completed", completed_count, delta=f"{(completed_count / max(total_workflows, 1) * 100):.0f}%")

        with col3:
            st.metric("Failed", failed_count, delta=None if failed_count == 0 else f"{failed_count} issues")

        with col4:
            st.metric("Running", running_count)

        # Additional stats
        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"**Medical Processing**: {medical_count}/{total_workflows} ({medical_percentage:.0f}%)")

        with col2:
            from utils.formatting import format_time_ago

            st.caption(f"**Last Activity**: {format_time_ago(last_activity)}")

        st.divider()

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("View All Workflows →", key=f"view_wf_{index}", use_container_width=True):
                st.session_state.selected_patient = patient_hash
                st.switch_page("pages/3_🔍_Workflows.py")

        with col2:
            if latest_workflow and st.button("Latest Consultation →", key=f"latest_{index}", use_container_width=True):
                st.session_state.selected_workflow = latest_workflow.get("workflow_id")
                st.switch_page("pages/3_🔍_Workflows.py")
