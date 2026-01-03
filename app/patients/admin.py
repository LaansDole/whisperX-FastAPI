"""
SQLAdmin admin views for patient workflow management.

This module defines the admin interface for viewing and managing patient workflow mappings.
"""

from sqladmin import ModelView
from .models import PatientWorkflowMapping


class PatientWorkflowAdmin(ModelView, model=PatientWorkflowMapping):
    """Admin view for PatientWorkflowMapping model"""

    # Basic configuration
    name = "Patient Workflow"
    name_plural = "Patient Workflows"
    icon = "fa-solid fa-hospital-user"

    # Columns to display in list view
    column_list = [
        PatientWorkflowMapping.id,
        PatientWorkflowMapping.patient_hash,
        PatientWorkflowMapping.workflow_id,
        PatientWorkflowMapping.status,
        PatientWorkflowMapping.department,
        PatientWorkflowMapping.created_at,
    ]

    # Default sorting
    column_default_sort = [(PatientWorkflowMapping.created_at, True)]  # DESC

    # Searchable columns
    column_searchable_list = [
        PatientWorkflowMapping.patient_hash,
        PatientWorkflowMapping.workflow_id,
        PatientWorkflowMapping.patient_name,
    ]

    # Column labels (more user-friendly)
    column_labels = {
        PatientWorkflowMapping.id: "ID",
        PatientWorkflowMapping.patient_name: "Patient Name",
        PatientWorkflowMapping.patient_hash: "Patient Hash",
        PatientWorkflowMapping.workflow_id: "Workflow ID",
        PatientWorkflowMapping.file_path: "Audio File Path",
        PatientWorkflowMapping.department: "Department",
        PatientWorkflowMapping.created_at: "Created At",
        PatientWorkflowMapping.status: "Status",
    }

    # Formatters for better display
    column_formatters = {
        PatientWorkflowMapping.workflow_id: lambda m, a: m.workflow_id[:40] + "..."
        if len(m.workflow_id) > 40
        else m.workflow_id,
        PatientWorkflowMapping.file_path: lambda m, a: m.file_path.split("/")[-1],  # Show only filename
    }

    # Permissions (read-only by default for safety)
    can_create = False  # Workflows created via API only
    can_edit = True  # Allow status updates
    can_delete = True  # Allow cleanup of old records
    can_view_details = True
    can_export = True  # Allow CSV export

    # Columns shown in detail view
    column_details_list = [
        PatientWorkflowMapping.id,
        PatientWorkflowMapping.patient_name,
        PatientWorkflowMapping.patient_hash,
        PatientWorkflowMapping.workflow_id,
        PatientWorkflowMapping.file_path,
        PatientWorkflowMapping.department,
        PatientWorkflowMapping.created_at,
        PatientWorkflowMapping.status,
    ]

    # Columns shown in create/edit forms
    form_columns = [
        PatientWorkflowMapping.status,  # Only allow editing status
        PatientWorkflowMapping.department,  # And department
    ]

    # Help text
    column_descriptions = {
        PatientWorkflowMapping.patient_hash: "8-character cryptographic hash of patient name (HIPAA-compliant)",
        PatientWorkflowMapping.workflow_id: "Temporal workflow identifier - used to query workflow status",
        PatientWorkflowMapping.status: "Workflow status: pending (reserved), active (running), completed, failed",
    }

    # Pagination
    page_size = 50
    page_size_options = [25, 50, 100, 200]
