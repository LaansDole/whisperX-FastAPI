"""Patient-related modules for HIPAA-compliant workflow management."""

from .filename_utils import (
    generate_patient_file_id,
    generate_consultation_filename,
    generate_workflow_result_filename,
    generate_anonymous_audio_filename,
    extract_patient_id_from_filename,
    generate_result_storage_path,
)

from .mapping import (
    store_patient_workflow,
    get_patient_by_workflow,
    get_workflows_by_patient_hash,
    get_patient_name_by_hash,
    get_all_patients,
)

__all__ = [
    # Filename utilities
    "generate_patient_file_id",
    "generate_consultation_filename",
    "generate_workflow_result_filename",
    "generate_anonymous_audio_filename",
    "extract_patient_id_from_filename",
    "generate_result_storage_path",
    # Patient mapping
    "store_patient_workflow",
    "get_patient_by_workflow",
    "get_workflows_by_patient_hash",
    "get_patient_name_by_hash",
    "get_all_patients",
]
