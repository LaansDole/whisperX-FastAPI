"""HIPAA-compliant filename generation utilities.

Generates secure filenames for medical transcription results without
exposing Protected Health Information (PHI).
"""

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from ..config import Config


def generate_patient_file_id(patient_id_encrypted: str) -> str:
    """Generate short hash from encrypted patient ID.
    
    Args:
        patient_id_encrypted: Encrypted patient identifier
        
    Returns:
        Short 8-character hash for use in filenames
    """
    # Create deterministic hash from encrypted ID
    hash_obj = hashlib.sha256(f"{patient_id_encrypted}{Config.HIPAA_SALT}".encode())
    return hash_obj.hexdigest()[:8]


def generate_consultation_filename(
    patient_id_encrypted: str,
    date: Optional[str] = None,
    department: Optional[str] = None,
    sequence: Optional[int] = None,
    extension: str = ".json"
) -> str:
    """Generate HIPAA-compliant filename for consultation transcription.
    
    Format: pt_{patient_hash}_{date}_{department}_{seq}{extension}
    Example: pt_a7f3c8e2_20251227_cardiology_001.json
    
    Args:
        patient_id_encrypted: Encrypted patient identifier
        date: Date string (YYYYMMDD), defaults to today
        department: Department name (sanitized)
        sequence: Sequence number for multiple consultations same day
        extension: File extension
        
    Returns:
        HIPAA-compliant filename
    """
    patient_hash = generate_patient_file_id(patient_id_encrypted)
    
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    
    # Sanitize department name (remove spaces, special chars)
    if department:
        department = "".join(c for c in department.lower() if c.isalnum() or c == "_")
    else:
        department = "general"
    
    # Build filename components
    components = [
        f"pt_{patient_hash}",
        date,
        department
    ]
    
    if sequence is not None:
        components.append(f"{sequence:03d}")
    
    filename = "_".join(components) + extension
    return filename


def generate_workflow_result_filename(
    workflow_id: str,
    extension: str = ".json"
) -> str:
    """Generate filename based on workflow ID.
    
    Format: wf_{workflow_id_short}{extension}
    Example: wf_abc123def456.json
    
    Args:
        workflow_id: Temporal workflow ID
        extension: File extension
        
    Returns:
        Filename based on workflow ID
    """
    # Extract UUID from workflow ID if present
    if "workflow-" in workflow_id:
        wf_uuid = workflow_id.split("workflow-")[-1]
        # Use first 12 chars of UUID
        short_id = wf_uuid.replace("-", "")[:12]
    else:
        short_id = workflow_id[:12]
    
    return f"wf_{short_id}{extension}"


def generate_anonymous_audio_filename(
    original_extension: str,
    patient_id_encrypted: Optional[str] = None
) -> str:
    """Generate anonymous filename for uploaded audio files.
    
    If patient_id_encrypted is provided, uses deterministic hash.
    Otherwise, uses random UUID.
    
    Args:
        original_extension: Original file extension (e.g., '.mp3')
        patient_id_encrypted: Optional encrypted patient ID
        
    Returns:
        Anonymous filename
    """
    if patient_id_encrypted:
        # Deterministic filename for same patient
        patient_hash = generate_patient_file_id(patient_id_encrypted)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"audio_{patient_hash}_{timestamp}{original_extension}"
    else:
        # Random UUID for anonymous uploads
        return f"{uuid.uuid4()}{original_extension}"


def extract_patient_id_from_filename(filename: str) -> Optional[str]:
    """Extract patient hash from HIPAA-compliant filename.
    
    Args:
        filename: Filename in format pt_{hash}_...
        
    Returns:
        Patient hash or None if not found
    """
    if filename.startswith("pt_"):
        parts = filename.split("_")
        if len(parts) >= 2:
            return parts[1]  # patient hash
    return None


def generate_result_storage_path(
    base_dir: str,
    patient_id_encrypted: str,
    filename: str
) -> str:
    """Generate full storage path for result files.
    
    Organizes files by patient hash subdirectory.
    Format: {base_dir}/{patient_hash[:2]}/{patient_hash}/{filename}
    
    Args:
        base_dir: Base storage directory
        patient_id_encrypted: Encrypted patient ID
        filename: File filename
        
    Returns:
        Full storage path
    """
    import os
    
    patient_hash = generate_patient_file_id(patient_id_encrypted)
    
    # Use first 2 chars for subdirectory (improves filesystem performance)
    subdir = patient_hash[:2]
    
    return os.path.join(base_dir, subdir, patient_hash, filename)
