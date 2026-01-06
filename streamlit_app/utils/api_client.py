"""
WhisperX API Client for Streamlit UI

This module provides a Python client for interacting with the WhisperX FastAPI backend.
"""

import httpx
from typing import Optional, Dict, Any
import streamlit as st


class WhisperXAPIClient:
    """Client for WhisperX FastAPI backend"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    def upload_audio(
        self,
        file_bytes: bytes,
        filename: str,
        patient_name: str,
        enable_medical: bool = False,
        provider_id: Optional[str] = None,
        encounter_date: Optional[str] = None,
        # WhisperX configuration options
        model: str = "base",
        language: str = "en",
        compute_type: str = "int8",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload audio file for transcription.

        Args:
            file_bytes: Audio file content
            filename: Original filename
            patient_name: Patient full name (HIPAA-compliant)
            enable_medical: Enable medical processing pipeline
            provider_id: Healthcare provider ID (required if enable_medical=True)
            encounter_date: Date of encounter (ISO format, optional)
            model: Whisper model to use (e.g., 'base', 'large-v3')
            language: Language code (en, vi, zh, yue)
            compute_type: Computation precision (int8, float16, float32)
            min_speakers: Minimum expected speakers for diarization
            max_speakers: Maximum expected speakers for diarization
            initial_prompt: Custom prompt with context or vocabulary hints

        Returns:
            Response with workflow ID

        Raises:
            httpx.HTTPError: If request fails
        """
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes)}
            data = {
                "patient_name": patient_name,
                "enable_medical_processing": str(enable_medical).lower(),
                # WhisperX model params
                "model": model,
                "language": language,
                "compute_type": compute_type,
            }

            if enable_medical and provider_id:
                data["provider_id"] = provider_id

            if encounter_date:
                data["encounter_date"] = encounter_date

            # Diarization params
            if min_speakers is not None:
                data["min_speakers"] = str(min_speakers)

            if max_speakers is not None:
                data["max_speakers"] = str(max_speakers)

            # ASR options
            if initial_prompt:
                data["initial_prompt"] = initial_prompt

            response = client.post(f"{self.base_url}/speech-to-text", files=files, data=data)
            response.raise_for_status()
            return response.json()

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get workflow status from Temporal.

        Args:
            workflow_id: Temporal workflow ID

        Returns:
            Workflow status information
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/temporal/workflow/{workflow_id}")
            response.raise_for_status()
            return response.json()

    def get_workflow_result(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get workflow result (transcription output).

        Args:
            workflow_id: Temporal workflow ID

        Returns:
            Transcription results with dialogue, entities, SOAP notes
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/temporal/workflow/{workflow_id}/result")
            response.raise_for_status()
            return response.json()

    def get_patient_workflows(
        self, patient_hash: str, status: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get workflows for a specific patient.

        Args:
            patient_hash: 8-character patient hash
            status: Optional status filter (RUNNING, COMPLETED, FAILED)
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip (pagination)

        Returns:
            Paginated list of workflows
        """
        with httpx.Client(timeout=self.timeout) as client:
            params = {"limit": limit, "offset": offset}
            if status:
                params["status"] = status

            response = client.get(f"{self.base_url}/temporal/patient/{patient_hash}/workflows", params=params)
            response.raise_for_status()
            return response.json()

    def get_latest_workflow(self, patient_hash: str) -> Dict[str, Any]:
        """
        Get the latest workflow for a patient.

        Args:
            patient_hash: 8-character patient hash

        Returns:
            Latest workflow information
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/temporal/patient/{patient_hash}/latest")
            response.raise_for_status()
            return response.json()

    def get_system_health(self) -> Dict[str, Any]:
        """
        Check system health (FastAPI, LM Studio, Vector DB).

        Returns:
            Health status of all components
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/health/medical")
            response.raise_for_status()
            return response.json()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics (admin endpoint).

        Returns:
            Database statistics including total patients and workflows
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/admin/database/stats")
            response.raise_for_status()
            return response.json()


@st.cache_resource
def get_api_client() -> WhisperXAPIClient:
    """
    Get cached API client instance.

    Returns:
        WhisperXAPIClient instance
    """
    import os

    # Try to get from secrets, fall back to environment variable, then default
    try:
        base_url = st.secrets.get("WHISPERX_API_URL", os.getenv("WHISPERX_API_URL", "http://localhost:8000"))
    except Exception:
        # If secrets.toml doesn't exist, use environment variable or default
        base_url = os.getenv("WHISPERX_API_URL", "http://localhost:8000")

    return WhisperXAPIClient(base_url=base_url)
