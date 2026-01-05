"""Utility functions for formatting and display"""

from datetime import datetime
from typing import Optional


def format_workflow_id(workflow_id: str) -> str:
    """
    Format workflow ID for display (truncate if too long).

    Args:
        workflow_id: Full workflow ID

    Returns:
        Formatted workflow ID
    """
    if len(workflow_id) > 40:
        return f"{workflow_id[:37]}..."
    return workflow_id


def extract_patient_hash(workflow_id: str) -> Optional[str]:
    """
    Extract patient hash from workflow ID.

    Workflow ID format: whisperx-wf-pt_<hash>-<timestamp>-<random>

    Args:
        workflow_id: Workflow ID

    Returns:
        8-character patient hash or None
    """
    try:
        if "pt_" in workflow_id:
            parts = workflow_id.split("pt_")[1]
            return parts.split("-")[0][:8]
    except (IndexError, AttributeError):
        pass
    return None


def format_timestamp(iso_timestamp: str) -> str:
    """
    Format ISO timestamp to human-readable format.

    Args:
        iso_timestamp: ISO format timestamp

    Returns:
        Human-readable timestamp
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return iso_timestamp


def format_time_ago(iso_timestamp: str) -> str:
    """
    Format timestamp as "X minutes ago" or "X hours ago".

    Args:
        iso_timestamp: ISO format timestamp

    Returns:
        Relative time string
    """
    try:
        from zoneinfo import ZoneInfo

        # Parse the ISO timestamp (keep timezone info)
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))

        # Get current time in UTC+7
        bangkok_tz = ZoneInfo("Asia/Bangkok")
        now = datetime.now(bangkok_tz)

        # Calculate difference
        diff = now - dt

        seconds = diff.total_seconds()
        if seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
    except (ValueError, AttributeError):
        return iso_timestamp


def get_status_emoji(status: str) -> str:
    """
    Get emoji for workflow status.

    Args:
        status: Workflow status (RUNNING, COMPLETED, FAILED, etc.)

    Returns:
        Status emoji
    """
    status_map = {
        "RUNNING": "⏳",
        "COMPLETED": "✅",
        "FAILED": "❌",
        "PENDING": "🕐",
        "CANCELLED": "🚫",
        "TIMED_OUT": "⏱️",
    }
    return status_map.get(status.upper(), "❓")


def get_status_color(status: str) -> str:
    """
    Get color for workflow status badge.

    Args:
        status: Workflow status

    Returns:
        Color name for Streamlit
    """
    color_map = {
        "RUNNING": "blue",
        "COMPLETED": "green",
        "FAILED": "red",
        "PENDING": "orange",
        "CANCELLED": "gray",
    }
    return color_map.get(status.upper(), "gray")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."
