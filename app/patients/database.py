"""SQLite database for patient workflow mappings."""

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from ..logger import logger


# Database location
DB_PATH = Path("./data/patient_mappings.db")


def init_database(fresh_start: bool = True):
    """
    Initialize SQLite database with schema.
    
    Args:
        fresh_start: If True, delete existing database for clean slate
    """
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Fresh start: delete existing database
    if fresh_start and DB_PATH.exists():
        logger.warning(f"CLEARING DATABASE for fresh start: {DB_PATH}")
        os.remove(DB_PATH)
        logger.info(f"Database cleared successfully")
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create patient_workflow_mappings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_workflow_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_hash TEXT NOT NULL,
            workflow_id TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            department TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(workflow_id)
        )
    """)
    
    # Create index on patient_hash for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_hash 
        ON patient_workflow_mappings(patient_hash)
    """)
    
    # Create index on workflow_id for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_workflow_id 
        ON patient_workflow_mappings(workflow_id)
    """)
    
    conn.commit()
    conn.close()
    
    logger.info(f"SQLite database initialized at {DB_PATH}")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        conn.close()


def store_patient_workflow_db(
    patient_name: str,
    patient_hash: str,
    workflow_id: str,
    file_path: str,
    department: Optional[str] = None,
    created_at: str = None
):
    """
    Store patient-workflow mapping in SQLite.
    
    Args:
        patient_name: Plain text patient name
        patient_hash: 8-char patient hash
        workflow_id: Temporal workflow ID
        file_path: Path to audio file
        department: Optional department name
        created_at: ISO timestamp (auto-generated if None)
    """
    if created_at is None:
        from datetime import datetime
        created_at = datetime.now().isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_workflow_mappings 
            (patient_name, patient_hash, workflow_id, file_path, department, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_name, patient_hash, workflow_id, file_path, department, created_at))
        
        # Real-time logging
        logger.info(f"DB INSERT: {patient_name} ({patient_hash}) -> {workflow_id}")
        logger.info(f"   File: {file_path}")
        if department:
            logger.info(f"   Department: {department}")
        
        # Show total count
        cursor.execute("SELECT COUNT(*) FROM patient_workflow_mappings")
        total = cursor.fetchone()[0]
        logger.info(f"   Total mappings in DB: {total}")


def get_patient_by_workflow_db(workflow_id: str) -> Optional[dict]:
    """
    Get patient info by workflow ID from SQLite.
    
    Args:
        workflow_id: Workflow ID
        
    Returns:
        Patient mapping dict or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_name, patient_hash, workflow_id, file_path, department, created_at
            FROM patient_workflow_mappings
            WHERE workflow_id = ?
        """, (workflow_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_workflows_by_patient_hash_db(patient_hash: str) -> list:
    """
    Get all workflows for a patient by hash from SQLite.
    
    Args:
        patient_hash: 8-char patient hash
        
    Returns:
        List of workflow mapping dicts
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_name, patient_hash, workflow_id, file_path, department, created_at
            FROM patient_workflow_mappings
            WHERE patient_hash = ?
            ORDER BY created_at DESC
        """, (patient_hash,))
        
        rows = cursor.fetchall()
        
        # Real-time logging
        logger.info(f"DB QUERY: patient_hash={patient_hash} -> Found {len(rows)} workflows")
        
        return [dict(row) for row in rows]


def get_patient_name_by_hash_db(patient_hash: str) -> Optional[str]:
    """
    Get patient name by hash from SQLite.
    
    Args:
        patient_hash: 8-char patient hash
        
    Returns:
        Plain text patient name or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_name
            FROM patient_workflow_mappings
            WHERE patient_hash = ?
            LIMIT 1
        """, (patient_hash,))
        
        row = cursor.fetchone()
        if row:
            return row["patient_name"]
        return None


def get_all_patients_db() -> list:
    """
    Get summary of all patients with workflow counts.
    
    Returns:
        List of patient summaries
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                patient_hash,
                patient_name,
                COUNT(*) as workflow_count,
                MAX(created_at) as latest_workflow
            FROM patient_workflow_mappings
            GROUP BY patient_hash
            ORDER BY latest_workflow DESC
        """)
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# Initialize database on module import
# Only do fresh start if explicitly requested via environment variable
import os
fresh_start_on_init = os.getenv("DB_FRESH_START", "false").lower() == "true"
init_database(fresh_start=fresh_start_on_init)
