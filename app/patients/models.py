"""
SQLAlchemy models for SQLAdmin integration.

This module defines SQLAlchemy ORM models that mirror the existing SQLite database schema.
These models are used ONLY for the admin panel - the main application continues to use
direct SQL queries for simplicity and performance.
"""

from sqlalchemy import Column, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..config import Config

# SQLAlchemy Base
Base = declarative_base()


class PatientWorkflowMapping(Base):
    """
    SQLAlchemy model for patient_workflow_mappings table.

    This matches the existing SQLite schema created in app/patients/database.py
    """

    __tablename__ = "patient_workflow_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(
        Text, nullable=False, comment="Plain text patient name (HIPAA-compliant hashing done separately)"
    )
    patient_hash = Column(Text, nullable=False, index=True, comment="8-character patient hash for HIPAA compliance")
    workflow_id = Column(Text, nullable=False, unique=True, comment="Temporal workflow ID")
    file_path = Column(Text, nullable=False, comment="Path to uploaded audio file")
    department = Column(Text, nullable=True, comment="Department name (optional)")
    created_at = Column(Text, nullable=False, comment="ISO timestamp with timezone")
    status = Column(
        Text, nullable=False, default="pending", comment="Workflow status: pending, active, completed, failed"
    )

    def __repr__(self):
        return f"<PatientWorkflow(hash={self.patient_hash}, workflow={self.workflow_id[:20]}...)>"


# Create engine and session for SQLAdmin
engine = create_engine(f"sqlite:///{Config.PATIENT_DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session():
    """Get database session for SQLAdmin"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
