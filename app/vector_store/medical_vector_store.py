"""FAISS + SQLite vector store for medical document storage and retrieval."""

import faiss
import numpy as np
import sqlite3
import hashlib
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalDocumentVectorStore:
    """Simplified vector store using FAISS + SQLite for medical consultations."""

    def __init__(
        self,
        storage_dir: str = "./vector_storage",
        embedding_dim: int = 768,
        index_type: str = "IndexFlatL2"
    ):
        """
        Initialize vector store.

        Args:
            storage_dir: Directory to store vector index and SQLite database
            embedding_dim: Dimension of embedding vectors (768 for nomic-embed-text-v1.5)
            index_type: FAISS index type (IndexFlatL2 for exact search)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)

        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Initialize FAISS index
        self.index = self._create_index()

        # Initialize SQLite database
        self.db_path = self.storage_dir / "medical_metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()

        # Load existing index if available
        self._load_index()

    def _create_index(self):
        """Create FAISS index based on configured type."""
        if self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexFlatIP":  # Inner Product
            return faiss.IndexFlatIP(self.embedding_dim)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
            return faiss.IndexFlatL2(self.embedding_dim)

    def _init_database(self):
        """Initialize SQLite schema for medical metadata."""
        cursor = self.conn.cursor()

        # Consultations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT UNIQUE NOT NULL,
                patient_id_encrypted TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                encounter_date TEXT NOT NULL,
                transcript_hash TEXT NOT NULL,
                transcript_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                vector_id INTEGER,
                metadata_json TEXT
            )
        """)

        # Medical entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_text TEXT NOT NULL,
                normalized_text TEXT,
                icd_code TEXT,
                confidence REAL,
                start_position INTEGER,
                end_position INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
            )
        """)

        # PHI detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phi_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT NOT NULL,
                phi_type TEXT NOT NULL,
                phi_text TEXT NOT NULL,
                start_position INTEGER,
                end_position INTEGER,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
            )
        """)

        # Structured documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structured_documents (
                consultation_id TEXT PRIMARY KEY,
                document_json TEXT NOT NULL,
                soap_note_json TEXT,
                clinical_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consultations_patient ON consultations(patient_id_encrypted)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consultations_date ON consultations(encounter_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_consultation ON medical_entities(consultation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON medical_entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phi_consultation ON phi_detections(consultation_id)")

        self.conn.commit()

    def _load_index(self):
        """Load existing FAISS index from disk if available."""
        index_path = self.storage_dir / "faiss_index.bin"
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                # Fall back to new index
                self.index = self._create_index()

    def save_index(self):
        """Persist FAISS index to disk."""
        index_path = self.storage_dir / "faiss_index.bin"
        try:
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _generate_consultation_id(self, transcript: str, patient_id: str, encounter_date: str) -> str:
        """Generate unique consultation ID from transcript hash."""
        content = f"{patient_id}_{encounter_date}_{transcript[:1000]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_transcript_hash(self, transcript: str) -> str:
        """Generate hash of transcript content for deduplication."""
        return hashlib.sha256(transcript.encode()).hexdigest()

    async def store_consultation(
        self,
        consultation_id: str,
        patient_id_encrypted: str,
        provider_id: str,
        encounter_date: str,
        transcript: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Store consultation embedding and metadata.

        Args:
            consultation_id: Unique consultation identifier
            patient_id_encrypted: Encrypted patient identifier
            provider_id: Provider identifier
            encounter_date: Date of encounter (ISO format)
            transcript: Full consultation transcript
            embedding: Vector embedding of transcript
            metadata: Additional metadata dictionary

        Returns:
            Vector ID in FAISS index
        """
        try:
            # Normalize embedding for better search
            embedding_normalized = embedding / np.linalg.norm(embedding)

            # Add to FAISS index
            self.index.add(embedding_normalized.reshape(1, -1))
            vector_id = self.index.ntotal - 1

            # Store metadata in SQLite
            transcript_hash = self._generate_transcript_hash(transcript)

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO consultations
                (consultation_id, patient_id_encrypted, provider_id, encounter_date,
                 transcript_hash, transcript_length, vector_id, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consultation_id,
                patient_id_encrypted,
                provider_id,
                encounter_date,
                transcript_hash,
                len(transcript),
                vector_id,
                json.dumps(metadata or {}),
                datetime.now().isoformat()
            ))

            self.conn.commit()
            logger.info(f"Stored consultation {consultation_id} with vector_id {vector_id}")

            return vector_id

        except Exception as e:
            logger.error(f"Failed to store consultation: {e}")
            self.conn.rollback()
            raise

    async def store_medical_entities(
        self,
        consultation_id: str,
        entities: List[Dict[str, Any]]
    ):
        """
        Store medical entities for a consultation.

        Args:
            consultation_id: Consultation identifier
            entities: List of medical entity dictionaries
        """
        try:
            cursor = self.conn.cursor()

            for entity in entities:
                cursor.execute("""
                    INSERT INTO medical_entities
                    (consultation_id, entity_type, entity_text, normalized_text,
                     icd_code, confidence, start_position, end_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    consultation_id,
                    entity.get("type"),
                    entity.get("text"),
                    entity.get("normalized"),
                    entity.get("code"),
                    entity.get("confidence"),
                    entity.get("start"),
                    entity.get("end")
                ))

            self.conn.commit()
            logger.info(f"Stored {len(entities)} medical entities for {consultation_id}")

        except Exception as e:
            logger.error(f"Failed to store medical entities: {e}")
            self.conn.rollback()
            raise

    async def store_phi_detections(
        self,
        consultation_id: str,
        phi_entities: List[Dict[str, Any]]
    ):
        """
        Store PHI detections for a consultation.

        Args:
            consultation_id: Consultation identifier
            phi_entities: List of PHI entity dictionaries
        """
        try:
            cursor = self.conn.cursor()

            for phi in phi_entities:
                cursor.execute("""
                    INSERT INTO phi_detections
                    (consultation_id, phi_type, phi_text, start_position,
                     end_position, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    consultation_id,
                    phi.get("type"),
                    phi.get("text"),
                    phi.get("start"),
                    phi.get("end"),
                    phi.get("confidence")
                ))

            self.conn.commit()
            logger.info(f"Stored {len(phi_entities)} PHI detections for {consultation_id}")

        except Exception as e:
            logger.error(f"Failed to store PHI detections: {e}")
            self.conn.rollback()
            raise

    async def store_structured_document(
        self,
        consultation_id: str,
        structured_doc: Dict[str, Any],
        soap_note: Dict[str, str] = None,
        clinical_summary: str = None
    ):
        """
        Store structured medical document.

        Args:
            consultation_id: Consultation identifier
            structured_doc: Structured document dictionary
            soap_note: SOAP note sections
            clinical_summary: Clinical summary text
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO structured_documents
                (consultation_id, document_json, soap_note_json, clinical_summary, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                consultation_id,
                json.dumps(structured_doc),
                json.dumps(soap_note) if soap_note else None,
                clinical_summary,
                datetime.now().isoformat()
            ))

            self.conn.commit()
            logger.info(f"Stored structured document for {consultation_id}")

        except Exception as e:
            logger.error(f"Failed to store structured document: {e}")
            self.conn.rollback()
            raise

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        patient_id_encrypted: str = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar consultations.

        Args:
            query_embedding: Query vector embedding
            patient_id_encrypted: Filter by patient (None for all patients)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar consultations with metadata
        """
        try:
            # Normalize query
            query_normalized = query_embedding / np.linalg.norm(query_embedding)

            # Search FAISS index (get more results for filtering)
            k = min(limit * 3, self.index.ntotal) if self.index.ntotal > 0 else 0
            if k == 0:
                return []

            distances, indices = self.index.search(
                query_normalized.reshape(1, -1),
                k
            )

            # Get metadata from SQLite
            results = []
            cursor = self.conn.cursor()

            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                # Calculate similarity score (convert distance to similarity)
                similarity = float(1 / (1 + distance))
                if similarity < similarity_threshold:
                    continue

                # Build query with optional patient filter
                query = """
                    SELECT consultation_id, patient_id_encrypted, provider_id,
                           encounter_date, transcript_hash, transcript_length,
                           created_at, metadata_json
                    FROM consultations
                    WHERE rowid = ?
                """
                params = [int(idx) + 1]  # Convert to 1-based rowid

                if patient_id_encrypted:
                    query += " AND patient_id_encrypted = ?"
                    params.append(patient_id_encrypted)

                cursor.execute(query, params)
                row = cursor.fetchone()

                if row:
                    result = {
                        "consultation_id": row[0],
                        "patient_id_encrypted": row[1],
                        "provider_id": row[2],
                        "encounter_date": row[3],
                        "similarity_score": similarity,
                        "vector_id": idx,
                        "transcript_hash": row[4],
                        "transcript_length": row[5],
                        "created_at": row[6],
                        "metadata": json.loads(row[7]) if row[7] else {}
                    }
                    results.append(result)

                if len(results) >= limit:
                    break

            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            logger.info(f"Found {len(results)} similar consultations")
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search similar consultations: {e}")
            return []

    async def get_consultation_details(
        self,
        consultation_id: str,
        include_entities: bool = True,
        include_phi: bool = False,
        include_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete consultation details.

        Args:
            consultation_id: Consultation identifier
            include_entities: Include medical entities
            include_phi: Include PHI detections (use carefully)
            include_structured: Include structured document

        Returns:
            Complete consultation details
        """
        try:
            cursor = self.conn.cursor()

            # Get basic consultation info
            cursor.execute("""
                SELECT * FROM consultations WHERE consultation_id = ?
            """, (consultation_id,))

            consultation_row = cursor.fetchone()
            if not consultation_row:
                return None

            result = {
                "consultation_id": consultation_row[1],
                "patient_id_encrypted": consultation_row[2],
                "provider_id": consultation_row[3],
                "encounter_date": consultation_row[4],
                "transcript_hash": consultation_row[5],
                "transcript_length": consultation_row[6],
                "created_at": consultation_row[7],
                "vector_id": consultation_row[9],
                "metadata": json.loads(consultation_row[10]) if consultation_row[10] else {}
            }

            # Get medical entities
            if include_entities:
                cursor.execute("""
                    SELECT entity_type, entity_text, normalized_text, icd_code,
                           confidence, start_position, end_position
                    FROM medical_entities
                    WHERE consultation_id = ?
                    ORDER BY start_position
                """, (consultation_id,))

                result["medical_entities"] = [
                    {
                        "type": row[0],
                        "text": row[1],
                        "normalized": row[2],
                        "icd_code": row[3],
                        "confidence": row[4],
                        "start_position": row[5],
                        "end_position": row[6]
                    }
                    for row in cursor.fetchall()
                ]

            # Get PHI detections (only if explicitly requested)
            if include_phi:
                cursor.execute("""
                    SELECT phi_type, phi_text, start_position, end_position, confidence
                    FROM phi_detections
                    WHERE consultation_id = ?
                    ORDER BY start_position
                """, (consultation_id,))

                result["phi_detections"] = [
                    {
                        "type": row[0],
                        "text": row[1],
                        "start_position": row[2],
                        "end_position": row[3],
                        "confidence": row[4]
                    }
                    for row in cursor.fetchall()
                ]

            # Get structured document
            if include_structured:
                cursor.execute("""
                    SELECT document_json, soap_note_json, clinical_summary
                    FROM structured_documents
                    WHERE consultation_id = ?
                """, (consultation_id,))

                structured_row = cursor.fetchone()
                if structured_row:
                    result["structured_document"] = json.loads(structured_row[0]) if structured_row[0] else {}
                    result["soap_note"] = json.loads(structured_row[1]) if structured_row[1] else {}
                    result["clinical_summary"] = structured_row[2]

            return result

        except Exception as e:
            logger.error(f"Failed to get consultation details: {e}")
            return None

    async def get_patient_consultations(
        self,
        patient_id_encrypted: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all consultations for a specific patient.

        Args:
            patient_id_encrypted: Encrypted patient identifier
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of consultation summaries
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT consultation_id, provider_id, encounter_date,
                       transcript_length, created_at, metadata_json
                FROM consultations
                WHERE patient_id_encrypted = ?
                ORDER BY encounter_date DESC
                LIMIT ? OFFSET ?
            """, (patient_id_encrypted, limit, offset))

            consultations = []
            for row in cursor.fetchall():
                consultation = {
                    "consultation_id": row[0],
                    "provider_id": row[1],
                    "encounter_date": row[2],
                    "transcript_length": row[3],
                    "created_at": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {}
                }
                consultations.append(consultation)

            return consultations

        except Exception as e:
            logger.error(f"Failed to get patient consultations: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            cursor = self.conn.cursor()

            # Count consultations
            cursor.execute("SELECT COUNT(*) FROM consultations")
            total_consultations = cursor.fetchone()[0]

            # Count unique patients
            cursor.execute("SELECT COUNT(DISTINCT patient_id_encrypted) FROM consultations")
            unique_patients = cursor.fetchone()[0]

            # Count entities by type
            cursor.execute("""
                SELECT entity_type, COUNT(*)
                FROM medical_entities
                GROUP BY entity_type
            """)
            entity_counts = dict(cursor.fetchall())

            return {
                "total_consultations": total_consultations,
                "unique_patients": unique_patients,
                "vectors_in_index": self.index.ntotal,
                "embedding_dimension": self.embedding_dim,
                "index_type": self.index_type,
                "entity_counts": entity_counts,
                "storage_path": str(self.storage_dir)
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close database connection and save index."""
        self.save_index()
        self.conn.close()
        logger.info("Vector store closed")