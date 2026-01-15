"""Unit tests for MedicalDocumentVectorStore.

Tests for vector storage and retrieval functionality, including:
- Patient-filtered similarity search
- Correct vector_id vs rowid handling
- Search with vectors not in global top-k
"""

import pytest
import numpy as np
import tempfile
import shutil

from app.vector_store.medical_vector_store import MedicalDocumentVectorStore


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = MedicalDocumentVectorStore(
        storage_dir=temp_dir,
        embedding_dim=128,  # Smaller for faster tests
        index_type="IndexFlatL2",
    )
    yield store
    store.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store_with_multi_patient_data(temp_vector_store):
    """Create a vector store with consultations from multiple patients.

    This simulates the real-world scenario where:
    - Multiple patients have consultations
    - One patient's vectors may not be in the global top-k
    - Patient filtering must search all vectors
    """
    store = temp_vector_store

    # Patient A: 3 consultations with similar embeddings (topic: cardiology)
    patient_a_embedding = np.random.rand(128).astype(np.float32)
    for i in range(3):
        consultation_id = f"cons_patient_a_{i}"
        embedding = patient_a_embedding + np.random.rand(128).astype(np.float32) * 0.01  # Very similar
        store.conn.execute("BEGIN")
        try:
            vector_id = store.index.ntotal
            store.index.add((embedding / np.linalg.norm(embedding)).reshape(1, -1))
            store.conn.execute(
                """
                INSERT INTO consultations
                (consultation_id, patient_id_encrypted, provider_id, encounter_date,
                 transcript_hash, transcript_length, vector_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    consultation_id,
                    "patient_a_hash",
                    "dr_cardio",
                    "2025-01-10",
                    f"hash_{i}",
                    100,
                    vector_id,
                    "{}",
                ),
            )
            # Add SOAP note
            store.conn.execute(
                """
                INSERT INTO structured_documents
                (consultation_id, document_json, soap_note_json)
                VALUES (?, ?, ?)
                """,
                (
                    consultation_id,
                    "{}",
                    '{"subjective": "Chest pain", "objective": "BP 140/90", "assessment": "Hypertension", "plan": "Monitor BP"}',
                ),
            )
            store.conn.commit()
        except Exception:
            store.conn.rollback()
            raise

    # Patient B: 2 consultations with different embeddings (topic: dermatology)
    patient_b_embedding = np.random.rand(128).astype(np.float32)
    for i in range(2):
        consultation_id = f"cons_patient_b_{i}"
        embedding = patient_b_embedding + np.random.rand(128).astype(np.float32) * 0.01
        store.conn.execute("BEGIN")
        try:
            vector_id = store.index.ntotal
            store.index.add((embedding / np.linalg.norm(embedding)).reshape(1, -1))
            store.conn.execute(
                """
                INSERT INTO consultations
                (consultation_id, patient_id_encrypted, provider_id, encounter_date,
                 transcript_hash, transcript_length, vector_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    consultation_id,
                    "patient_b_hash",
                    "dr_derm",
                    "2025-01-11",
                    f"hash_b_{i}",
                    100,
                    vector_id,
                    "{}",
                ),
            )
            # Add SOAP note
            store.conn.execute(
                """
                INSERT INTO structured_documents
                (consultation_id, document_json, soap_note_json)
                VALUES (?, ?, ?)
                """,
                (
                    consultation_id,
                    "{}",
                    '{"subjective": "Rash", "objective": "Red patches", "assessment": "Eczema", "plan": "Topical cream"}',
                ),
            )
            store.conn.commit()
        except Exception:
            store.conn.rollback()
            raise

    return store


class TestVectorStoreBasicOperations:
    """Test basic vector store operations."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_consultation(self, temp_vector_store):
        """Test storing and retrieving a single consultation."""
        store = temp_vector_store

        embedding = np.random.rand(128).astype(np.float32)
        vector_id = await store.store_consultation(
            consultation_id="test_cons_001",
            patient_id_encrypted="test_patient",
            provider_id="dr_test",
            encounter_date="2025-01-15",
            transcript="Test transcript",
            embedding=embedding,
            metadata={"test": "metadata"},
        )

        assert vector_id == 0
        assert store.index.ntotal == 1

        # Verify database entry
        cursor = store.conn.cursor()
        cursor.execute(
            "SELECT consultation_id, vector_id FROM consultations WHERE consultation_id = ?", ("test_cons_001",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test_cons_001"
        assert row[1] == vector_id

    @pytest.mark.asyncio
    async def test_multiple_consultations_different_vector_ids(self, temp_vector_store):
        """Test that multiple consultations get sequential vector_ids."""
        store = temp_vector_store

        for i in range(5):
            embedding = np.random.rand(128).astype(np.float32)
            vector_id = await store.store_consultation(
                consultation_id=f"cons_{i}",
                patient_id_encrypted="patient_test",
                provider_id="dr_test",
                encounter_date="2025-01-15",
                transcript=f"Transcript {i}",
                embedding=embedding,
            )
            assert vector_id == i


class TestPatientFilteredSearch:
    """Test patient-filtered similarity search - THE CRITICAL BUG FIX."""

    @pytest.mark.asyncio
    async def test_search_finds_patient_vectors_not_in_global_topk(self, vector_store_with_multi_patient_data):
        """REGRESSION TEST: Search must find patient vectors even if not in global top-k.

        This test reproduces the bug where:
        1. Patient B's vectors rank lower globally than Patient A's
        2. But when searching for Patient B, we MUST still find them
        3. The fix: search ALL vectors (k=ntotal), then filter by patient
        """
        store = vector_store_with_multi_patient_data

        # Create a query that's similar to Patient A's embeddings (cardiology topic)
        # This ensures Patient A's vectors will rank higher globally
        query_embedding = np.random.rand(128).astype(np.float32)
        # Make it similar to the first patient A embedding
        cursor = store.conn.cursor()
        cursor.execute("SELECT vector_id FROM consultations WHERE patient_id_encrypted = 'patient_a_hash' LIMIT 1")
        patient_a_vector_id = cursor.fetchone()[0]
        patient_a_vector = store.index.reconstruct(int(patient_a_vector_id))
        query_embedding = patient_a_vector + np.random.rand(128).astype(np.float32) * 0.05

        # Search for Patient B (whose vectors are less similar to the query)
        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted="patient_b_hash",
            limit=5,
            similarity_threshold=0.3,
        )

        # CRITICAL: Must find Patient B's consultations even though they rank lower globally
        assert len(results) == 2, f"Expected 2 results for patient_b_hash, got {len(results)}"
        assert all(r["patient_id_encrypted"] == "patient_b_hash" for r in results)
        assert results[0]["consultation_id"] in ["cons_patient_b_0", "cons_patient_b_1"]

    @pytest.mark.asyncio
    async def test_search_without_patient_filter_returns_global_topk(self, vector_store_with_multi_patient_data):
        """Test that search without patient filter returns global top-k."""
        store = vector_store_with_multi_patient_data

        # Create a query similar to Patient A
        query_embedding = np.random.rand(128).astype(np.float32)

        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted=None,  # No patient filter
            limit=3,
            similarity_threshold=0.0,
        )

        # Should return top 3 globally (likely mix of patients)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_with_high_threshold_filters_low_similarity(self, vector_store_with_multi_patient_data):
        """Test that high similarity threshold filters out low-similarity results."""
        store = vector_store_with_multi_patient_data

        # Random query (likely low similarity to all stored vectors)
        query_embedding = np.random.rand(128).astype(np.float32)

        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted="patient_a_hash",
            limit=5,
            similarity_threshold=0.99,  # Very high threshold
        )

        # Should return 0 or very few results due to high threshold
        assert len(results) <= 3


class TestVectorIdVsRowId:
    """Test correct usage of vector_id vs rowid - THE CRITICAL BUG FIX."""

    @pytest.mark.asyncio
    async def test_search_uses_vector_id_not_rowid(self, temp_vector_store):
        """REGRESSION TEST: Search must use vector_id, not rowid for lookup.

        This test ensures the SQL query uses:
            WHERE vector_id = ?
        instead of:
            WHERE rowid = ?
        """
        store = temp_vector_store

        # Store consultations
        embeddings = []
        for i in range(3):
            embedding = np.random.rand(128).astype(np.float32)
            embeddings.append(embedding)
            await store.store_consultation(
                consultation_id=f"cons_{i}",
                patient_id_encrypted="patient_test",
                provider_id="dr_test",
                encounter_date="2025-01-15",
                transcript=f"Transcript {i}",
                embedding=embedding,
            )

        # Search with the exact first embedding
        results = await store.search_similar(
            query_embedding=embeddings[0],
            patient_id_encrypted="patient_test",
            limit=3,
            similarity_threshold=0.5,
        )

        # Should find all 3 consultations
        assert len(results) == 3
        # First result should be exact match (highest similarity)
        assert results[0]["consultation_id"] == "cons_0"
        assert results[0]["similarity_score"] > 0.99  # Near perfect match

        # Verify vector_id matches what's returned
        for result in results:
            cursor = store.conn.cursor()
            cursor.execute(
                "SELECT consultation_id FROM consultations WHERE vector_id = ?",
                (int(result["vector_id"]),),  # Convert numpy.int64 to Python int
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == result["consultation_id"]


class TestSearchLimits:
    """Test search limit and k-value handling."""

    @pytest.mark.asyncio
    async def test_search_respects_limit_parameter(self, vector_store_with_multi_patient_data):
        """Test that limit parameter controls number of results returned."""
        store = vector_store_with_multi_patient_data

        query_embedding = np.random.rand(128).astype(np.float32)

        # Search with limit=1
        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted="patient_a_hash",
            limit=1,
            similarity_threshold=0.0,
        )
        assert len(results) <= 1

        # Search with limit=2
        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted="patient_a_hash",
            limit=2,
            similarity_threshold=0.0,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_with_empty_index_returns_empty(self, temp_vector_store):
        """Test that search on empty index returns empty list."""
        store = temp_vector_store

        query_embedding = np.random.rand(128).astype(np.float32)
        results = await store.search_similar(
            query_embedding=query_embedding,
            patient_id_encrypted="any_patient",
            limit=10,
        )

        assert results == []


class TestConsultationDetails:
    """Test getting consultation details with enriched data."""

    @pytest.mark.asyncio
    async def test_get_consultation_details_includes_soap(self, vector_store_with_multi_patient_data):
        """Test that consultation details include SOAP notes."""
        store = vector_store_with_multi_patient_data

        details = await store.get_consultation_details(
            consultation_id="cons_patient_a_0",
            include_structured=True,
        )

        assert details is not None
        assert details["consultation_id"] == "cons_patient_a_0"
        assert "soap_note" in details
        assert "assessment" in details["soap_note"]
        assert "subjective" in details["soap_note"]
        assert "objective" in details["soap_note"]
        assert "plan" in details["soap_note"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_consultation_returns_none(self, temp_vector_store):
        """Test that getting nonexistent consultation returns None."""
        store = temp_vector_store

        details = await store.get_consultation_details(
            consultation_id="nonexistent_id",
        )

        assert details is None


class TestNormalization:
    """Test embedding normalization."""

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized_on_storage(self, temp_vector_store):
        """Test that embeddings are normalized before storage."""
        store = temp_vector_store

        # Create an embedding with known norm
        embedding = np.array([3.0, 4.0] + [0.0] * 126, dtype=np.float32)  # Norm = 5

        vector_id = await store.store_consultation(
            consultation_id="test_norm",
            patient_id_encrypted="patient_test",
            provider_id="dr_test",
            encounter_date="2025-01-15",
            transcript="Test",
            embedding=embedding,
        )

        # Retrieve the stored embedding from FAISS
        stored_embedding = store.index.reconstruct(int(vector_id))

        # Check that it's normalized (norm ≈ 1)
        norm = np.linalg.norm(stored_embedding)
        assert abs(norm - 1.0) < 0.01, f"Expected norm ≈ 1.0, got {norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
