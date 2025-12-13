# ADR 003: Vector Database Strategy for Medical RAG

**Date**: 2024-12-13  
**Status**: Proposed  
**Decision Makers**: ML Engineering & Infrastructure Teams  
**Related**: [ADR 002](002-lm-studio-integration-strategy.md)

## Context

The original plan proposes Weaviate as the vector database for medical document storage and retrieval. However, this introduces:
- Additional infrastructure complexity
- Separate service to manage and secure
- Learning curve for team
- Potential single point of failure

**Requirements**:
- Store medical consultation embeddings
- Fast similarity search (<500ms)
- HIPAA-compliant storage
- Support for metadata filtering
- On-premises deployment capability

## Decision

Adopt a **simplified, Python-native vector storage** approach using existing tools:

### Phase 1: FAISS with SQLite (Weeks 1-6)

Use Facebook's FAISS for vector search with SQLite for metadata:

```python
import faiss
import numpy as np
import sqlite3
from typing import List, Dict
from pathlib import Path

class MedicalDocumentVectorStore:
    """Simplified vector store using FAISS + SQLite"""
    
    def __init__(self, storage_dir: str = "./vector_storage", embedding_dim: int = 768):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # FAISS index for embeddings
        # Default to 768 for nomic-embed-text-v1.5 (LM Studio)
        # Use 384 for all-MiniLM-L6-v2 if needed
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # SQLite for metadata
        self.db_path = self.storage_dir / "metadata.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT UNIQUE NOT NULL,
                patient_id_encrypted TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                encounter_date TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_text TEXT NOT NULL,
                standardized_code TEXT,
                confidence REAL,
                FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patient 
            ON consultations(patient_id_encrypted)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_encounter_date 
            ON consultations(encounter_date)
        """)
        
        self.conn.commit()
    
    async def store_consultation(self, 
                                 consultation_id: str,
                                 embedding: np.ndarray,
                                 metadata: Dict) -> int:
        """Store consultation embedding and metadata"""
        
        # Add to FAISS index
        embedding_normalized = embedding / np.linalg.norm(embedding)
        self.index.add(embedding_normalized.reshape(1, -1))
        vector_id = self.index.ntotal - 1
        
        # Store metadata in SQLite
        cursor = self.conn.execute("""
            INSERT INTO consultations 
            (consultation_id, patient_id_encrypted, provider_id, encounter_date, content_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (
            consultation_id,
            metadata["patient_id_encrypted"],
            metadata["provider_id"],
            metadata["encounter_date"],
            metadata["content_hash"]
        ))
        
        # Store medical entities
        for entity in metadata.get("medical_entities", []):
            self.conn.execute("""
                INSERT INTO medical_entities
                (consultation_id, entity_type, entity_text, standardized_code, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                consultation_id,
                entity["entity_type"],
                entity["text"],
                entity.get("standardized_code"),
                entity.get("confidence")
            ))
        
        self.conn.commit()
        return vector_id
    
    async def search_similar(self, 
                            query_embedding: np.ndarray,
                            patient_id: str = None,
                            limit: int = 10) -> List[Dict]:
        """Search for similar consultations"""
        
        # Normalize query
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_normalized.reshape(1, -1), 
            limit * 2  # Get more to filter by patient
        )
        
        # Get metadata from SQLite
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Query metadata
            query = """
                SELECT consultation_id, patient_id_encrypted, provider_id, 
                       encounter_date, content_hash
                FROM consultations
                WHERE rowid = ?
            """
            if patient_id:
                query += " AND patient_id_encrypted = ?"
                cursor = self.conn.execute(query, (int(idx) + 1, patient_id))
            else:
                cursor = self.conn.execute(query, (int(idx) + 1,))
            
            row = cursor.fetchone()
            if row:
                results.append({
                    "consultation_id": row[0],
                    "patient_id_encrypted": row[1],
                    "provider_id": row[2],
                    "encounter_date": row[3],
                    "similarity_score": float(1 / (1 + distance)),  # Convert distance to similarity
                    "vector_id": idx
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def save_index(self):
        """Persist FAISS index to disk"""
        index_path = self.storage_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
    
    def load_index(self):
        """Load FAISS index from disk"""
        index_path = self.storage_dir / "faiss_index.bin"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
```

**Advantages**:
- ✅ No additional service to manage
- ✅ Python-native, easy to integrate
- ✅ Fast performance (FAISS is production-grade)
- ✅ Simple backup (just files)
- ✅ Easy encryption (file-level)
- ✅ Lower operational complexity

### Phase 2: Add Index Optimization (Weeks 7-8)

Optimize FAISS index for better performance:

```python
class OptimizedVectorStore(MedicalDocumentVectorStore):
    """Optimized FAISS with IVF clustering"""
    
    def __init__(self, storage_dir: str = "./vector_storage"):
        super().__init__(storage_dir)
        
        # Use IVF (Inverted File Index) for faster search
        # After collecting enough vectors (e.g., 10K+)
        self.nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
    
    def train_index(self, sample_embeddings: np.ndarray):
        """Train IVF index on sample data"""
        if not self.index.is_trained:
            self.index.train(sample_embeddings)
    
    async def search_similar(self, query_embedding: np.ndarray, 
                            limit: int = 10) -> List[Dict]:
        """Faster search with IVF"""
        # Set number of clusters to probe
        self.index.nprobe = 10  # Trade-off between speed and accuracy
        
        return await super().search_similar(query_embedding, limit)
```

### Phase 3: Optional Weaviate (Post-MVP, if needed)

Only migrate to Weaviate if:
- Volume exceeds 1M consultations
- Need advanced features (hybrid search, GraphQL)
- Require distributed deployment
- Team has Weaviate expertise

```python
class WeaviateMigration:
    """Migration path to Weaviate if needed"""
    
    async def migrate_from_faiss(self, faiss_store: MedicalDocumentVectorStore):
        """Migrate existing FAISS data to Weaviate"""
        # Export from SQLite + FAISS
        # Import to Weaviate
        # Validate consistency
        # Switch traffic gradually
        pass
```

## Comparison: FAISS vs Weaviate

| Feature | FAISS + SQLite | Weaviate |
|---------|---------------|----------|
| Setup Complexity | Low | High |
| Operational Overhead | Minimal | Moderate |
| Search Performance | Excellent | Excellent |
| Scalability | 100K-1M vectors | 10M+ vectors |
| Query Flexibility | Basic | Advanced (GraphQL) |
| HIPAA Compliance | Easy (file encryption) | Moderate (service config) |
| Backup/Restore | Simple (file copy) | Complex (database backup) |
| Learning Curve | Low | Moderate-High |
| Cost | Free | Free (OSS) but infra costs |

## Consequences

### Positive
- ✅ **Faster MVP**: No new service to deploy
- ✅ **Lower complexity**: Fewer moving parts
- ✅ **Easier debugging**: Standard Python tools
- ✅ **Better security**: Fewer network boundaries
- ✅ **Simpler backups**: File-based storage

### Negative
- ⚠️ **Limited scale**: Not ideal for 10M+ vectors
- ⚠️ **Basic queries**: No GraphQL or advanced filtering
- ⚠️ **Single machine**: No distributed deployment initially
- ⚠️ **Manual optimization**: Need to tune FAISS parameters

### Risks
- May need to migrate if scale exceeds expectations
- FAISS in-memory requirements for large datasets
- SQLite concurrent write limitations

## Mitigation Strategies

1. **Scale Monitoring**:
   ```python
   class VectorStoreMonitor:
       def check_scale_thresholds(self):
           if self.index.ntotal > 500_000:
               logger.warning("Approaching scale limit, consider migration")
           
           if self.search_latency_p95 > 500:  # ms
               logger.warning("Search latency degrading, optimize index")
   ```

2. **Incremental Migration**:
   ```python
   # Design for easy migration from day 1
   class VectorStoreInterface(ABC):
       @abstractmethod
       async def store_consultation(self, ...): pass
       
       @abstractmethod
       async def search_similar(self, ...): pass
   
   # Can swap implementations without code changes
   vector_store: VectorStoreInterface = FaissVectorStore()
   # Later: vector_store = WeaviateVectorStore()
   ```

3. **Performance Optimization**:
   ```python
   # Use memory mapping for large indices
   faiss.write_index(index, "large_index.bin")
   index = faiss.read_index("large_index.bin", faiss.IO_FLAG_MMAP)
   
   # Add GPU support if available
   if faiss.get_num_gpus() > 0:
       index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
   ```

## Implementation Plan

### Week 1-2: FAISS Setup
- [ ] Implement basic FAISS + SQLite store
- [ ] Create embedding generation pipeline
- [ ] Add encryption for patient IDs
- [ ] Test with sample consultations

### Week 3-4: Search & Retrieval
- [ ] Implement similarity search
- [ ] Add metadata filtering
- [ ] Create patient-specific queries
- [ ] Benchmark search performance

### Week 5-6: Optimization
- [ ] Add index persistence
- [ ] Implement batch operations
- [ ] Add caching layer
- [ ] Performance tuning

### Week 7-8: Production Readiness
- [ ] Implement backup/restore
- [ ] Add monitoring and alerting
- [ ] Create migration path to Weaviate
- [ ] Load testing

## Alternatives Considered

1. **Weaviate from start**: Too complex for MVP
2. **Pinecone/Milvus**: Requires external services, not fully open-source
3. **ChromaDB**: Good option, but FAISS more established
4. **PostgreSQL pgvector**: Possible, but separate service still needed
5. **Pure SQLite embeddings**: Too slow for similarity search

## Notes on LM Studio Integration

### Embedding Generation with LM Studio

LM Studio supports embedding models through its API. Use the `/embeddings` endpoint:

```python
# Generate embeddings via LM Studio
async def generate_embedding_lm_studio(text: str) -> np.ndarray:
    """Generate embeddings using LM Studio's embedding endpoint"""
    from app.llm.lm_studio_client import LMStudioClient
    
    client = LMStudioClient()
    embedding = await client.generate_embedding(text)
    return np.array(embedding, dtype=np.float32)

# Update FAISS initialization for LM Studio embeddings
# nomic-embed-text-v1.5: 768 dimensions
# all-MiniLM-L6-v2: 384 dimensions
vector_store = MedicalDocumentVectorStore(
    storage_dir="./vector_storage",
    embedding_dim=768  # for nomic-embed-text-v1.5
)
```

### Recommended Embedding Models in LM Studio

1. **nomic-embed-text-v1.5** (768-dim)
   - Best for medical text similarity
   - High quality semantic search
   - Available in LM Studio model browser

2. **all-MiniLM-L6-v2** (384-dim)
   - Lightweight and fast
   - Good for general embeddings
   - Lower memory footprint

## References
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FAISS Best Practices](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Vector Database Comparison](https://www.bentoml.com/blog/benchmarking-vector-databases-a-comprehensive-guide)