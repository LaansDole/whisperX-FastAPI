# ADR 003: Vector Database Strategy for Medical RAG

- **Date**: 2024-12-13  
- **Status**: Proposed  
- **Decision Makers**: ML Engineering & Infrastructure Teams  
- **Related**: [ADR 002](002-lm-studio-integration-strategy.md)

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

Adopt a **simplified, Python-native vector storage** approach using FAISS for vector search with SQLite for metadata.

### Technical Approach

**FAISS + SQLite Architecture**:
- FAISS (Facebook AI Similarity Search) for efficient vector similarity search
- SQLite for consultation metadata and medical entities
- File-based storage for simple backups and encryption
- Embedding dimension: 768 (for nomic-embed-text-v1.5) or 384 (for all-MiniLM-L6-v2)

**Key Capabilities**:
- Store consultation embeddings with encrypted patient IDs
- Fast similarity search with metadata filtering
- Support for patient-specific queries
- Index persistence and optimization

### Migration Path

Start with FAISS + SQLite for MVP. Consider migrating to Weaviate only if:
- Volume exceeds 1M consultations
- Need advanced features (hybrid search, GraphQL)
- Require distributed deployment
- Team has Weaviate expertise

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
- **Faster MVP**: No new service to deploy
- **Lower complexity**: Fewer moving parts
- **Easier debugging**: Standard Python tools
- **Better security**: Fewer network boundaries
- **Simpler backups**: File-based storage

### Negative
- **Limited scale**: Not ideal for 10M+ vectors
- **Basic queries**: No GraphQL or advanced filtering
- **Single machine**: No distributed deployment initially
- **Manual optimization**: Need to tune FAISS parameters

### Risks
- May need to migrate if scale exceeds expectations
- FAISS in-memory requirements for large datasets
- SQLite concurrent write limitations

## Mitigation Strategies

### Scale Monitoring
Alert when approaching scale limits (>500K consultations) or search latency degradation (>500ms p95).

### Incremental Migration
Abstract vector store interface to allow swapping implementations without code changes.

### Performance Optimization
- Memory mapping for large indices
- GPU support if available
- IVF (Inverted File Index) clustering for datasets over 10K vectors

## Alternatives Considered

1. **Weaviate from start**: Too complex for MVP
2. **Pinecone/Milvus**: Requires external services, not fully open-source
3. **ChromaDB**: Good option, but FAISS more established
4. **PostgreSQL pgvector**: Separate service still needed
5. **Pure SQLite embeddings**: Too slow for similarity search

## Integration with LM Studio

LM Studio provides embedding generation through its `/embeddings` endpoint, supporting:
- **nomic-embed-text-v1.5** (768-dim) - Best for medical text similarity
- **all-MiniLM-L6-v2** (384-dim) - Lightweight and fast

## References
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FAISS Best Practices](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Vector Database Comparison](https://www.bentoml.com/blog/benchmarking-vector-databases-a-comprehensive-guide)