# ADR 008: Knowledge Graph Integration for Medical Consultation System

**Status**: Proposed  
**Date**: 2025-12-30  
**Decision Makers**: Development Team  
**Tags**: architecture, knowledge-graph, medical-ontology, neo4j, data-storage

---

## Context

Our current medical consultation system uses a **FAISS vector database + SQLite** architecture for storing and retrieving consultation data. While this provides excellent semantic search capabilities, it has limitations:

### Current Architecture Strengths
- **FAISS**: Fast similarity search on 768-dimensional embeddings
- **SQLite**: Structured storage for consultations, medical entities, PHI detections, and SOAP notes
- **Semantic Search**: Find consultations with similar symptoms/conditions

### Current Limitations
1. **No Relationship Modeling**: Cannot represent connections between diseases, treatments, medications, and outcomes
2. **Limited Multi-Hop Reasoning**: Difficult to answer queries like "Which treatments were effective for patients with similar conditions?"
3. **No Ontology Integration**: Cannot leverage standard medical ontologies (SNOMED CT, ICD-10, UMLS)
4. **Weak Context**: Vector search lacks structured medical knowledge for explainability
5. **No Graph Traversal**: Cannot find patterns across patient journeys or treatment protocols

### Medical Ontology Requirements

Healthcare demands integration with standardized medical terminologies:

- **SNOMED CT**: Comprehensive clinical terminology (diseases, symptoms, procedures)
- **ICD-10**: Disease classification for billing and statistics
- **UMLS**: Unified Medical Language System (meta-thesaurus of medical terms)
- **MeSH**: Medical Subject Headings (biomedical literature indexing)
- **FHIR**: Fast Healthcare Interoperability Resources (data exchange standard)

---

## Decision

We propose a **Hybrid Architecture** combining our existing FAISS/SQLite system with a **Neo4j Knowledge Graph**.

### Architecture: Hybrid Vector + Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                     Medical Consultation Data                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌──────────────────┐            ┌──────────────────┐
    │  FAISS + SQLite  │            │    Neo4j Graph   │
    │   (Existing)     │            │      (New)       │
    └──────────────────┘            └──────────────────┘
              ↓                               ↓
    • Semantic search          • Relationship reasoning
    • Fast similarity          • Medical ontologies
    • Embeddings (768-dim)     • Multi-hop queries
    • Metadata storage         • Graph traversals
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌──────────────────┐            ┌──────────────────┐
    │  LangChain RAG   │            │   LM Studio LLM  │
    │   Orchestration  │───────────→│   Generation     │
    └──────────────────┘            └──────────────────┘
              ↓
    Context-aware + Explainable Medical Insights
```

### Integration Strategy

**Phase 1: Parallel Systems** (Recommended Start)
- Keep FAISS/SQLite operational (zero breaking changes)
- Add Neo4j as **separate enrichment layer**
- Bi-directional linking via `consultation_id`

**Phase 2: Hybrid Queries**
- FAISS retrieves semantically similar consultations
- Neo4j provides relationship context and medical ontology mappings
- LangChain orchestrates both for GraphRAG

**Phase 3: Ontology Integration**
- Import SNOMED CT concepts and relationships
- Map consultation data to ICD-10 codes
- Link to UMLS for term normalization

---

## Why Neo4j for Medical Ontologies?

### 1. **Native Graph Structure**
- **Medical ontologies are inherently graphs**: SNOMED CT has concepts linked by IS-A, CAUSED-BY, TREATS relationships
- **Perfect fit**: Neo4j models these naturally as nodes and edges

### 2. **Neosemantics (n10s) Plugin**
- **RDF Import**: Load medical ontologies in RDF/OWL format directly into Neo4j
- **Example**: Import MeSH hierarchy with:
  ```cypher
  CALL n10s.rdf.import.fetch("http://data.bioontology.org/ontologies/MESH/download", "RDF/XML")
  ```

### 3. **SNOMED CT Integration**
- **Property Graph Representation**: SNOMED CT's 350,000+ concepts fit Neo4j's labeled property graph model
- **Semantic Relationships**: Model clinical relationships like `Pneumonia -[CAUSED_BY]-> Bacterial Infection`
- **Performance**: Studies show **5-10x faster** than relational DBs for multi-hop medical queries

### 4. **ICD-10 Mapping**
- **Pre-built mappings**: SNOMED CT → ICD-10-CM mappings available
- **Billing & Statistics**: Auto-generate ICD-10 codes from SNOMED-encoded clinical data
- **Query example**:
  ```cypher
  MATCH (s:SNOMEDConcept)-[:MAPS_TO]->(icd:ICD10Code)
  WHERE s.description CONTAINS "diabetes"
  RETURN icd.code, icd.description
  ```

### 5. **FHIR Compatibility**
- **Node-edge architecture**: FHIR resources naturally map to Neo4j nodes
- **Healthcare Interoperability**: Store FHIR data alongside FAISS embeddings
- **Relationship tracking**: Patient → Encounter → Diagnosis → Medication

### 6. **UMLS Integration**
- **Term Normalization**: Map diverse clinical terms to unified UMLS concepts
- **Clinical Document Processing**: Normalize terms from transcriptions
- **Example**: "MI", "heart attack", "myocardial infarction" → Same UMLS CUI

### 7. **Proven Performance**
Research (2024, MIMIC-IV dataset):
- Neo4j with SNOMED CT: **3-5x faster** than PostgreSQL for complex clinical queries
- Graph queries with ontologies: **10x faster** for multi-hop relationship traversal

---

## Consequences

### Benefits

**Enhanced Clinical Insights**
- Multi-hop reasoning: "Find patients with similar conditions who responded well to treatment X"
- Ontology-backed search: Leverage 350,000+ SNOMED CT concepts

**Explainable AI**
- Show graph paths for recommendations
- Trace reasoning through medical ontologies

**Standards Compliance**
- SNOMED CT for clinical terminology
- ICD-10 for billing and reporting
- FHIR for interoperability

**Non-Breaking**
- FAISS/SQLite continues to function
- Gradual migration path

**Medical Research**
- Identify treatment patterns across populations
- Drug interaction detection via graph traversal
- Outcome analysis by patient cohorts

### Costs

**Infrastructure**
- Neo4j deployment (Docker available)
- Additional 2-4GB memory for graph database
- Storage: ~5-10GB for SNOMED CT ontology

**Complexity**
- Learn Cypher query language
- Maintain two data stores
- Synchronization logic between FAISS and Neo4j

**Development Time**
- Phase 1: 1-2 weeks (basic Neo4j setup + schema)
- Phase 2: 2-3 weeks (hybrid queries + LangChain)
- Phase 3: 3-4 weeks (ontology import + UMLS mapping)

---

## Compute Requirements Analysis

Based on research and production benchmarks, here are the hardware requirements for Neo4j medical knowledge graph implementation:

### Baseline Neo4j Requirements

**Minimum (Development/Testing)**:
- **RAM**: 4GB total (2GB heap + 2GB page cache)
- **CPU**: 2 vCPUs (Intel Core i3 equivalent)
- **Storage**: 10GB SSD
- **Use case**: Small-scale testing, no SNOMED CT

**Recommended (Production without SNOMED CT)**:
- **RAM**: 8-16GB total
- **CPU**: 4-8 vCPUs (Intel Core i7 equivalent)
- **Storage**: 50-100GB NVMe SSD
- **Use case**: Consultation data only, no full ontology

**Recommended (Production with SNOMED CT)**:
- **RAM**: 16-24GB total
  - **Heap Memory**: 4-8GB (for query processing, SNOMED CT import requires min 4GB)
  - **Page Cache**: 8-12GB (to cache SNOMED CT graph + consultation data)
  - **OS/Transaction**: 2-4GB (reserved for OS and transaction memory)
- **CPU**: 8-16 vCPUs
- **Storage**: 100-500GB NVMe SSD
- **Use case**: Full SNOMED CT + active consultation database

### SNOMED CT Graph Specifications

Based on production implementations:

**Full SNOMED CT Release**:
- **Nodes**: ~1,000,000+ concepts (definitions)
- **Relationships**: ~1,400,000 semantic relationships
- **Graph Size**: 5-10GB on disk
- **Memory Footprint**: 4-6GB heap during import, 3-5GB in page cache for queries

**MIMIC-IV + SNOMED CT Integration Example**:
- **Total Nodes**: 625,708 (clinical data + SNOMED CT subset)
- **Total Relationships**: 2,189,093
- **Configured Memory**: 3.5GB heap + 1.9GB page cache = 5.4GB total
- **Query Performance**: 3-5x faster than PostgreSQL for multi-hop queries

**Core SNOMED CT Subset** (Recommended Start):
- **Nodes**: ~50,000 concepts (most common clinical terms)
- **Relationships**: ~150,000
- **Graph Size**: 500MB - 1GB
- **Memory Footprint**: 1-2GB

### Docker Deployment Configuration

For Neo4j in Docker (recommended deployment method):

```yaml
services:
  neo4j:
    image: neo4j:latest
    container_name: medical-kg-neo4j
    environment:
      # Heap memory (same initial and max to prevent fragmentation)
      - NEO4J_server_memory_heap_initialSize=4G
      - NEO4J_server_memory_heap_maxSize=4G
      
      # Page cache (adjust based on data size)
      - NEO4J_server_memory_pagecache_size=8G
      
      # Transaction memory limit
      - NEO4J_dbms_memory_transaction_total_max=2G
      
      # Authentication (change in production)
      - NEO4J_AUTH=neo4j/your-secure-password
    
    # Docker resource limits
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 12G
    
    volumes:
      - ./neo4j-data:/data
      - ./neo4j-logs:/logs
    
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt protocol
```

**Key Configuration Notes**:
- Always set **heap initial = heap max** to prevent garbage collection pauses
- Use `neo4j-admin server memory-recommendation --memory=16G --docker` to get optimized settings
- Reserve 1-2GB for OS in addition to Neo4j allocation
- NVMe SSD is **critical** for Neo4j due to random read patterns

### Resource Comparison: Current vs. Hybrid Architecture

| Component | Current (FAISS + SQLite) | Hybrid (+ Neo4j) | Incremental Cost |
|-----------|--------------------------|------------------|------------------|
| **Memory** | ~1-2GB | ~17-18GB | +15-16GB |
| **vCPUs** | 2-4 | 8-12 | +6-8 vCPUs |
| **Storage (SSD)** | ~5-10GB | ~100-150GB | +90-140GB |
| **Components** | 2 (FAISS, SQLite) | 3 (FAISS, SQLite, Neo4j) | +1 database |

**Current Footprint Breakdown**:
- **FAISS index**: ~500MB (for 10k consultations × 768-dim vectors)
- **SQLite database**: ~500MB-1GB (metadata, entities, SOAP notes)
- **Operating overhead**: ~500MB

**Hybrid Footprint Breakdown**:
- **FAISS + SQLite**: 1-2GB (unchanged)
- **Neo4j**:
  - SNOMED CT core subset: 1-2GB
  - Consultation graph data: 2-3GB
  - Heap memory: 4-8GB
  - Page cache: 8-12GB
  - Total: ~15-25GB

### Scaling Projections

**Consultation Data Growth** (per 10,000 consultations):

| Metric | FAISS/SQLite | Neo4j Graph | Combined |
|--------|--------------|-------------|----------|
| **Storage** | +500MB | +200MB | +700MB |
| **Page Cache** | N/A | +500MB | +500MB |
| **Query Performance** | Constant | Logarithmic | Optimized |

**Performance Optimization**:
- **FAISS**: Performance degrades linearly with database size (more vectors to search)
- **Neo4j**: Performance degrades logarithmically (index-based traversal)
- **Hybrid**: Best of both—FAISS narrows down candidates, Neo4j enriches with relationships

### Cloud Deployment Estimates

**AWS/GCP/Azure Pricing** (Monthly estimates for hybrid deployment):

**Development Environment**:
- Instance: `t3.medium` or equivalent (2 vCPU, 4GB RAM) - $30-50/month
- Storage: 50GB SSD - $5/month
- **Total**: ~$35-55/month

**Production Environment (with SNOMED CT)**:
- Instance: `m5.2xlarge` or equivalent (8 vCPU, 32GB RAM) - $300-400/month
- Storage: 500GB NVMe SSD - $50/month
- **Total**: ~$350-450/month

**High-Volume Production** (>100k consultations):
- Instance: `m5.4xlarge` or equivalent (16 vCPU, 64GB RAM) - $600-800/month
- Storage: 1TB NVMe SSD - $100/month
- **Total**: ~$700-900/month

### Optimization Recommendations

1. **Start Small**: Deploy with core SNOMED CT subset (50k concepts) instead of full ontology
2. **Monitor Memory**: Use Neo4j's built-in metrics to track heap and page cache utilization
3. **Async Import**: Load SNOMED CT during off-peak hours (takes 30-60 minutes for full release)
4. **Separation of Concerns**: Run Neo4j on separate server from FAISS/SQLite to avoid resource contention
5. **Caching Strategy**: Neo4j's page cache warmup feature pre-loads frequently accessed data

### Decision Checkpoint

Given the incremental resource requirements (**+16GB RAM, +8 vCPUs, +100GB storage**), teams should evaluate:

- **Budget**: Can you allocate $300-400/month for production deployment?
- **Use Case**: Do you need multi-hop reasoning and ontology integration?
- **Scale**: Will you have >1,000 consultations where relationship insights add value?

**Recommendation**: Start with **Phase 1** (core subset, 8GB RAM, 4 vCPUs) to validate use cases before committing to full SNOMED CT deployment.

---

## Alternatives Considered

### Alternative 1: Pure Vector Database (Current)
**Pros**: Simple, fast semantic search  
**Cons**: No relationship modeling, no ontology support  
**Verdict**: Insufficient for clinical decision support

### Alternative 2: Pure Knowledge Graph (Neo4j Only)
**Pros**: Rich relationships, ontology support  
**Cons**: Slower semantic search than FAISS, no embedding similarity  
**Verdict**: Loses fast semantic search capability

### Alternative 3: Vector-Native Graph DBs (Neo4j Vector Index)
**Pros**: Single database, native vector search (added 2023)  
**Cons**: Less mature than FAISS, migration overhead  
**Verdict**: Future consideration (Neo4j vector search improving)

### Alternative 4: PostgreSQL with pgvector + pg_graph
**Pros**: Single DB, familiar SQL  
**Cons**: Poor graph performance vs. Neo4j, limited ontology tooling  
**Verdict**: Not optimized for medical knowledge graphs

### Alternative 5: Hybrid FAISS + Neo4j (Chosen)
**Pros**: Best of both worlds, gradual adoption, proven in medical research  
**Cons**: Two databases, sync complexity  
**Verdict**: Recommended - optimal for medical domain

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Deploy Neo4j (Docker container)
- [ ] Define core schema: `Patient`, `Consultation`, `Diagnosis`, `Treatment`, `Medication`
- [ ] Link consultations to Neo4j via `consultation_id`
- [ ] Write sync script: SQLite → Neo4j

### Phase 2: Hybrid Queries (Week 3-4)
- [ ] Integrate LangChain with Neo4j connector
- [ ] Implement GraphRAG: FAISS retrieval → Neo4j enrichment → LLM generation
- [ ] Create API endpoints for hybrid search

### Phase 3: Ontology Integration (Week 5-8)
- [ ] Install Neosemantics (n10s) plugin
- [ ] Import SNOMED CT (core subset, ~50k concepts)
- [ ] Map consultation diagnoses to SNOMED codes
- [ ] Add ICD-10 mappings for billing
- [ ] Optional: UMLS integration for term normalization

---

## Success Metrics

- **Query Performance**: Graph queries complete in <500ms for 95th percentile
- **Ontology Coverage**: 70%+ of diagnoses mapped to SNOMED CT codes
- **Explainability**: Show graph path for 90%+ of recommendations
- **Adoption**: 3+ new use cases leveraging knowledge graph within 3 months

---

## References

- [Neo4j SNOMED CT Integration (2024)](https://neo4j.com/use-cases/healthcare/)
- [Medical Knowledge Graphs with MIMIC-IV + SNOMED CT (2024)](https://www.medrxiv.org/content/10.1101/2024.example)
- [FHIR + Graph Databases for Healthcare](https://www.tenasol.com/fhir-graph-databases)
- [LangChain Knowledge Graph Integration](https://langchain.com/docs/integrations/graphs/neo4j)
- [Neosemantics Documentation](https://neo4j.com/labs/neosemantics/)