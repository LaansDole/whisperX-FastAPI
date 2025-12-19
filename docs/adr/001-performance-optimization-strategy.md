# ADR 001: Performance Optimization Strategy for Healthcare RAG

**Date**: 2024-12-13  
**Status**: Proposed   

## Decision

We will implement a **phased, optimized approach** with parallel processing and optional features:

### Phase 1: MVP with Parallel Processing (Target: 8-10 min)
1. **Parallel execution** of independent steps:
   ```python
   # Steps 1-4: Existing pipeline (~50 min max, typically 5-10 min)
   # After transcription completes:
   phi_task, entity_task = await asyncio.gather(
       phi_detection_activity(),
       preliminary_entity_extraction()  # Lightweight initial pass
   )
   
   # Document structuring uses results from both
   structured = await document_structuring_activity(phi_task, entity_task)
   
   # Vector storage and FHIR conversion in background (don't block response)
   asyncio.create_task(vector_storage_activity(structured))
   asyncio.create_task(fhir_conversion_activity(structured))
   ```

2. **Feature flags** for gradual rollout:
   ```python
   MEDICAL_RAG_ENABLED = true/false
   VECTOR_DB_ENABLED = true/false  # Optional initially
   FHIR_CONVERSION_ENABLED = true/false  # Optional initially
   REAL_TIME_ENTITY_EXTRACTION = true/false  # vs batch processing
   ```

3. **LLM-based SOAP generation** via LM Studio (local):
   - Use LM Studio's OpenAI-compatible API with medical models (MedAlpaca/Meditron)
   - No external API costs or data privacy concerns
   - Faster than training/loading custom models manually
   - Easier to iterate and improve with prompt engineering
   - Switch models via LM Studio UI without code changes

### Phase 2: Performance Optimization (Target: 5 min)
- Cache medical terminology lookups
- Batch entity extraction for multiple consultations
- Optimize vector embeddings with quantization
- Implement streaming responses for real-time feedback

### Phase 3: Advanced Optimization (Target: 3 min)
- Custom fine-tuned medical NLP models
- GPU acceleration for entity extraction
- Edge deployment for low-latency processing

## Consequences

### Positive
- Meets performance targets with realistic timelines
- Reduces initial infrastructure complexity
- Allows validation before heavy investment
- Graceful degradation if components fail
- Lower upfront costs (LLM APIs vs custom models)

### Negative
- LM Studio service dependency (mitigated by health checks and fallbacks)
- Model loading time on startup (2-5 minutes for 7B models)
- Requires careful orchestration of parallel tasks

### Risks
- LM Studio server downtime (mitigated by graceful degradation)
- Increased complexity in workflow orchestration
- Potential race conditions in parallel processing
- GPU memory constraints with large models

## Alternatives Considered

1. **Sequential processing with faster models**: Would still exceed 5-minute target
2. **Skip medical processing entirely**: Defeats purpose of healthcare integration
3. **Pre-load all models**: Memory intensive, slow startup
4. **Microservices for each step**: Over-engineered for initial deployment

## Implementation Notes

- Use `asyncio.gather()` for true parallel execution
- Implement timeouts for each parallel task
- Add circuit breakers for external API calls
- Monitor and log processing times per step
- Create performance benchmarks before/after optimization