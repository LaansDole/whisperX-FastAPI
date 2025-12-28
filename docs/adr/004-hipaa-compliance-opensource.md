# ADR 004: HIPAA Compliance for Open-Source Deployment

**Date**: 2024-12-13
**Status**: Proposed
**Decision Makers**: Security & Compliance Teams
**Related**: [ADR 002](002-lm-studio-integration-strategy.md), [ADR 003](003-vector-database-strategy.md)

## Context

The Healthcare RAG system must achieve full HIPAA compliance while using:
- LM Studio for local LLMs (no external APIs, all processing on-premises)
- On-premises vector storage (FAISS + SQLite)
- Local processing pipeline (no PHI data leaves infrastructure)

**HIPAA Requirements**:
- Technical safeguards (encryption, access control, audit trails)
- Administrative safeguards (policies, training, incident response)
- Physical safeguards (secure facilities, device security)
- Breach notification procedures

## Decision

Implement comprehensive HIPAA technical safeguards with the existing local architecture, ensuring all PHI processing remains on-premises with proper access controls and audit trails.

## HIPAA Compliance Status

### Technical Safeguards
- **Access Control**: Unique user IDs, emergency access, automatic logoff
- **Audit Controls**: Track all PHI access, immutable audit logs
- **Integrity**: Hash chaining for audit trail integrity
- **Transmission Security**: TLS 1.3 for all network communication
- **Encryption**: AES-256-GCM for data at rest

## HIPAA Compliance with LM Studio

### Key Considerations

1. **Data Locality**
   - LM Studio runs entirely on-premises
   - No data transmitted to external servers
   - All model inference happens locally
   - Compliant with HIPAA data storage requirements

2. **Access Control**
   - LM Studio API configured for localhost-only access (127.0.0.1)
   - Application-level access control via FastAPI RBAC
   - Audit logging captures all LM Studio API calls

3. **Audit Trail**
   - All LLM operations logged through application layer
   - Request/response tracking for PHI processing
   - Timestamp and user tracking maintained
   - Integration with HIPAAAuditLogger

4. **Encryption**
   - PHI encrypted at rest before LM Studio processing
   - LM Studio processes anonymized data where possible
   - Results encrypted before storage
   - Local network traffic (localhost) inherently secure

5. **Model Security**
   - Models stored locally in LM Studio cache
   - No model updates from external sources during PHI processing
   - Model versioning tracked for compliance
   - Deterministic outputs with temperature=0 for critical operations

### Recommended LM Studio Security Configuration

```yaml
lm_studio:
  host: "127.0.0.1"  # Localhost only - no external access
  port: 1234
  cors_enabled: false  # Disable CORS for security
  auto_update: false  # Disable automatic model updates
  telemetry: false    # Disable telemetry/analytics
  log_retention_days: 90  # HIPAA requires 90 days minimum
```

## Consequences

### Positive
- **Full HIPAA compliance** without external dependencies
- **Complete data control** - no data leaves infrastructure
- **Audit trail integrity** with immutable logging
- **Fine-grained access control** with role-based permissions

### Negative
- **Implementation complexity** - requires security expertise
- **Ongoing maintenance** - regular security audits needed
- **Training requirements** - staff must understand HIPAA policies

### Risks
- Security vulnerabilities in implementation
- Human error in PHI handling
- Incomplete audit trail coverage
- Key management failures

## References
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [HIPAA Safe Harbor Method](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- [NIST Cryptographic Standards](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)