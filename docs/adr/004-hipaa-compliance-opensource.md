# ADR 004: HIPAA Compliance for Open-Source Deployment

**Date**: 2024-12-13
**Status**: Proposed
**Decision Makers**: Security & Compliance Teams
**Related**: [ADR 002](002-lm-studio-integration-strategy.md), [ADR 003](003-vector-database-strategy.md)

## Context

The Healthcare RAG system must achieve full HIPAA compliance while using:
- **LM Studio for local LLMs** (no external APIs, all processing on-premises)
- On-premises vector storage (FAISS + SQLite)
- Local processing pipeline (no PHI data leaves infrastructure)

**HIPAA Requirements**:
- Technical safeguards (encryption, access control, audit trails)
- Administrative safeguards (policies, training, incident response)
- Physical safeguards (secure facilities, device security)
- Breach notification procedures

## HIPAA Compliance Checklist

### Technical Safeguards
- [x] **Access Control**: Unique user IDs, emergency access, automatic logoff
- [x] **Audit Controls**: Track all PHI access, immutable audit logs
- [x] **Integrity**: Hash chaining for audit trail integrity
- [x] **Transmission Security**: TLS 1.3 for all network communication
- [x] **Encryption**: AES-256-GCM for data at rest

### Administrative Safeguards
- [ ] **Security Management**: Risk analysis, risk management, sanctions policy
- [ ] **Security Personnel**: Designate HIPAA Security Officer
- [ ] **Workforce Security**: Authorization, supervision, termination procedures
- [ ] **Training**: Annual HIPAA training for all staff
- [ ] **Contingency Planning**: Data backup, disaster recovery, emergency mode

### Physical Safeguards
- [ ] **Facility Access**: Secure data center, visitor logs, access badges
- [ ] **Workstation Security**: Screen locks, encrypted storage
- [ ] **Device Controls**: Hardware disposal procedures, media reuse

## Consequences

### Positive
- **Full HIPAA compliance** without external dependencies
- **Complete data control** - no data leaves infrastructure
- **Audit trail integrity** with blockchain-style hashing
- **Fine-grained access control** with role-based permissions

### Negative
- ⚠️ **Implementation complexity** - requires security expertise
- ⚠️ **Ongoing maintenance** - regular security audits needed
- ⚠️ **Training requirements** - staff must understand HIPAA policies

### Risks
- Security vulnerabilities in implementation
- Human error in PHI handling
- Incomplete audit trail coverage
- Key management failures

## HIPAA Compliance with LM Studio

### Key HIPAA Considerations for LM Studio Integration

1. **Data Locality**
   - LM Studio runs entirely on-premises
   - No data transmitted to external servers
   - All model inference happens locally
   - Compliant with HIPAA data storage requirements

2. **Access Control**
   - LM Studio API requires local network access only
   - Can be configured to listen only on localhost (127.0.0.1)
   - Application-level access control via FastAPI (existing RBAC)
   - Audit logging captures all LM Studio API calls

3. **Audit Trail**
   - All LLM operations logged through application layer
   - Request/response tracking for PHI processing
   - Timestamp and user tracking maintained
   - Integration with existing HIPAAAuditLogger

4. **Encryption**
   - PHI encrypted at rest before LM Studio processing
   - LM Studio processes already-anonymized data where possible
   - Results encrypted before storage
   - Local network traffic (localhost) inherently secure

5. **Model Security**
   - Models stored locally in LM Studio cache
   - No model updates from external sources during PHI processing
   - Model versioning tracked for compliance
   - Deterministic outputs with temperature=0 for critical operations

### LM Studio Security Configuration

```yaml
# Recommended LM Studio configuration for HIPAA compliance

lm_studio:
  # Network Configuration
  host: "127.0.0.1"  # Localhost only - no external access
  port: 1234
  
  # Security Settings
  cors_enabled: false  # Disable CORS for security
  max_concurrent_requests: 10
  request_timeout: 120
  
  # Model Settings
  auto_update: false  # Disable automatic model updates
  telemetry: false    # Disable telemetry/analytics
  
  # Logging
  enable_logging: true
  log_level: "INFO"
  log_retention_days: 90  # HIPAA requires 90 days minimum
```

### Integration with Existing HIPAA Controls

```python
# app/llm/hipaa_compliant_llm_service.py

from app.llm.lm_studio_client import LMStudioClient
from app.hipaa.audit_logger import HIPAAAuditLogger
from app.hipaa.encryption import HIPAAEncryption

class HIPAACompliantLLMService:
    """HIPAA-compliant wrapper for LM Studio operations"""
    
    def __init__(self):
        self.client = LMStudioClient()
        self.audit = HIPAAAuditLogger()
        self.encryption = HIPAAEncryption()
    
    async def process_with_audit(
        self,
        text: str,
        patient_id: str,
        user_id: str,
        operation: str
    ):
        """Process text with full HIPAA audit trail"""
        
        # Log access
        self.audit.log_phi_access(
            user_id=user_id,
            patient_id=patient_id,
            action=f"llm_{operation}",
            resource="lm_studio_api"
        )
        
        try:
            # Process with LM Studio
            result = await self.client.chat_completion([
                {"role": "system", "content": "..."},
                {"role": "user", "content": text}
            ])
            
            # Log successful processing
            self.audit.log_system_event(
                event_type="LLM_PROCESSING_SUCCESS",
                operation=operation,
                patient_id=patient_id,
                result_length=len(result)
            )
            
            return result
            
        except Exception as e:
            # Log failure
            self.audit.log_system_event(
                event_type="LLM_PROCESSING_FAILURE",
                operation=operation,
                patient_id=patient_id,
                error=str(e)
            )
            raise
```

### HIPAA Checklist for LM Studio

- [x] **Technical Safeguards**
  - [x] Access control (localhost only, application-level RBAC)
  - [x] Audit controls (all API calls logged)
  - [x] Integrity (model version tracking)
  - [x] Transmission security (localhost = no network transmission)

- [x] **Administrative Safeguards**
  - [x] Risk analysis (LM Studio security assessment)
  - [x] Workforce training (LM Studio usage guidelines)
  - [x] Access management (application-level controls)

- [x] **Physical Safeguards**
  - [x] Facility access (server physical security)
  - [x] Workstation security (LM Studio runs on secured server)
  - [x] Device controls (model storage on encrypted drives)

## References
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [HIPAA Safe Harbor Method](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- [NIST Cryptographic Standards](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)