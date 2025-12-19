"""PHI minimization and anonymization for HIPAA Safe Harbor compliance."""

import re
import hashlib
import logging
from typing import List, Dict, Pattern
from datetime import datetime

from .access_control import HealthcareRole

logger = logging.getLogger(__name__)


class PHIMinimizer:
    """Minimize and anonymize PHI according to HIPAA Safe Harbor standards."""

    # HIPAA Safe Harbor identifiers that must be removed
    HIPAA_IDENTIFIERS = {
        "name": "Names",
        "address": "Geographic subdivisions smaller than state",
        "date": "All dates (except year)",
        "phone": "Telephone numbers",
        "fax": "Fax numbers",
        "email": "Email addresses",
        "ssn": "Social Security numbers",
        "mrn": "Medical Record Numbers",
        "account_number": "Account numbers",
        "certificate_number": "Certificate/license numbers",
        "vehicle_id": "Vehicle identifiers",
        "device_id": "Device identifiers",
        "url": "Web URLs",
        "ip_address": "IP addresses",
        "biometric_id": "Biometric identifiers",
        "photo": "Full face photographic images",
        "other_id": "Other unique identifying numbers",
    }

    def __init__(self):
        """Initialize PHI minimizer with regex patterns."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for PHI detection."""
        self.patterns: Dict[str, Pattern] = {
            # Dates (various formats)
            "date": re.compile(
                r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
                r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
                r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4})\b",
                re.IGNORECASE,
            ),
            # Phone numbers (US format)
            "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b"),
            # Social Security Numbers
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            # Email addresses
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            # Medical Record Numbers
            "mrn": re.compile(
                r"\b(?:MRN|Patient\s+ID|Medical\s+Record\s+Number)?\s*[:#]?\s*([A-Z]{0,2}\d{6,10})\b", re.IGNORECASE
            ),
            # URLs
            "url": re.compile(r'\bhttps?://[^\s<>"]+|www\.[^\s<>"]+\b'),
            # IP addresses
            "ip_address": re.compile(
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
                r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
            ),
            # Address patterns (simplified)
            "address": re.compile(
                r"\b\d+\s+([A-Z][a-z]*\s*)+(?:Street|St|Avenue|Ave|Road|Rd|"
                r"Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b",
                re.IGNORECASE,
            ),
            # License/certificate numbers
            "certificate_number": re.compile(
                r"\b(?:License|Cert|Certificate)\s*#?\s*([A-Z]{1,3}\d{4,8})\b", re.IGNORECASE
            ),
            # Vehicle identifiers
            "vehicle_id": re.compile(
                r"\b[A-Z0-9]{17}\b"  # VIN format
            ),
            # Account numbers
            "account_number": re.compile(r"\bAccount\s*#?\s*(\d{6,12})\b", re.IGNORECASE),
        }

        # Name patterns (more complex)
        self.name_patterns = [
            # Doctor/Provider titles
            re.compile(r"\b(?:Dr|Doctor|MD|DO|RN|LPN|PA|NP|CRNP)\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", re.IGNORECASE),
            # Patient name patterns
            re.compile(r"\b(?:Patient|Name|Pt)\s*[:\-]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", re.IGNORECASE),
            # Common name patterns (capitalize first letter)
            re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?=\s+(?:is|was|has|presents|reports|complains))"),
        ]

    def anonymize_for_research(self, text: str, keep_year_only: bool = True, age_threshold: int = 89) -> str:
        """
        Anonymize text for research use (HIPAA Safe Harbor).

        Args:
            text: Text to anonymize
            keep_year_only: If True, keep only year from dates
            age_threshold: Ages above this are removed (HIPAA requirement)

        Returns:
            Anonymized text
        """
        anonymized = text

        # Remove/replace dates
        if keep_year_only:
            # Extract year from dates and keep only that
            def date_replacer(match):
                date_str = match.group()
                # Extract 4-digit year
                year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
                if year_match:
                    return f"[DATE: {year_match.group()}]"
                else:
                    return "[DATE]"

            anonymized = self.patterns["date"].sub(date_replacer, anonymized)
        else:
            anonymized = self.patterns["date"].sub("[DATE]", anonymized)

        # Remove phone numbers
        anonymized = self.patterns["phone"].sub("[PHONE]", anonymized)

        # Remove SSNs
        anonymized = self.patterns["ssn"].sub("[SSN]", anonymized)

        # Remove emails
        anonymized = self.patterns["email"].sub("[EMAIL]", anonymized)

        # Remove MRNs
        anonymized = self.patterns["mrn"].sub("[MRN]", anonymized)

        # Remove URLs
        anonymized = self.patterns["url"].sub("[URL]", anonymized)

        # Remove IP addresses
        anonymized = self.patterns["ip_address"].sub("[IP_ADDRESS]", anonymized)

        # Remove addresses
        anonymized = self.patterns["address"].sub("[ADDRESS]", anonymized)

        # Remove certificate numbers
        anonymized = self.patterns["certificate_number"].sub("[CERTIFICATE]", anonymized)

        # Remove vehicle IDs
        anonymized = self.patterns["vehicle_id"].sub("[VEHICLE_ID]", anonymized)

        # Remove account numbers
        anonymized = self.patterns["account_number"].sub("[ACCOUNT]", anonymized)

        # Remove names
        anonymized = self._anonymize_names(anonymized)

        # Remove ages above threshold
        anonymized = self._remove_high_ages(anonymized, age_threshold)

        return anonymized

    def _anonymize_names(self, text: str) -> str:
        """Anonymize names using pattern matching."""
        anonymized = text

        # Apply name patterns
        for pattern in self.name_patterns:

            def name_replacer(match):
                return "[NAME]"

            anonymized = pattern.sub(name_replacer, anonymized)

        return anonymized

    def _remove_high_ages(self, text: str, threshold: int) -> str:
        """Remove ages above HIPAA threshold (89)."""
        # Pattern for ages
        age_pattern = re.compile(
            r"\b(?:age|aged?)(?:\s+is|:)?\s*(\d{1,3})\s*(?:years?|yrs?|yo|year-old|year\s+old)?\b", re.IGNORECASE
        )

        def age_replacer(match):
            age_str = match.group(1)
            try:
                age = int(age_str)
                if age > threshold:
                    return "[AGE_RESTRICTED]"
                elif age < 0:
                    return "[AGE_INVALID]"
                else:
                    return match.group()  # Keep valid ages
            except ValueError:
                return match.group()

        return age_pattern.sub(age_replacer, text)

    def detect_phi(self, text: str) -> Dict[str, List[Dict[str, any]]]:
        """
        Detect PHI entities in text.

        Args:
            text: Text to scan for PHI

        Returns:
            Dictionary with PHI entities by type
        """
        phi_entities = {}

        for phi_type, pattern in self.patterns.items():
            matches = []
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "type": phi_type,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8,  # Base confidence for regex matches
                    }
                )

            if matches:
                phi_entities[phi_type] = matches

        # Add name detection
        name_matches = []
        for pattern in self.name_patterns:
            for match in pattern.finditer(text):
                name_matches.append(
                    {
                        "type": "name",
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.6,  # Lower confidence for name patterns
                    }
                )

        if name_matches:
            phi_entities["name"] = name_matches

        return phi_entities

    def check_minimum_necessary(
        self, requested_fields: List[str], user_role: HealthcareRole, purpose: str
    ) -> List[str]:
        """
        Filter fields to minimum necessary for purpose.

        Args:
            requested_fields: List of requested field names
            user_role: User's healthcare role
            purpose: Purpose of access (treatment, research, billing, etc.)

        Returns:
            Filtered list of allowed fields
        """
        # Define minimum necessary fields by role and purpose
        minimum_fields_map = {
            (HealthcareRole.PHYSICIAN, "treatment"): [
                "patient_id",
                "name",
                "dob",
                "medical_history",
                "current_medications",
                "allergies",
                "vital_signs",
                "diagnosis",
                "treatment_plan",
                "progress_notes",
            ],
            (HealthcareRole.NURSE, "treatment"): [
                "patient_id",
                "name",
                "dob",
                "current_medications",
                "allergies",
                "vital_signs",
                "care_instructions",
                "mar_status",
            ],
            (HealthcareRole.RESEARCHER, "research"): [
                "anonymized_id",
                "age_range",
                "diagnosis",
                "treatment_outcome",
                "procedures",
                "medications",
                "comorbidities",
            ],
            (HealthcareRole.ADMINISTRATOR, "billing"): [
                "patient_id",
                "name",
                "insurance_info",
                "services_rendered",
                "charges",
                "billing_codes",
            ],
            (HealthcareRole.AUDITOR, "auditing"): [
                "patient_id",
                "access_log",
                "modification_history",
                "audit_trail",
                "compliance_status",
            ],
        }

        minimum_fields = minimum_fields_map.get((user_role, purpose), [])

        # Add role-specific base fields
        role_base_fields = self._get_role_base_fields(user_role)
        minimum_fields.extend(role_base_fields)

        # Remove duplicates and filter requested fields
        allowed_fields = []
        seen_fields = set()

        for field in requested_fields:
            if field in minimum_fields and field not in seen_fields:
                allowed_fields.append(field)
                seen_fields.add(field)

        return allowed_fields

    def _get_role_base_fields(self, role: HealthcareRole) -> List[str]:
        """Get base fields that role always needs."""
        base_fields = {
            HealthcareRole.PHYSICIAN: ["consultation_id", "timestamp", "patient_status"],
            HealthcareRole.NURSE: ["consultation_id", "timestamp", "patient_status"],
            HealthcareRole.RESEARCHER: ["anonymized_id", "timestamp", "research_consent"],
            HealthcareRole.ADMINISTRATOR: ["consultation_id", "timestamp", "billing_status"],
            HealthcareRole.AUDITOR: ["consultation_id", "timestamp", "audit_required"],
        }

        return base_fields.get(role, ["consultation_id", "timestamp"])

    def deidentify_dataset(
        self,
        records: List[Dict[str, any]],
        patient_id_field: str = "patient_id",
        name_fields: List[str] = None,
        date_fields: List[str] = None,
        keep_reference_ids: bool = False,
    ) -> List[Dict[str, any]]:
        """
        De-identify a dataset of records.

        Args:
            records: List of record dictionaries
            patient_id_field: Field containing patient ID
            name_fields: List of fields containing names
            date_fields: List of fields containing dates
            keep_reference_ids: If True, keep anonymized reference IDs

        Returns:
            De-identified records
        """
        if name_fields is None:
            name_fields = ["patient_name", "provider_name", "guardian_name"]
        if date_fields is None:
            date_fields = ["dob", "admission_date", "discharge_date", "visit_date"]

        deidentified_records = []
        patient_id_map = {}  # Map real patient_id to anonymized_id

        for record in records:
            deidentified_record = record.copy()

            # Anonymize patient ID
            if patient_id_field in record:
                real_patient_id = record[patient_id_field]
                if real_patient_id not in patient_id_map:
                    # Generate anonymized ID
                    anonymized_id = self._generate_anonymous_id(real_patient_id)
                    patient_id_map[real_patient_id] = anonymized_id

                deidentified_record[patient_id_field] = patient_id_map[real_patient_id]

            # Remove name fields
            for field in name_fields:
                if field in deidentified_record:
                    del deidentified_record[field]

            # Anonymize date fields (keep year only)
            for field in date_fields:
                if field in deidentified_record and deidentified_record[field]:
                    deidentified_record[field] = self._anonymize_date_year_only(deidentified_record[field])

            # Remove other PHI fields
            phi_fields_to_remove = ["ssn", "mrn", "phone", "email", "address", "ip_address"]
            for field in phi_fields_to_remove:
                if field in deidentified_record:
                    del deidentified_record[field]

            # Add de-identification metadata
            deidentified_record["deidentified"] = True
            deidentified_record["deidentification_date"] = datetime.now().isoformat()

            deidentified_records.append(deidentified_record)

        return deidentified_records

    def _generate_anonymous_id(self, real_id: str) -> str:
        """Generate anonymous ID from real ID."""
        # Use SHA-256 hash of real ID with salt
        salt = "hipaa_anonymization_salt_change_in_production"
        combined = f"{real_id}{salt}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        return f"ANON_{hash_value[:16].upper()}"

    def _anonymize_date_year_only(self, date_value: str) -> str:
        """Extract year from date string."""
        try:
            # Try to parse as datetime and extract year
            date_obj = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            return date_obj.year
        except (ValueError, AttributeError):
            # Fallback to regex extraction
            year_match = re.search(r"\b(19|20)\d{2}\b", str(date_value))
            if year_match:
                return year_match.group()
            else:
                return "[YEAR]"

    def assess_phi_risk(self, text: str) -> Dict[str, any]:
        """
        Assess PHI risk level of text.

        Args:
            text: Text to assess

        Returns:
            Risk assessment dictionary
        """
        phi_entities = self.detect_phi(text)

        # Count PHI types
        phi_types_found = list(phi_entities.keys())
        phi_count = sum(len(matches) for matches in phi_entities.values())

        # Assess risk level
        high_risk_types = ["ssn", "name", "address", "email", "phone"]
        medium_risk_types = ["date", "mrn", "account_number"]
        low_risk_types = ["url", "ip_address"]

        high_risk_count = sum(len(phi_entities.get(t, [])) for t in high_risk_types if t in phi_entities)
        medium_risk_count = sum(len(phi_entities.get(t, [])) for t in medium_risk_types if t in phi_entities)
        low_risk_count = sum(len(phi_entities.get(t, [])) for t in low_risk_types if t in phi_entities)

        # Determine risk level
        if high_risk_count >= 3:
            risk_level = "HIGH"
        elif high_risk_count >= 1 or medium_risk_count >= 3:
            risk_level = "MEDIUM"
        elif phi_count >= 2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        return {
            "risk_level": risk_level,
            "phi_types_found": phi_types_found,
            "total_phi_count": phi_count,
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "recommendations": self._get_risk_recommendations(risk_level, phi_types_found),
        }

    def _get_risk_recommendations(self, risk_level: str, phi_types: List[str]) -> List[str]:
        """Get recommendations based on risk level."""
        recommendations = []

        if risk_level == "HIGH":
            recommendations.extend(
                [
                    "Immediate de-identification required",
                    "Remove all high-risk identifiers",
                    "Consider full anonymization",
                    "Review data handling procedures",
                ]
            )
        elif risk_level == "MEDIUM":
            recommendations.extend(
                [
                    "Partial de-identification required",
                    "Remove medium-risk identifiers",
                    "Apply minimum necessary principle",
                    "Implement access controls",
                ]
            )
        elif risk_level == "LOW":
            recommendations.extend(
                ["Apply PHI minimization", "Consider anonymization for research", "Document data use purpose"]
            )
        else:
            recommendations.append("Low PHI risk, but maintain security practices")

        # Add specific recommendations based on PHI types
        if "ssn" in phi_types:
            recommendations.append("Remove or encrypt SSN immediately")
        if "name" in phi_types:
            recommendations.append("Replace names with codes or remove")
        if "address" in phi_types:
            recommendations.append("Remove detailed geographic information")

        return recommendations
