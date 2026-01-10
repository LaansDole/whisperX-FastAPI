"""Medical LLM service for processing healthcare consultation transcripts."""

import json
import re
import logging
from typing import Dict, List, Any
from datetime import datetime

from .lm_studio_client import LMStudioClient
from ..config import Config

logger = logging.getLogger(__name__)


class MedicalLLMService:
    """High-level medical LLM operations using LM Studio."""

    def __init__(self, client: LMStudioClient = None):
        """Initialize medical LLM service.

        Args:
            client: LM Studio client. If None, creates a new client with default config.
        """
        self.client = client or LMStudioClient()

    async def detect_phi(self, text: str) -> Dict[str, Any]:
        """
        Detect Protected Health Information in text.

        Args:
            text: Medical consultation transcript

        Returns:
            Dict with PHI entities and their locations
        """
        system_prompt = """You are a HIPAA-compliant PHI detection system.
Identify and extract all Protected Health Information including:
- Names, addresses, dates
- Phone numbers, emails, SSNs
- Medical Record Numbers (MRNs)
- Account numbers, identifiers
- Ages over 89 (HIPAA special protection)

Return results as JSON with this exact format:
{
  "phi_detected": true/false,
  "entities": [
    {
      "type": "name|address|date|phone|email|ssn|mrn|identifier|age",
      "text": "the actual PHI text",
      "start": character_start_position,
      "end": character_end_position,
      "confidence": 0.0-1.0
    }
  ]
}

Be precise with character positions. Set confidence based on certainty."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Detect PHI in this medical text:\n\n{text}"},
            ]

            response = await self.client.chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic for PHI detection
                max_tokens=2048,
            )

            return self._parse_phi_response(response)

        except Exception as e:
            logger.error(f"PHI detection failed: {e}")
            return {"phi_detected": False, "entities": [], "error": str(e)}

    async def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from consultation transcript.

        Args:
            text: Medical consultation transcript

        Returns:
            List of entities with type, text, confidence, and normalized terms
        """
        system_prompt = """You are a medical entity extraction system.
Extract and categorize medical information with ICD-10 codes where possible:

Entity types:
- diagnosis: Medical conditions/diagnoses
- medication: Medications with dosage/frequency
- procedure: Medical procedures/tests
- symptom: Patient-reported symptoms
- vital_sign: Vitals (BP, HR, temperature, etc.)
- allergy: Allergies and reactions
- immunization: Vaccines and immunizations
- lab_result: Laboratory test results

Return JSON format:
{
  "entities": [
    {
      "type": "entity_type",
      "text": "original text",
      "normalized": "standardized term",
      "code": "ICD-10/CPT/SNOMED code if available",
      "confidence": 0.0-1.0,
      "details": "additional context"
    }
  ]
}

Use standard medical terminology and coding systems."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract medical entities:\n\n{text}"},
            ]

            response = await self.client.chat_completion(messages=messages)
            return self._parse_entities_response(response)

        except Exception as e:
            logger.error(f"Medical entity extraction failed: {e}")
            return []

    async def generate_soap_note(self, transcript: str) -> Dict[str, str]:
        """
        Generate SOAP note from consultation transcript.

        Args:
            transcript: Medical consultation transcript

        Returns:
            Dict with Subjective, Objective, Assessment, Plan sections
        """
        system_prompt = """You are a medical documentation specialist.
Generate a comprehensive SOAP note from the consultation transcript.

Format:
**Subjective**: Patient's reported symptoms, history, chief complaint
**Objective**: Clinical observations, exam findings, vital signs, test results
**Assessment**: Diagnosis, clinical impression, differential diagnoses
**Plan**: Treatment plan, medications, follow-up, referrals, patient education

Guidelines:
- Use professional medical terminology appropriately
- Be thorough but concise
- Include relevant positive and negative findings
- Ensure all sections are complete and clinically relevant
- Maintain chronological flow
- Use proper medical abbreviations judiciously"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate SOAP note for:\n\n{transcript}"},
            ]

            response = await self.client.chat_completion(messages=messages, max_tokens=3072)

            return self._parse_soap_response(response)

        except Exception as e:
            logger.error(f"SOAP note generation failed: {e}")
            return {"subjective": "", "objective": "", "assessment": "", "plan": "", "error": str(e)}

    async def detect_phi_in_dialogue(self, dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect Protected Health Information in speaker-attributed dialogue.

        Args:
            dialogue_data: Structured dialogue with speaker roles from TranscriptionTransformer

        Returns:
            Dict with PHI entities and their locations, including speaker attribution
        """
        system_prompt = """You are a HIPAA-compliant PHI detection system.
Identify and extract all Protected Health Information from this doctor-patient dialogue including:
- Names, addresses, dates
- Phone numbers, emails, SSNs
- Medical Record Numbers (MRNs)
- Account numbers, identifiers
- Ages over 89 (HIPAA special protection)

The dialogue is formatted with speaker roles (Doctor/Patient).
For each PHI entity detected, note which speaker mentioned it.

Return results as JSON with this exact format:
{
  "phi_detected": true/false,
  "entities": [
    {
      "type": "name|address|date|phone|email|ssn|mrn|identifier|age",
      "text": "the actual PHI text",
      "speaker_role": "doctor|patient|unknown",
      "start": character_start_position,
      "end": character_end_position,
      "confidence": 0.0-1.0
    }
  ]
}

Be precise with character positions. Set confidence based on certainty."""

        try:
            # Format dialogue for LLM
            formatted_dialogue = self._format_dialogue_for_prompt(dialogue_data)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Detect PHI in this medical dialogue:\n\n{formatted_dialogue}"},
            ]

            response = await self.client.chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic for PHI detection
                max_tokens=2048,
            )

            return self._parse_phi_response(response)

        except Exception as e:
            logger.error(f"PHI detection in dialogue failed: {e}")
            return {"phi_detected": False, "entities": [], "error": str(e)}

    async def extract_entities_with_speaker(self, dialogue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract medical entities from speaker-attributed dialogue.

        Args:
            dialogue_data: Structured dialogue with speaker roles from TranscriptionTransformer

        Returns:
            List of entities with type, text, confidence, normalized terms, and speaker attribution
        """
        system_prompt = """You are a medical entity extraction system.
Extract and categorize medical information from this doctor-patient dialogue with ICD-10 codes where possible.

Entity types:
- diagnosis: Medical conditions/diagnoses (typically from Doctor)
- medication: Medications with dosage/frequency (typically from Doctor)
- procedure: Medical procedures/tests (typically from Doctor)
- symptom: Patient-reported symptoms (typically from Patient)
- vital_sign: Vitals (BP, HR, temperature, etc.) (typically from Doctor)
- allergy: Allergies and reactions (can be from either)
- immunization: Vaccines and immunizations (typically from Doctor)
- lab_result: Laboratory test results (typically from Doctor)

For each entity, note which speaker (Doctor/Patient) mentioned it.

Return JSON format:
{
  "entities": [
    {
      "type": "entity_type",
      "text": "original text",
      "speaker_role": "doctor|patient|unknown",
      "normalized": "standardized term",
      "code": "ICD-10/CPT/SNOMED code if available",
      "confidence": 0.0-1.0,
      "details": "additional context"
    }
  ]
}

Use standard medical terminology and coding systems."""

        try:
            # Format dialogue for LLM
            formatted_dialogue = self._format_dialogue_for_prompt(dialogue_data)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract medical entities:\n\n{formatted_dialogue}"},
            ]

            response = await self.client.chat_completion(messages=messages)
            return self._parse_entities_response(response)

        except Exception as e:
            logger.error(f"Medical entity extraction from dialogue failed: {e}")
            return []

    async def generate_soap_from_dialogue(self, dialogue_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate SOAP note from speaker-attributed dialogue.

        Args:
            dialogue_data: Structured dialogue with speaker roles from TranscriptionTransformer

        Returns:
            Dict with Subjective, Objective, Assessment, Plan sections
        """
        system_prompt = """You are a medical documentation specialist.
Generate a comprehensive SOAP note from this doctor-patient dialogue.

The dialogue is formatted with speaker roles:
- Doctor: Healthcare provider's statements, questions, observations, and recommendations
- Patient: Patient's responses, symptoms, and reported history

Use this attribution to properly categorize information:
**Subjective**: Patient's reported symptoms, history, chief complaint (from Patient lines)
**Objective**: Doctor's clinical observations, exam findings, vital signs, test results (from Doctor lines)
**Assessment**: Doctor's diagnosis, clinical impression, differential diagnoses (from Doctor lines)
**Plan**: Doctor's treatment plan, medications, follow-up, referrals, patient education (from Doctor lines)

Guidelines:
- Leverage speaker attribution to accurately separate subjective (patient-reported) from objective (doctor-observed) findings
- Use professional medical terminology appropriately
- Be thorough but concise
- Include relevant positive and negative findings
- Ensure all sections are complete and clinically relevant
- Maintain chronological flow in each section
- Use proper medical abbreviations judiciously"""

        try:
            # Format dialogue for LLM
            formatted_dialogue = self._format_dialogue_for_prompt(dialogue_data)

            # Get speaker summary for context
            speaker_mapping = dialogue_data.get("speaker_mapping", {})

            context = f"""Speaker-Attributed Medical Dialogue:

{formatted_dialogue}

Speaker Summary:
"""
            for speaker_id, info in speaker_mapping.items():
                role = info.get("role", "unknown")
                confidence = info.get("confidence", 0)
                context += f"- {speaker_id}: {role.title()} (confidence: {confidence:.2f})\n"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate SOAP note:\n\n{context}"},
            ]

            response = await self.client.chat_completion(messages=messages, max_tokens=3072)

            return self._parse_soap_response(response)

        except Exception as e:
            logger.error(f"SOAP note generation from dialogue failed: {e}")
            return {"subjective": "", "objective": "", "assessment": "", "plan": "", "error": str(e)}

    async def process_speaker_dialogue(self, dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process speaker-attributed dialogue through full medical pipeline.

        This is the main entry point for processing WhisperX-transformed dialogue.
        It orchestrates PHI detection, entity extraction, and SOAP note generation
        with speaker context awareness.

        Args:
            dialogue_data: Structured dialogue from TranscriptionTransformer containing:
                - dialogue: List of speaker-attributed segments
                - speaker_mapping: Role assignments
                - statistics: Speaking time and counts
                - metadata: Consultation metadata

        Returns:
            Dict containing:
                - phi_detection: PHI entities with speaker attribution
                - entities: Medical entities with speaker context
                - soap_note: SOAP note leveraging speaker roles
                - speaker_mapping: Original speaker role assignments
                - statistics: Dialogue statistics
        """
        try:
            logger.info("Processing speaker-attributed dialogue through medical pipeline")

            results = {
                "has_speaker_attribution": True,
                "speaker_mapping": dialogue_data.get("speaker_mapping", {}),
                "statistics": dialogue_data.get("statistics", {}),
                "metadata": dialogue_data.get("consultation_metadata", {}),
            }

            # Step 1: PHI Detection with speaker context
            try:
                phi_result = await self.detect_phi_in_dialogue(dialogue_data)
                results["phi_detection"] = phi_result
                logger.info(f"PHI detection: {phi_result.get('phi_detected', False)}")
            except Exception as e:
                logger.error(f"PHI detection failed: {e}")
                results["phi_detection"] = {"error": str(e), "phi_detected": False, "entities": []}

            # Step 2: Entity extraction with speaker attribution
            try:
                entities = await self.extract_entities_with_speaker(dialogue_data)
                results["entities"] = entities
                logger.info(f"Extracted {len(entities)} medical entities")
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
                results["entities"] = []

            # Step 3: SOAP note generation from dialogue
            try:
                soap_note = await self.generate_soap_from_dialogue(dialogue_data)
                results["soap_note"] = soap_note
                logger.info("SOAP note generated successfully")
            except Exception as e:
                logger.error(f"SOAP generation failed: {e}")
                results["soap_note"] = {"error": str(e)}

            logger.info("Speaker-attributed dialogue processing complete")
            return results

        except Exception as e:
            logger.error(f"Speaker dialogue processing failed: {e}")
            raise

    def _format_dialogue_for_prompt(self, dialogue_data: Dict[str, Any]) -> str:
        """
        Format speaker-attributed dialogue for LLM prompt consumption.

        Args:
            dialogue_data: Structured dialogue with speaker roles

        Returns:
            Formatted dialogue string optimized for LLM processing
        """
        dialogue_segments = dialogue_data.get("dialogue", [])

        if not dialogue_segments:
            return "No dialogue available"

        lines = []
        for segment in dialogue_segments:
            role = segment.get("speaker_role", "unknown").title()
            text = segment.get("text", "").strip()

            if text:
                lines.append(f"{role}: {text}")

        return "\n".join(lines)

    async def structure_medical_document(
        self, transcript: str, phi_data: Dict = None, entities: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create structured medical document from consultation components.

        Args:
            transcript: Original consultation transcript
            phi_data: Detected PHI information
            entities: Extracted medical entities

        Returns:
            Structured document with all clinical sections
        """
        context = f"""
Original Transcript:
{transcript}

{self._format_entities_summary(entities) if entities else "No medical entities identified."}

{self._format_phi_summary(phi_data) if phi_data else "No PHI detected."}
"""

        system_prompt = """Structure this medical consultation into a comprehensive clinical document.

Include these sections:
1. Consultation Metadata (date, duration, type)
2. Chief Complaint
3. History of Present Illness
4. Past Medical History
5. Medications
6. Allergies
7. Review of Systems
8. Physical Examination
9. Assessment/Diagnosis
10. Treatment Plan
11. Follow-up Instructions
12. Patient Education

Return structured JSON:
{
  "consultation_id": "generated_id",
  "timestamp": "ISO_timestamp",
  "metadata": {...},
  "sections": {
    "chief_complaint": {...},
    "history_present_illness": {...},
    "past_medical_history": {...},
    "medications": [...],
    "allergies": [...],
    "review_systems": {...},
    "physical_exam": {...},
    "assessment": [...],
    "treatment_plan": [...],
    "follow_up": {...},
    "patient_education": {...}
  },
  "summary": "brief clinical summary"
}

Extract all clinically relevant information and organize it properly."""

        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]

            response = await self.client.chat_completion(messages=messages, max_tokens=4096)

            return self._parse_structured_document(response)

        except Exception as e:
            logger.error(f"Document structuring failed: {e}")
            return {
                "consultation_id": f"failed_{datetime.now(Config.TIMEZONE).isoformat()}",
                "timestamp": datetime.now(Config.TIMEZONE).isoformat(),
                "error": str(e),
                "sections": {},
                "summary": "Document structuring failed",
            }

    async def generate_clinical_summary(self, structured_document: Dict[str, Any]) -> str:
        """
        Generate a concise clinical summary from structured document.

        Args:
            structured_document: Structured medical document

        Returns:
            Concise clinical summary
        """
        try:
            # Extract key information from structured document
            chief_complaint = structured_document.get("sections", {}).get("chief_complaint", {})
            assessment = structured_document.get("sections", {}).get("assessment", [])
            plan = structured_document.get("sections", {}).get("treatment_plan", [])

            context = f"""
Chief Complaint: {chief_complaint.get("text", "Not specified")}

Assessment: {json.dumps(assessment, indent=2)}

Treatment Plan: {json.dumps(plan, indent=2)}
"""

            messages = [
                {
                    "role": "system",
                    "content": """Generate a concise clinical summary (2-3 paragraphs) for medical records.
Focus on:
1. Patient presentation and chief complaint
2. Key findings and assessment
3. Treatment plan and follow-up

Use professional medical language and be clinically precise.""",
                },
                {"role": "user", "content": f"Generate clinical summary:\n\n{context}"},
            ]

            return await self.client.chat_completion(messages=messages, max_tokens=1024)

        except Exception as e:
            logger.error(f"Clinical summary generation failed: {e}")
            return "Clinical summary generation failed"

    def _parse_phi_response(self, response: str) -> Dict[str, Any]:
        """Parse PHI detection response."""
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            extracted = self._extract_json_from_markdown(response)
            if not extracted:
                logger.error(
                    f"Failed to parse PHI response as JSON or extract JSON from markdown. Response: {response!r}"
                )
                return {
                    "error": "Failed to parse PHI response as JSON or extract JSON from markdown.",
                    "details": str(e),
                    "raw_response": response,
                }
            return extracted

    def _parse_entities_response(self, response: str) -> List[Dict]:
        """Parse medical entity extraction response."""
        try:
            data = json.loads(response)
            entities = data if isinstance(data, list) else data.get("entities", [])
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            # Try markdown extraction
            extracted = self._extract_json_from_markdown(response)
            if isinstance(extracted, dict):
                return extracted.get("entities", [])
            logger.warning(
                "Failed to parse entities from LLM response. Returning empty list. Response was: %r", response
            )
            return []

    def _parse_soap_response(self, response: str) -> Dict[str, str]:
        """Parse SOAP note sections."""
        sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}

        # Parse sections using both markdown and plain text patterns
        current_section = None
        lines = response.split("\n")

        for line in lines:
            line_stripped = line.strip()

            # Pattern 1: Standalone section headers (on their own line)
            if re.match(r"^(?:\*\*|#+\s*)?subjective(?:\*\*)?:?\s*$", line_stripped, re.IGNORECASE):
                current_section = "subjective"
                continue
            elif re.match(r"^(?:\*\*|#+\s*)?objective(?:\*\*)?:?\s*$", line_stripped, re.IGNORECASE):
                current_section = "objective"
                continue
            elif re.match(r"^(?:\*\*|#+\s*)?assessment(?:\*\*)?:?\s*$", line_stripped, re.IGNORECASE):
                current_section = "assessment"
                continue
            elif re.match(r"^(?:\*\*|#+\s*)?plan(?:\*\*)?:?\s*$", line_stripped, re.IGNORECASE):
                current_section = "plan"
                continue

            # Pattern 2: Inline headers with content on the same line (e.g., **Subjective**: content)
            inline_match = re.match(
                r"^(?:\*\*|#+\s*)?(subjective|objective|assessment|plan)(?:\*\*)?:\s*(.+)$",
                line_stripped,
                re.IGNORECASE,
            )
            if inline_match:
                section_name = inline_match.group(1).lower()
                content = inline_match.group(2).strip()
                current_section = section_name
                if content:
                    sections[current_section] = content
                continue

            # Add content to current section
            if current_section and line_stripped:
                # Clean up markdown formatting
                clean_line = re.sub(r"^\*\*([^*]+)\*\*:\s*", r"\1: ", line_stripped)
                clean_line = re.sub(r"^#+\s*", "", clean_line)

                if sections[current_section]:
                    sections[current_section] += "\n" + clean_line
                else:
                    sections[current_section] = clean_line

        return sections

    def _parse_structured_document(self, response: str) -> Dict[str, Any]:
        """Parse structured medical document response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            extracted = self._extract_json_from_markdown(response)
            if extracted:
                return extracted

            # Fallback: create basic structure
            return {
                "consultation_id": f"parsed_{datetime.now(Config.TIMEZONE).isoformat()}",
                "timestamp": datetime.now(Config.TIMEZONE).isoformat(),
                "raw_response": response,
                "sections": {},
                "summary": "Document parsing required manual intervention",
            }

    def _format_entities_summary(self, entities: List[Dict]) -> str:
        """Format entities for prompt context."""
        if not entities:
            return "No medical entities identified."

        entity_groups = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity.get("text", ""))

        summary_parts = []
        for entity_type, texts in entity_groups.items():
            summary_parts.append(f"{entity_type.title()}: {', '.join(texts)}")

        return "Identified Medical Entities:\n" + "\n".join(summary_parts)

    def _format_phi_summary(self, phi_data: Dict) -> str:
        """Format PHI data for prompt context."""
        if not phi_data.get("phi_detected", False):
            return "No PHI detected in the transcript."

        entities = phi_data.get("entities", [])
        if not entities:
            return "PHI detection returned no specific entities."

        phi_types = {}
        for entity in entities:
            phi_type = entity.get("type", "unknown")
            if phi_type not in phi_types:
                phi_types[phi_type] = []
            phi_types[phi_type].append(entity.get("text", ""))

        summary_parts = []
        for phi_type, texts in phi_types.items():
            summary_parts.append(f"{phi_type.title()}: {len(texts)} instance(s)")

        return "PHI Detected:\n" + "\n".join(summary_parts)

    def _extract_json_from_markdown(self, text: str) -> Dict:
        """Extract JSON from markdown code blocks or plain text."""
        # Pattern 1: Match content inside triple backticks (with or without 'json' label)
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, text, re.IGNORECASE)
        for match in matches:
            candidate = match.strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # Pattern 2: Find any JSON object or array in plain text (non-greedy)
        json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Pattern 3: Last resort - find any {...} or [...] and try to parse
        simple_pattern = r"(\{.*?\}|\[.*?\])"
        matches = re.findall(simple_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return {}

    async def close(self):
        """Close the underlying LM Studio client."""
        await self.client.close()
