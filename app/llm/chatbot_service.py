"""RAG-powered chatbot service for medical consultations.

This service provides conversational access to patient medical records
using vector similarity search and LLM-based response generation.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from .lm_studio_client import LMStudioClient, LMStudioConfig
from ..config import Config

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a single chat message."""

    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        self.role = role  # "user", "assistant", or "system"
        self.content = content
        self.timestamp = timestamp or datetime.now(Config.TIMEZONE).isoformat()

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    """Manages conversation history for a chat session."""

    def __init__(self, max_messages: int = 20):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append(ChatMessage(role, content))
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            # Always keep system message if present
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1) :]
            else:
                self.messages = self.messages[-self.max_messages :]

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM API."""
        return [msg.to_dict() for msg in self.messages]

    def clear(self):
        """Clear conversation history."""
        self.messages = []


class MedicalChatbotService:
    """RAG-powered chatbot for querying patient medical data."""

    SYSTEM_PROMPT = """You are a medical assistant helping healthcare providers access and understand patient records.

Your role:
1. Answer questions about patient medical history, consultations, and treatments
2. Summarize clinical information clearly and accurately
3. Highlight important medical details like diagnoses, medications, and allergies
4. Flag any concerning patterns or contraindications

Guidelines:
- Use professional medical terminology appropriately
- Be concise but comprehensive
- Always cite the source consultation when referencing specific information
- If information is not available in the context, clearly state this
- Never make up or assume medical information
- **Privacy & PHI**: Some records have Protected Health Information (PHI) like names, dates, or addresses protected/redacted. If a user asks for identifying details that are not present, explain that they have been protected to maintain patient privacy.

The following context contains relevant patient consultation records that you should use to answer questions."""

    def __init__(self, client: Optional[LMStudioClient] = None):
        """Initialize chatbot service.

        Args:
            client: LM Studio client. If None, creates a new client with config from env.
        """
        if client:
            self.client = client
        else:
            config = LMStudioConfig(
                base_url=Config.LM_STUDIO_BASE_URL,
                timeout=Config.LM_STUDIO_TIMEOUT,
                temperature=0.3,  # Slightly higher for conversational responses
                max_tokens=Config.LM_STUDIO_MAX_TOKENS,
                model=Config.LM_STUDIO_MODEL,
            )
            self.client = LMStudioClient(config)

        # Store conversation histories by session ID
        self.conversations: Dict[str, ConversationHistory] = {}

    def _get_or_create_conversation(self, session_id: str) -> ConversationHistory:
        """Get or create a conversation history for a session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationHistory()
        return self.conversations[session_id]

    def _format_context_from_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM."""
        if not search_results:
            return "No relevant patient records found."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get("metadata", {}) or {}
            has_phi = metadata.get("has_phi", False)
            phi_status = "[PHI Protected]" if has_phi else "[No PHI Detected]"

            context_parts.append(f"""
--- Consultation {i} ---
Date: {result.get("encounter_date", "Unknown")}
Similarity Score: {result.get("similarity_score", 0):.2f}
Consultation ID: {result.get("consultation_id", "Unknown")}
Privacy Status: {phi_status}
""")
            # Add structured document info if available
            if result.get("structured_document"):
                doc = result["structured_document"]
                if doc.get("clinical_summary"):
                    context_parts.append(f"Clinical Summary: {doc['clinical_summary']}")

            # Add SOAP note if available
            if result.get("soap_note"):
                soap = result["soap_note"]
                if soap.get("assessment"):
                    context_parts.append(f"Assessment: {soap['assessment']}")
                if soap.get("plan"):
                    context_parts.append(f"Plan: {soap['plan']}")

            # Add medical entities if available
            if result.get("medical_entities"):
                entities = result["medical_entities"]
                diagnoses = [e["text"] for e in entities if e.get("type") == "diagnosis"]
                medications = [e["text"] for e in entities if e.get("type") == "medication"]

                if diagnoses:
                    context_parts.append(f"Diagnoses: {', '.join(diagnoses)}")
                if medications:
                    context_parts.append(f"Medications: {', '.join(medications)}")

        return "\n".join(context_parts)

    async def query(
        self,
        user_query: str,
        patient_id_encrypted: str,
        session_id: Optional[str] = None,
        additional_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query with RAG context.

        Args:
            user_query: The user's question
            patient_id_encrypted: Encrypted patient ID for context retrieval
            session_id: Optional session ID for conversation continuity
            additional_context: Pre-fetched search results (optional)

        Returns:
            Dict with response, sources, and metadata
        """
        session_id = session_id or f"session_{datetime.now(Config.TIMEZONE).timestamp()}"
        conversation = self._get_or_create_conversation(session_id)

        try:
            # Fetch patient context if not provided
            if additional_context is None:
                additional_context = await self._fetch_patient_context(user_query, patient_id_encrypted)

            # Format context
            context_text = self._format_context_from_results(additional_context)

            # Build prompt with context
            system_message = f"{self.SYSTEM_PROMPT}\n\n--- PATIENT RECORDS ---\n{context_text}"

            # Initialize conversation with system prompt if empty
            if not conversation.messages:
                conversation.add_message("system", system_message)
            else:
                # Update system message with latest context
                if conversation.messages[0].role == "system":
                    conversation.messages[0].content = system_message

            # Add user query
            conversation.add_message("user", user_query)

            # Get response from LLM
            messages = conversation.get_messages_for_api()
            response_text = await self.client.chat_completion(messages=messages, temperature=0.3, max_tokens=1024)

            # Add assistant response to conversation
            conversation.add_message("assistant", response_text)

            # Build response
            return {
                "response": response_text,
                "session_id": session_id,
                "sources": [
                    {
                        "consultation_id": r.get("consultation_id"),
                        "encounter_date": r.get("encounter_date"),
                        "similarity_score": r.get("similarity_score"),
                        "provider_id": r.get("provider_id"),
                        "soap_note": r.get("soap_note"),
                        "has_phi": r.get("metadata", {}).get("has_phi", False),
                    }
                    for r in additional_context
                ],
                "context_used": len(additional_context) > 0,
                "timestamp": datetime.now(Config.TIMEZONE).isoformat(),
            }

        except Exception as e:
            logger.error(f"Chatbot query failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "session_id": session_id,
                "sources": [],
                "context_used": False,
                "error": str(e),
                "timestamp": datetime.now(Config.TIMEZONE).isoformat(),
            }

    async def _fetch_patient_context(
        self, query: str, patient_id_encrypted: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch relevant patient context using vector similarity search."""
        from ..vector_store.medical_vector_store import MedicalDocumentVectorStore

        if not Config.ENABLE_VECTOR_STORAGE:
            logger.warning("Vector storage not enabled, returning empty context")
            return []

        try:
            # Generate embedding for query
            query_embedding = await self.client.generate_embedding(query, Config.EMBEDDING_MODEL)

            # Initialize vector store
            vector_store = MedicalDocumentVectorStore(
                storage_dir=Config.VECTOR_DB_PATH, embedding_dim=Config.EMBEDDING_DIMENSION
            )

            try:
                # Search for similar consultations
                results = await vector_store.search_similar(
                    query_embedding=np.array(query_embedding, dtype=np.float32),
                    patient_id_encrypted=patient_id_encrypted,
                    limit=limit,
                    similarity_threshold=0.3,  # Lower threshold for more context
                )

                # Enrich results with full details
                enriched_results = []
                for result in results:
                    details = await vector_store.get_consultation_details(
                        consultation_id=result["consultation_id"],
                        include_entities=True,
                        include_phi=False,  # Don't include PHI in context
                        include_structured=True,
                    )
                    if details:
                        enriched_results.append({**result, **details})
                    else:
                        enriched_results.append(result)

                return enriched_results
            finally:
                vector_store.close()

        except Exception as e:
            logger.error(f"Failed to fetch patient context: {e}")
            return []

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]

    async def close(self):
        """Close the underlying LM Studio client."""
        await self.client.close()
