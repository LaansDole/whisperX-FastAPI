"""Medical Q&A Page - RAG Chatbot for Patient Medical Records."""

import streamlit as st
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import get_api_client
from utils import extract_patient_hash, format_workflow_id

st.set_page_config(page_title="Medical Q&A", page_icon="💬", layout="wide")

# Initialize API client
api_client = get_api_client()

# Handle Streamlit cache issue - if medical_chat method doesn't exist, clear cache
if not hasattr(api_client, "medical_chat"):
    st.cache_resource.clear()
    st.warning("🔄 API client updated. Please refresh the page (F5 or Cmd+R).")
    st.stop()

# Page header
st.title("💬 Medical Q&A")
st.markdown("Query past patient consultations using RAG-powered chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_patient" not in st.session_state:
    st.session_state.current_patient = ""


def generate_patient_hash(patient_name: str) -> str:
    """Generate 8-char patient hash from name (same logic as backend)."""
    import hashlib
    import os

    # MUST match app/patients/filename_utils.py and app/config.py HIPAA_SALT default
    salt = os.getenv("HIPAA_SALT", "default_salt_change_in_production")
    combined = f"{patient_name}{salt}"  # name FIRST, salt SECOND (backend order)
    return hashlib.sha256(combined.encode()).hexdigest()[:8]


# Sidebar for patient selection
with st.sidebar:
    st.header("Patient Selection")

    # Method selector
    input_method = st.radio(
        "Select patient by:",
        options=["Patient Name", "Patient Hash", "Recent Uploads"],
        horizontal=True,
        help="Choose how to identify the patient",
    )

    patient_hash = ""

    if input_method == "Patient Name":
        patient_name = st.text_input(
            "Patient Name",
            placeholder="e.g., John Smith",
            help="Enter the full patient name - hash will be auto-generated",
        )
        if patient_name:
            patient_hash = generate_patient_hash(patient_name)
            st.caption(f"Generated hash: `{patient_hash}`")

    elif input_method == "Patient Hash":
        patient_hash = st.text_input(
            "Patient Hash",
            value=st.session_state.current_patient,
            placeholder="e.g., abc123de",
            help="8-character patient identifier hash",
        )

    elif input_method == "Recent Uploads":
        if "recent_uploads" in st.session_state and st.session_state.recent_uploads:
            # Build options from recent uploads
            options = ["Select a workflow..."]
            hash_map = {}
            for upload in st.session_state.recent_uploads[:10]:
                wf_id = upload.get("workflow_id", "")
                extracted_hash = extract_patient_hash(wf_id)
                if extracted_hash:
                    display_name = f"{extracted_hash} ({format_workflow_id(wf_id)})"
                    options.append(display_name)
                    hash_map[display_name] = extracted_hash

            selected = st.selectbox("Select from recent uploads", options)
            if selected != "Select a workflow...":
                patient_hash = hash_map.get(selected, "")
        else:
            st.info("No recent uploads. Upload an audio file first.")

    # Update session state if changed
    if patient_hash and patient_hash != st.session_state.current_patient:
        st.session_state.current_patient = patient_hash
        st.session_state.chat_history = []  # Clear chat history on patient change
        st.rerun()

    st.divider()

    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What medications were prescribed?",
        "Summarize the last consultation",
        "Any abnormal findings?",
        "What was the diagnosis?",
        "What follow-up was recommended?",
    ]

    for question in example_questions:
        if st.button(f"📝 {question}", key=question, use_container_width=True):
            st.session_state.pending_question = question

# Main chat interface
if not patient_hash:
    st.info("👈 Please enter a Patient Hash in the sidebar to start querying")
else:
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📎 {len(message['sources'])} Sources"):
                        for idx, source in enumerate(message["sources"], 1):
                            similarity = source.get("similarity_score", 0) * 100
                            consultation_id = source.get("consultation_id", "Unknown")
                            encounter_date = source.get("encounter_date", "Unknown")
                            provider_id = source.get("provider_id", "Unknown")

                            st.markdown(f"**Source {idx}** (📊 {similarity:.0f}% match)")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.caption(f"📅 {encounter_date}")
                            with col2:
                                st.caption(f"👨‍⚕️ {provider_id}")

                            # Display SOAP note if available
                            soap = source.get("soap_note")
                            if soap and isinstance(soap, dict):
                                with st.expander("📋 SOAP Note"):
                                    for section in ["subjective", "objective", "assessment", "plan"]:
                                        if section in soap:
                                            st.markdown(f"**{section.title()}:** {soap[section][:200]}...")
                            st.divider()

    # Chat input
    if "pending_question" in st.session_state:
        user_query = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_query = st.chat_input("Ask a question about this patient...")

    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query the RAG chatbot
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical records..."):
                try:
                    result = api_client.medical_chat(
                        query=user_query, patient_hash=patient_hash, session_id=st.session_state.get("session_id")
                    )

                    # Store session ID
                    if "session_id" in result:
                        st.session_state.session_id = result["session_id"]

                    # Display answer (backend returns 'response' key)
                    answer = result.get("response", "No answer generated")
                    st.markdown(answer)

                    # Display sources
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander(f"📎 {len(sources)} Sources"):
                            for idx, source in enumerate(sources, 1):
                                similarity = source.get("similarity_score", 0) * 100
                                consultation_id = source.get("consultation_id", "Unknown")
                                encounter_date = source.get("encounter_date", "Unknown")
                                provider_id = source.get("provider_id", "Unknown")

                                st.markdown(f"**Source {idx}** (📊 {similarity:.0f}% match)")
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.caption(f"📅 {encounter_date}")
                                with col2:
                                    st.caption(f"👨‍⚕️ {provider_id}")

                                # Display SOAP note if available
                                soap = source.get("soap_note")
                                if soap and isinstance(soap, dict):
                                    with st.expander("📋 SOAP Note"):
                                        for section in ["subjective", "objective", "assessment", "plan"]:
                                            if section in soap:
                                                st.markdown(f"**{section.title()}:** {soap[section][:200]}...")
                                st.divider()

                    # Add assistant message to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error querying medical records: {error_msg}")

                    # Show helpful messages based on error
                    if "503" in error_msg or "unavailable" in error_msg.lower():
                        st.warning(
                            """
                            **Service Unavailable**

                            This could mean:
                            - LM Studio is not running
                            - Vector storage is disabled
                            - Medical processing is not enabled

                            Please check your service configuration.
                            """
                        )
                    elif "404" in error_msg:
                        st.info(
                            f"""
                            **No Data Found**

                            No consultations found for patient `{patient_hash}`.

                            Make sure to:
                            1. Upload audio with medical processing enabled
                            2. Wait for workflow completion
                            3. Use the correct patient hash
                            """
                        )

                    # Add error to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"❌ Error: {error_msg}", "sources": []}
                    )

        st.rerun()

# Footer with stats
with st.sidebar:
    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Messages", len(st.session_state.chat_history))
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if "session_id" in st.session_state:
            del st.session_state.session_id
        st.rerun()
