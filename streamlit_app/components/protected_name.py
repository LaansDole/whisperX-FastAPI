"""
Protected Patient Name Component

Displays patient names with privacy protection via click-to-reveal mechanism.
Complies with HIPAA requirements by hiding PHI by default.
"""

import streamlit as st


def render_protected_name(patient_hash: str, patient_name: str = None, inline: bool = True):
    """
    Display patient name with click-to-reveal protection.

    Args:
        patient_hash: 8-character patient hash (always visible)
        patient_name: Plain text patient name (PHI - protected)
        inline: If True, display inline. If False, display as separate rows.

    Behavior:
        - Default: Name hidden, showing only hash
        - User clicks "Show Name" button
        - Name revealed inline permanently (until page refresh)
        - State stored in st.session_state
    """
    # Session state key for this patient's reveal status
    reveal_key = f"reveal_{patient_hash}"

    # Initialize state if not exists
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    # Check if name is revealed
    if st.session_state[reveal_key] and patient_name:
        # Name is revealed
        if inline:
            st.markdown(f"**{patient_name}** (`{patient_hash}`)")
        else:
            st.markdown(f"**Patient**: {patient_name}")
            st.caption(f"Hash: {patient_hash}")
    else:
        # Name is hidden - show reveal button
        if inline:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔓 Show Name", key=f"btn_reveal_{patient_hash}", help="Click to reveal patient name"):
                    st.session_state[reveal_key] = True
                    st.rerun()
            with col2:
                st.caption(f"Patient: `{patient_hash}`")
        else:
            if st.button(
                "🔓 Show Name",
                key=f"btn_reveal_{patient_hash}",
                use_container_width=True,
                help="Click to reveal patient name",
            ):
                st.session_state[reveal_key] = True
                st.rerun()
            st.caption(f"Hash: {patient_hash}")


def render_protected_name_simple(patient_hash: str, patient_name: str = None):
    """
    Simple inline version of protected patient name.

    Args:
        patient_hash: 8-character patient hash
        patient_name: Plain text patient name (PHI - protected)

    Returns:
        Inline display with toggle button
    """
    reveal_key = f"reveal_{patient_hash}"

    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    if st.session_state[reveal_key] and patient_name:
        return f"**{patient_name}** (`{patient_hash}`)"
    else:
        return f"Patient: `{patient_hash}`"
