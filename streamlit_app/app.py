"""
whisperX Medical Transcription UI

Main Streamlit application entry point.
Navigation to pages is handled via streamlit's multipage app framework.
"""

import streamlit as st

# Page configuration - set before any other st commands
st.set_page_config(
    page_title="whisperX Medical Transcription",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #2196F3, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(76, 175, 80, 0.1));
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.switch_page("pages/0_🏠_Home.py")
