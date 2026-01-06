# ADR 009: Streamlit Demonstration UI

- **Date**: 2026-01-06
- **Status**: Accepted

## Context

The whisperX Medical Transcription API needed a demonstration interface to showcase capabilities without requiring API integration. A lightweight, easy-to-deploy UI was needed for:
- Testing and demonstration purposes
- Non-technical stakeholders to interact with the system
- Quick prototyping of workflow scenarios

## Decision

We implemented a **Streamlit-based demonstration UI** with the following features:

### Pages

| Page | Purpose |
|------|---------|
| 🏠 Home | Dashboard with stats and recent activity |
| 📤 Upload | File upload + live voice recording |
| 📊 Workflows | Real-time workflow tracking |
| 👥 Patients | Patient search by hash |

### Key Design Choices

1. **Multi-page Navigation**
   - Consistent sidebar navigation across all pages
   - Numbered page files (`0_🏠_Home.py`, `1_📤_Upload.py`, etc.)

2. **Dual Input Methods**
   - Tab-based UI: "📁 Upload File" and "🎙️ Record Audio"
   - `st.audio_input` widget for browser-based recording (16kHz WAV)
   - No external dependencies (ffmpeg not required)

3. **Sidebar Configuration**
   - WhisperX model selection (tiny → large-v3-turbo)
   - Language selection (en, vi, zh, yue)
   - Speaker diarization settings (min/max speakers)
   - Initial prompt for vocabulary hints

4. **API Client Integration**
   - Thin wrapper around FastAPI backend (`api_client.py`)
   - Cached client instance via `@st.cache_resource`

## Consequences

### Positive
- Zero additional infrastructure (runs alongside FastAPI)
- Rapid iteration for UI/UX testing
- Browser-based audio capture works in Docker
- No HIPAA-critical data stored in UI layer

### Negative
- Not production-ready (demo only)
- No authentication/authorization
- Limited error handling compared to production UI
- Test coverage not required (per project policy)
