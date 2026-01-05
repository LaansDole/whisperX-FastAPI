# ADR 009: Live Voice Recording in Streamlit UI

- **Date**: 2026-01-05
- **Status**: Accepted

## Context

The whisperX Medical Transcription system required an additional input method for audio capture beyond file uploads. Users needed the ability to record audio directly from their microphone within the Streamlit UI, similar to the file upload functionality but without needing to pre-record audio externally.

## Decision

We will use **Streamlit's built-in `st.audio_input` widget** to implement live voice recording.

### Key Design Choices

1. **Built-in Widget Over Third-Party Libraries**
   - `st.audio_input` is native to Streamlit 1.52+ (already a dependency)
   - No additional packages (e.g., `streamlit-audiorecorder`, `streamlit-mic-recorder`)
   - No ffmpeg dependency required
   - Returns standard `UploadedFile` (BytesIO) compatible with existing API

2. **Tab-Based UI**
   - Two tabs: "📁 Upload File" and "🎙️ Record Audio"
   - Clear separation of input methods
   - Shared patient information form structure

3. **Audio Configuration**
   - 16000 Hz sample rate (optimal for speech recognition)
   - WAV format output (compatible with WhisperX)
   - Generated filename pattern: `recorded_YYYYMMDD_HHMMSS.wav`

4. **Same Backend API**
   - Both upload and recording use identical `api_client.upload_audio()` method
   - No backend changes required
   - Recordings processed through existing workflow pipeline

## Consequences

### Positive
- Zero new dependencies
- Consistent with existing upload flow
- Optimal audio quality for speech recognition
- Works in Docker environments (browser-based capture)

### Negative
- Requires user to grant microphone permission in browser
- HTTPS required for production (browser security policy)
- Recording quality depends on user's microphone

### Alternatives Considered

1. **`streamlit-audiorecorder`**: Requires ffmpeg system dependency
2. **`streamlit-mic-recorder`**: Additional package, similar functionality
3. **`streamlit-webrtc`**: Over-engineered for simple recording use case
4. **External recording app**: Poor UX, requires file upload step
