"""Main entry point for the FastAPI application."""

from .warnings_filter import filter_warnings

filter_warnings()

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, RedirectResponse
from scalar_fastapi import get_scalar_api_reference

from .config import Config
from .routers import stt, stt_services, temporal_tasks, medical
from .temporal.manager import temporal_manager
from .trace_middleware import TraceMiddleware

# Load environment variables from .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    This function is used to perform startup and shutdown tasks for the FastAPI application.
    It saves the OpenAPI JSON and connects to the Temporal server.
    Args:
        app (FastAPI): The FastAPI application instance.
    """
    await temporal_manager.get_client()
    yield


tags_metadata = [
    {
        "name": "Speech-2-Text",
        "description": """Complete speech-to-text processing pipeline with transcription, alignment, and diarization.
        
**Primary Endpoints:**
- `/speech-to-text` - Upload audio/video files for processing
- `/speech-to-text-url` - Process files from URLs

**Supported Formats:** MP3, WAV, M4A, FLAC, OGG, MP4, AVI, MOV

**Features:**
- Automatic speech recognition using WhisperX
- Word-level alignment with timestamps
- Speaker diarization (who spoke when)
- Multi-language support (English, Vietnamese, Chinese, Cantonese)
- Temporal workflow orchestration for reliability
        """,
    },
    {
        "name": "Speech-2-Text services",
        "description": """Individual processing services for granular control.
        
Use these endpoints when you need specific processing steps rather than the full pipeline:
- **Transcribe** - Convert speech to text
- **Align** - Add precise word-level timestamps
- **Diarize** - Identify different speakers
- **Combine** - Merge transcription with speaker labels
        """,
    },
    {
        "name": "Tasks Management",
        "description": """Monitor and manage Temporal workflow tasks.
        
**Capabilities:**
- Check workflow status and progress
- Retrieve completed results
- Query task history
- Monitor processing workflows

All speech-to-text operations return a workflow ID that can be used with these endpoints.
        """,
    },
    {
        "name": "Medical",
        "description": """Medical transcript processing with HIPAA compliance features.
        
**Features:**
- PHI (Protected Health Information) detection
- Medical entity extraction (diagnoses, medications, procedures)
- SOAP note generation from consultations
- RAG-powered medical chatbot
- Vector similarity search for consultations
- HIPAA audit logging

**Security:** All operations include access control and comprehensive audit trails.
        """,
    },
    {
        "name": "Health",
        "description": """Health check endpoints for monitoring and orchestration.
        
- `/health` - Simple health check
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Readiness check with dependency validation
        """,
    },
]

app = FastAPI(
    title="WhisperX API - Advanced Speech Processing",
    description=f"""
# WhisperX RESTful API

Welcome to the **WhisperX API** - a powerful, production-ready service for advanced speech-to-text processing with transcription, alignment, diarization, and medical document analysis.

## Quick Start

### Basic Transcription Example

```bash
# Upload an audio file for processing
curl -X POST "http://localhost:8000/speech-to-text?language=en&model=base" \\
  -F "file=@audio.mp3"

# Response: {{"identifier": "whisperx-workflow-abc123", "message": "Workflow started"}}

# Check workflow status
curl "http://localhost:8000/temporal/workflow/whisperx-workflow-abc123"
```

### Process from URL

```bash
curl -X POST "http://localhost:8000/speech-to-text-url?language=vi" \\
  -F "url=https://example.com/audio.mp3"
```

## Key Features

### Speech Processing
- **Multi-language Support** - 4 languages: English (en), Vietnamese (vi), Chinese (zh), and Cantonese (yue)
- **High Accuracy** - Powered by WhisperX state-of-the-art models
- **Word-level Timestamps** - Precise alignment for subtitles and analytics
- **Speaker Diarization** - Automatic speaker identification and separation
- **Temporal Workflows** - Reliable, fault-tolerant processing

### Medical Processing
- **PHI Detection** - HIPAA-compliant Protected Health Information detection
- **Medical NER** - Extract diagnoses, medications, procedures, and symptoms
- **SOAP Notes** - Auto-generate structured clinical documentation
- **RAG Chatbot** - Query patient records with AI-powered search
- **Audit Logging** - Complete HIPAA audit trail

## Supported Formats

**Audio:** {", ".join(Config.AUDIO_EXTENSIONS)}

**Video:** {", ".join(Config.VIDEO_EXTENSIONS)}

## Architecture

This API uses Temporal.io for workflow orchestration, ensuring:
- Automatic retries on failures
- Progress tracking and monitoring
- Horizontal scalability
- Fault tolerance

## Resources

- [WhisperX GitHub](https://github.com/m-bain/whisperX) - Core processing engine
- [API Documentation](/scalar) - Interactive API explorer
- [Swagger UI](/docs) - Traditional OpenAPI documentation

## Security & Compliance

- HIPAA-compliant medical processing
- Access control and authentication
- Comprehensive audit logging
- PHI detection and de-identification

---

**Version:** 0.0.1 | **License:** MIT
    """,
    version="0.0.1",
    contact={
        "name": "WhisperX API Support",
        "url": "https://github.com/m-bain/whisperX",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # Hide schemas section by default
        "displayRequestDuration": True,  # Show request duration
        "filter": True,  # Enable filtering
        "tryItOutEnabled": True,  # Enable Try it out by default
    },
)

# Add trace middleware
app.add_middleware(TraceMiddleware)

# Include routers
app.include_router(stt.stt_router)
app.include_router(stt_services.service_router)
app.include_router(temporal_tasks.temporal_router)
app.include_router(medical.router)


@app.get("/", include_in_schema=False)
async def index():
    """Redirect to the modern Scalar API documentation."""
    return RedirectResponse(url="/scalar", status_code=307)


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    """Scalar API Documentation - Modern, interactive API explorer."""
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


# Health check endpoints
@app.get("/health", tags=["Health"], summary="Simple health check")
async def health_check():
    """Verify the service is up and running.

    Returns a simple status response to confirm the API service is operational.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "Service is running"},
    )


@app.get("/health/live", tags=["Health"], summary="Liveness check")
async def liveness_check():
    """Check if the application is running.

    Used by orchestration systems like Kubernetes to detect if the app is alive.
    Returns timestamp along with status information.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "timestamp": time.time(),
            "message": "Application is live",
        },
    )


@app.get("/health/ready", tags=["Health"], summary="Readiness check")
async def readiness_check():
    """Check if the application is ready to accept requests.
    Verifies dependencies like the Temporal server are connected and ready.
    Returns HTTP 200 if all systems are operational, HTTP 503 if any dependency
    has failed.
    """
    try:
        # Check temporal connection
        await temporal_manager.get_client()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "temporal": "connected",
                "message": "Application is ready to accept requests",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "temporal": "disconnected",
                "message": f"Application is not ready: {str(e)}",
            },
        )
