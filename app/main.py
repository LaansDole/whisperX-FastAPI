"""Main entry point for the FastAPI application."""

from .warnings_filter import filter_warnings

filter_warnings()

import time 
from contextlib import asynccontextmanager 

from dotenv import load_dotenv 
from fastapi import FastAPI, status 
from fastapi.responses import JSONResponse, RedirectResponse 

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
        "description": "Operations related to transcript",
    },
    {
        "name": "Speech-2-Text services",
        "description": "Individual services for transcript",
    },
    {
        "name": "Tasks Management",
        "description": "Manage tasks.",
    },
    {
        "name": "Health",
        "description": "Health check endpoints to monitor application status",
    },
]

app = FastAPI(
    title="whisperX REST service",
    description=f"""
    # whisperX RESTful API

    Welcome to the whisperX RESTful API! This API provides a suite of audio processing services to enhance and analyze your audio content.

    ## Documentation:

    For detailed information on request and response formats, consult the [WhisperX Documentation](https://github.com/m-bain/whisperX).

    ## Services:

    Speech-2-Text provides a suite of audio processing services to enhance and analyze your audio content. The following services are available:

    1. Transcribe: Transcribe an audio/video  file into text.
    2. Align: Align the transcript to the audio/video file.
    3. Diarize: Diarize an audio/video file into speakers.
    4. Combine Transcript and Diarization: Combine the transcript and diarization results.

    ## Supported file extensions:
    AUDIO_EXTENSIONS = {Config.AUDIO_EXTENSIONS}

    VIDEO_EXTENSIONS = {Config.VIDEO_EXTENSIONS}

    """,
    version="0.0.1",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
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
    """Redirect to the documentation."""
    return RedirectResponse(url="/docs", status_code=307)


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
