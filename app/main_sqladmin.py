"""Main entry point for the FastAPI application."""

from .warnings_filter import filter_warnings

filter_warnings()

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from .logger import logger
from .temporal.manager import temporal_manager

# Load environment variables from .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    This function is used to perform startup and shutdown tasks for the FastAPI application.
    It initializes the patient database, saves the OpenAPI JSON, and connects to the Temporal server.
    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Initialize patient database
    import os
    from .patients.database import init_db

    fresh_start = os.getenv("DB_FRESH_START", "false").lower() == "true"
    init_db(fresh_start=fresh_start)
    logger.info(f"Patient database initialized (fresh_start={fresh_start})")

    # Connect to Temporal
    await temporal_manager.get_client()
    yield
