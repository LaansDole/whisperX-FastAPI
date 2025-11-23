"""Temporal client manager for connection handling."""

from temporalio.client import Client
from app.logger import logger
from .config import config


class TemporalManager:
    def __init__(self):
        self._client: Client | None = None

    async def get_client(self) -> Client | None:
        if self._client is None:
            try:
                self._client = await Client.connect(
                    config.TEMPORAL_SERVER_URL, namespace=config.TEMPORAL_NAMESPACE
                )
                logger.info("Successfully connected to Temporal.")
            except Exception as e:
                logger.warning(
                    "Failed to connect to Temporal. "
                    f"Temporal-dependent features will not be available: {e}"
                )
                self._client = None  # Explicitly set to None on failure
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None


temporal_manager = TemporalManager()
