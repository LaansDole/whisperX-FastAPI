"""LM Studio client wrapper for OpenAI-compatible API access."""

import httpx
import logging
import asyncio
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LMStudioConfig(BaseModel):
    """Configuration for LM Studio connection."""

    base_url: str = Field(default="http://localhost:1234/v1", description="LM Studio API base URL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    temperature: float = Field(default=0.1, description="Sampling temperature (0.0-1.0)")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: float = Field(default=0.0, description="Presence penalty (-2.0 to 2.0)")
    model: str = Field(default="", description="Model name (empty uses LM Studio's loaded model)")


class LMStudioClient:
    """OpenAI-compatible client for LM Studio."""

    def __init__(self, config: LMStudioConfig = None):
        """Initialize LM Studio client with configuration.

        Args:
            config: LM Studio configuration. If None, uses default configuration.
        """
        self.config = config or LMStudioConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        Send chat completion request to LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            frequency_penalty: Frequency penalty (overrides config)
            presence_penalty: Presence penalty (overrides config)
            stream: Whether to stream the response

        Returns:
            Generated text response

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If response format is invalid
        """
        for attempt in range(self.config.max_retries):
            try:
                # Build request payload
                payload = {
                    "messages": messages,
                    "temperature": temperature or self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "top_p": top_p or self.config.top_p,
                    "frequency_penalty": frequency_penalty or self.config.frequency_penalty,
                    "presence_penalty": presence_penalty or self.config.presence_penalty,
                    "stream": stream,
                }
                # Add model if specified (otherwise LM Studio uses loaded model)
                effective_model = model or self.config.model
                if effective_model:
                    payload["model"] = effective_model

                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    raise ValueError("Invalid response format: no choices returned")
                first_choice = result["choices"][0]
                if "message" not in first_choice or "content" not in first_choice["message"]:
                    raise ValueError("Invalid response format: missing 'message' or 'content' in choices")
                return first_choice["message"]["content"]

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

            except httpx.RequestError as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise

                await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.error(f"Unexpected error in chat completion: {e}")
                raise

    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generate embeddings using LM Studio's embedding model.

        Args:
            text: Text to embed
            model: Embedding model name (uses default if None)

        Returns:
            Embedding vector as list of floats

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If response format is invalid
        """
        try:
            payload = {
                "input": text,
            }
            if model:
                payload["model"] = model

            response = await self.client.post("/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()

            if "data" not in result or not result["data"]:
                raise ValueError("Invalid embedding response format")

            first_data = result["data"][0]
            if not isinstance(first_data, dict) or "embedding" not in first_data:
                raise ValueError("Invalid embedding response format: missing 'embedding' key")
            return first_data["embedding"]
        except Exception as e:
            logger.error(f"Unexpected error in embedding generation: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if LM Studio server is accessible and has models loaded.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                models = response.json()
                return "data" in models and len(models["data"]) > 0
            return False
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in LM Studio.

        Returns:
            List of model dictionaries with id, object, created, owned_by fields
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            models = response.json()
            return models.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary
        """
        try:
            response = await self.client.get(f"/models/{model_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return {}

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to LM Studio and return status information.

        Returns:
            Dictionary with connection status and available models
        """
        is_healthy = await self.health_check()
        models = await self.list_models() if is_healthy else []

        return {
            "status": "ok" if is_healthy else "error",
            "url": self.config.base_url,
            "models_loaded": len(models),
            "models": models,
            "timeout": self.config.timeout,
        }

    async def close(self):
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
