"""
Embedding generation for Indexter-RLM.

Supports multiple embedding providers:
- Local: FastEmbed models (default, no API key needed)
- OpenAI: text-embedding-3-small/large (requires OPENAI_API_KEY)

Usage:
    from indexter_rlm.embeddings import get_embedder

    embedder = get_embedder()
    vectors = await embedder.embed(["hello world", "code snippet"])
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from .config import EmbeddingProvider, settings

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for embedding generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the vector dimensions."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...


class LocalEmbedder(Embedder):
    """FastEmbed-based local embedding generation."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        """Initialize with a FastEmbed model.

        Args:
            model: The FastEmbed model name.
        """
        self._model = model
        self._embedder = None
        self._dims: int | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            # Lazy init to get dimensions
            self._ensure_embedder()
        return self._dims or 384

    def _ensure_embedder(self) -> None:
        """Lazy-load the FastEmbed model."""
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self._model)
            # Get dimensions from a test embedding
            test = list(self._embedder.embed(["test"]))[0]
            self._dims = len(test)
            logger.info(f"Loaded FastEmbed model: {self._model} ({self._dims} dims)")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using FastEmbed."""
        self._ensure_embedder()
        # FastEmbed returns a generator, convert to list
        embeddings = list(self._embedder.embed(texts))
        return [list(e) for e in embeddings]


class OpenAIEmbedder(Embedder):
    """OpenAI API-based embedding generation."""

    # Dimensions for known OpenAI models
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        """Initialize with an OpenAI model.

        Args:
            model: The OpenAI embedding model name.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS.get(self._model, 1536)

    def _ensure_client(self) -> None:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
                logger.info(f"Initialized OpenAI client for model: {self._model}")
            except ImportError as err:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from err

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        self._ensure_client()

        # OpenAI API call
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings


def get_embedder() -> Embedder:
    """Get an embedder based on current settings.

    Returns:
        An Embedder instance configured according to settings.

    Raises:
        ValueError: If OpenAI provider is selected but no API key is available.
    """
    provider = settings.embedding.provider
    model = settings.embedding.model

    if provider == EmbeddingProvider.openai:
        api_key = settings.embedding.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OpenAI provider selected but no API key found. Falling back to local FastEmbed."
            )
            return LocalEmbedder(model="BAAI/bge-small-en-v1.5")
        return OpenAIEmbedder(model=model, api_key=api_key)
    else:
        return LocalEmbedder(model=model)


# Module-level cached embedder
_embedder: Embedder | None = None


def get_cached_embedder() -> Embedder:
    """Get a cached embedder instance.

    This is useful for repeated embedding operations to avoid
    re-initializing the model.
    """
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def clear_embedder_cache() -> None:
    """Clear the cached embedder. For testing."""
    global _embedder
    _embedder = None
