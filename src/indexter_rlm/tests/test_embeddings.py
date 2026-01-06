"""Tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indexter_rlm.config import EmbeddingProvider
from indexter_rlm.embeddings import (
    LocalEmbedder,
    OpenAIEmbedder,
    clear_embedder_cache,
    get_embedder,
)


class TestLocalEmbedder:
    """Tests for LocalEmbedder."""

    def test_model_name(self):
        """Test model_name property."""
        embedder = LocalEmbedder(model="BAAI/bge-small-en-v1.5")
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_default_model(self):
        """Test default model."""
        embedder = LocalEmbedder()
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    async def test_embed(self):
        """Test embedding generation with mocked embedder."""
        embedder = LocalEmbedder()

        # Mock the internal embedder
        mock_internal = MagicMock()
        mock_internal.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedder._embedder = mock_internal
        embedder._dims = 3

        result = await embedder.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    def test_model_name(self):
        """Test model_name property."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        assert embedder.model_name == "text-embedding-3-small"

    def test_dimensions_small(self):
        """Test dimensions for text-embedding-3-small."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        assert embedder.dimensions == 1536

    def test_dimensions_large(self):
        """Test dimensions for text-embedding-3-large."""
        embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="test-key")
        assert embedder.dimensions == 3072

    def test_requires_api_key(self):
        """Test that OpenAIEmbedder requires an API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIEmbedder(model="text-embedding-3-small")

    def test_uses_env_api_key(self):
        """Test that OpenAIEmbedder uses OPENAI_API_KEY from env."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            embedder = OpenAIEmbedder(model="text-embedding-3-small")
            assert embedder._api_key == "env-key"

    async def test_embed(self):
        """Test embedding generation with mock client."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")

        # Mock the OpenAI client response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        embedder._client = mock_client

        result = await embedder.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


class TestGetEmbedder:
    """Tests for get_embedder factory function."""

    def setup_method(self):
        """Clear embedder cache before each test."""
        clear_embedder_cache()

    def teardown_method(self):
        """Clear embedder cache after each test."""
        clear_embedder_cache()

    def test_get_local_embedder(self):
        """Test getting a local embedder."""
        with patch("indexter_rlm.embeddings.settings") as mock_settings:
            mock_settings.embedding.provider = EmbeddingProvider.local
            mock_settings.embedding.model = "BAAI/bge-small-en-v1.5"

            embedder = get_embedder()

            assert isinstance(embedder, LocalEmbedder)
            assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_get_openai_embedder(self):
        """Test getting an OpenAI embedder."""
        with patch("indexter_rlm.embeddings.settings") as mock_settings:
            mock_settings.embedding.provider = EmbeddingProvider.openai
            mock_settings.embedding.model = "text-embedding-3-small"
            mock_settings.embedding.openai_api_key = "test-key"

            embedder = get_embedder()

            assert isinstance(embedder, OpenAIEmbedder)
            assert embedder.model_name == "text-embedding-3-small"

    def test_fallback_to_local_without_api_key(self):
        """Test fallback to local when no OpenAI API key."""
        with patch("indexter_rlm.embeddings.settings") as mock_settings:
            mock_settings.embedding.provider = EmbeddingProvider.openai
            mock_settings.embedding.model = "text-embedding-3-small"
            mock_settings.embedding.openai_api_key = None

            with patch.dict("os.environ", {}, clear=True):
                embedder = get_embedder()

                # Should fall back to local
                assert isinstance(embedder, LocalEmbedder)

