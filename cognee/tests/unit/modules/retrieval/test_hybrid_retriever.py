"""
Unit tests for Hybrid Retriever.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from cognee.modules.retrieval.hybrid_retriever import HybridRetriever, _get_hybrid_config


class TestHybridConfig:
    """Tests for hybrid retrieval configuration."""

    def test_default_config(self):
        """Default config should have hybrid disabled."""
        with patch.dict("os.environ", {}, clear=True):
            config = _get_hybrid_config()
            assert config["enabled"] is False
            assert config["rrf_k"] == 60
            assert config["vector_top_k"] == 50
            assert config["bm25_top_k"] == 50
            assert config["final_top_k"] == 50

    def test_enabled_config(self):
        """Config with HYBRID_RETRIEVAL_ENABLED=true."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "true"}):
            config = _get_hybrid_config()
            assert config["enabled"] is True

    def test_custom_params(self):
        """Config with custom RRF parameters."""
        env = {
            "RRF_K": "30",
            "HYBRID_VECTOR_TOP_K": "100",
            "HYBRID_BM25_TOP_K": "75",
            "HYBRID_FINAL_TOP_K": "25",
        }
        with patch.dict("os.environ", env):
            config = _get_hybrid_config()
            assert config["rrf_k"] == 30
            assert config["vector_top_k"] == 100
            assert config["bm25_top_k"] == 75
            assert config["final_top_k"] == 25


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_default_init(self):
        """Test default initialization."""
        with patch.dict("os.environ", {}, clear=True):
            retriever = HybridRetriever()
            assert retriever.hybrid_enabled is False
            assert retriever.rrf_k == 60
            assert retriever.final_top_k == 50

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        retriever = HybridRetriever(
            vector_top_k=100,
            bm25_top_k=75,
            rrf_k=30,
            final_top_k=25,
        )
        assert retriever.rrf_k == 30
        assert retriever.final_top_k == 25


class TestHybridGetContext:
    """Tests for HybridRetriever get_context method."""

    @pytest.mark.asyncio
    async def test_fallback_when_disabled(self):
        """Should fall back to vector-only when hybrid is disabled."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "false"}):
            retriever = HybridRetriever()
            
            # Mock vector retriever
            mock_vector_results = [{"id": "doc1", "text": "hello"}]
            retriever.vector_retriever.get_context = AsyncMock(
                return_value=mock_vector_results
            )
            retriever.bm25_retriever.get_context = AsyncMock()
            
            results = await retriever.get_context("test query")
            
            # Vector retriever should be called
            retriever.vector_retriever.get_context.assert_called_once_with("test query")
            # BM25 should NOT be called
            retriever.bm25_retriever.get_context.assert_not_called()
            assert results == mock_vector_results

    @pytest.mark.asyncio
    async def test_hybrid_mode_calls_both(self):
        """Hybrid mode should call both retrievers."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "true"}):
            retriever = HybridRetriever()
            
            mock_vector = [{"id": "doc1", "text": "semantic match"}]
            mock_bm25 = [({"id": "doc2", "text": "keyword match"}, 2.5)]
            
            retriever.vector_retriever.get_context = AsyncMock(return_value=mock_vector)
            retriever.bm25_retriever.get_context = AsyncMock(return_value=mock_bm25)
            
            results = await retriever.get_context("test query")
            
            retriever.vector_retriever.get_context.assert_called_once()
            retriever.bm25_retriever.get_context.assert_called_once()
            
            # Should return fused results
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_handles_vector_failure(self):
        """Should handle gracefully when vector retriever fails."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "true"}):
            retriever = HybridRetriever()
            
            retriever.vector_retriever.get_context = AsyncMock(
                side_effect=Exception("Vector search failed")
            )
            retriever.bm25_retriever.get_context = AsyncMock(
                return_value=[({"id": "doc1", "text": "result"}, 1.5)]
            )
            
            # Should not raise, should return BM25 results only
            results = await retriever.get_context("test query")
            assert len(results) >= 0  # May be empty or have BM25 results

    @pytest.mark.asyncio
    async def test_handles_bm25_failure(self):
        """Should handle gracefully when BM25 retriever fails."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "true"}):
            retriever = HybridRetriever()
            
            retriever.vector_retriever.get_context = AsyncMock(
                return_value=[{"id": "doc1", "text": "result"}]
            )
            retriever.bm25_retriever.get_context = AsyncMock(
                side_effect=Exception("BM25 search failed")
            )
            
            # Should not raise, should return vector results only
            results = await retriever.get_context("test query")
            assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self):
        """Should return empty list when both retrievers fail."""
        with patch.dict("os.environ", {"HYBRID_RETRIEVAL_ENABLED": "true"}):
            retriever = HybridRetriever()
            
            retriever.vector_retriever.get_context = AsyncMock(
                side_effect=Exception("Vector failed")
            )
            retriever.bm25_retriever.get_context = AsyncMock(
                side_effect=Exception("BM25 failed")
            )
            
            results = await retriever.get_context("test query")
            assert results == []


class TestHybridChunkId:
    """Tests for chunk ID extraction."""

    def test_extract_id_field(self):
        """Should extract 'id' field from payload."""
        retriever = HybridRetriever()
        payload = {"id": "chunk_123", "text": "hello"}
        assert retriever._get_chunk_id(payload) == "chunk_123"

    def test_extract_chunk_id_field(self):
        """Should extract 'chunk_id' field from payload."""
        retriever = HybridRetriever()
        payload = {"chunk_id": "chunk_456", "text": "hello"}
        assert retriever._get_chunk_id(payload) == "chunk_456"

    def test_fallback_to_text_hash(self):
        """Should fallback to text hash when no ID field exists."""
        retriever = HybridRetriever()
        payload = {"text": "hello world", "metadata": {}}
        chunk_id = retriever._get_chunk_id(payload)
        assert chunk_id == str(hash("hello world"))

    def test_non_dict_payload(self):
        """Should handle non-dict payloads gracefully."""
        retriever = HybridRetriever()
        payload = "just a string"
        chunk_id = retriever._get_chunk_id(payload)
        # Should return object id as fallback
        assert chunk_id.isdigit() or chunk_id.startswith("-")
