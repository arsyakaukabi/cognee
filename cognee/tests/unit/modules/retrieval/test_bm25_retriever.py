"""
Unit tests for BM25 Retriever.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.bm25_retriever import BM25Retriever


class TestBM25Tokenizer:
    """Tests for BM25 tokenizer function."""

    def test_tokenizer_lowercase(self):
        """Tokenizer should lowercase all tokens."""
        retriever = BM25Retriever(top_k=5)
        tokens = retriever._bm25_tokenizer("Hello WORLD Test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenizer_special_chars(self):
        """Tokenizer should extract only word characters."""
        retriever = BM25Retriever(top_k=5)
        tokens = retriever._bm25_tokenizer("hello, world! test-case")
        assert tokens == ["hello", "world", "test", "case"]

    def test_tokenizer_empty_string(self):
        """Tokenizer should return empty list for empty input."""
        retriever = BM25Retriever(top_k=5)
        tokens = retriever._bm25_tokenizer("")
        assert tokens == []

    def test_tokenizer_with_stop_words(self):
        """Tokenizer should filter out stop words when provided."""
        retriever = BM25Retriever(top_k=5, stop_words=["the", "a", "is"])
        tokens = retriever._bm25_tokenizer("The quick fox is a fast animal")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_tokenizer_numbers(self):
        """Tokenizer should handle numbers."""
        retriever = BM25Retriever(top_k=5)
        tokens = retriever._bm25_tokenizer("version 2.0 release 123")
        assert "version" in tokens
        assert "2" in tokens
        assert "0" in tokens
        assert "release" in tokens
        assert "123" in tokens


class TestBM25Initialization:
    """Tests for BM25 retriever initialization."""

    def test_init_defaults(self):
        """Test default initialization values."""
        retriever = BM25Retriever()
        assert retriever.top_k == 10
        assert retriever.with_scores is False
        assert retriever.bm25_index is None
        assert retriever._initialized is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        retriever = BM25Retriever(top_k=20, with_scores=True, stop_words=["the"])
        assert retriever.top_k == 20
        assert retriever.with_scores is True
        assert "the" in retriever.stop_words


class TestBM25GetContext:
    """Tests for BM25 get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_empty_query(self):
        """Empty query after tokenization should return empty results."""
        retriever = BM25Retriever(top_k=5)
        
        # Mock initialization
        retriever._initialized = True
        retriever._corpus_ids = ["chunk1", "chunk2"]
        retriever.chunks = {
            "chunk1": ["hello", "world"],
            "chunk2": ["test", "document"],
        }
        retriever.payloads = {
            "chunk1": {"id": "chunk1", "text": "hello world"},
            "chunk2": {"id": "chunk2", "text": "test document"},
        }
        
        # Create mock BM25 index
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [0.5, 0.3]
        retriever.bm25_index = mock_bm25
        
        # Query with only special chars that tokenize to empty
        results = await retriever.get_context("... !!!")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_context_returns_top_k(self):
        """Should return at most top_k results."""
        retriever = BM25Retriever(top_k=2)
        
        # Mock initialization
        retriever._initialized = True
        retriever._corpus_ids = ["chunk1", "chunk2", "chunk3"]
        retriever.chunks = {
            "chunk1": ["hello"],
            "chunk2": ["world"],
            "chunk3": ["test"],
        }
        retriever.payloads = {
            "chunk1": {"id": "chunk1", "text": "hello"},
            "chunk2": {"id": "chunk2", "text": "world"},
            "chunk3": {"id": "chunk3", "text": "test"},
        }
        
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [0.8, 0.5, 0.3]
        retriever.bm25_index = mock_bm25
        
        results = await retriever.get_context("hello")
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_get_context_with_scores(self):
        """With with_scores=True, should return (payload, score) tuples."""
        retriever = BM25Retriever(top_k=2, with_scores=True)
        
        retriever._initialized = True
        retriever._corpus_ids = ["chunk1", "chunk2"]
        retriever.chunks = {
            "chunk1": ["hello"],
            "chunk2": ["world"],
        }
        retriever.payloads = {
            "chunk1": {"id": "chunk1", "text": "hello"},
            "chunk2": {"id": "chunk2", "text": "world"},
        }
        
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [2.5, 1.0]
        retriever.bm25_index = mock_bm25
        
        results = await retriever.get_context("hello")
        
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        payload, score = results[0]
        assert "id" in payload
        assert isinstance(score, (int, float))

    @pytest.mark.asyncio
    async def test_get_context_no_index(self):
        """Should return empty when BM25 index is not available."""
        retriever = BM25Retriever(top_k=5)
        retriever._initialized = True
        retriever.bm25_index = None
        retriever._corpus_ids = []
        
        results = await retriever.get_context("test query")
        assert results == []
