"""
Hybrid Retriever combining vector search and BM25 lexical search with RRF fusion.

This module provides a hybrid retrieval approach that runs both vector-based
semantic search and BM25-based lexical search in parallel, then combines
results using Reciprocal Rank Fusion (RRF) for improved retrieval quality.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Type

from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.bm25_retriever import BM25Retriever
from cognee.modules.retrieval.utils.rrf_fusion import rrf_fusion_with_payloads
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


def _get_hybrid_config() -> Dict[str, Any]:
    """
    Load hybrid retrieval configuration from environment variables.

    Returns:
        Dictionary with configuration values.
    """
    return {
        "enabled": os.getenv("HYBRID_RETRIEVAL_ENABLED", "false").lower() == "true",
        "rrf_k": int(os.getenv("RRF_K", "60")),
        "vector_top_k": int(os.getenv("HYBRID_VECTOR_TOP_K", "50")),
        "bm25_top_k": int(os.getenv("HYBRID_BM25_TOP_K", "50")),
        "final_top_k": int(os.getenv("HYBRID_FINAL_TOP_K", "50")),
    }


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector and BM25 search with RRF fusion.

    This retriever runs both semantic (vector) and lexical (BM25) search
    in parallel, then fuses results using Reciprocal Rank Fusion to provide
    the best of both approaches.

    When hybrid mode is disabled via configuration, falls back to vector-only
    search for backward compatibility.

    Attributes:
        vector_retriever: ChunksRetriever for semantic search.
        bm25_retriever: BM25Retriever for lexical search.
        rrf_k: RRF smoothing constant.
        final_top_k: Number of final results after fusion.
        hybrid_enabled: Whether hybrid mode is active.
    """

    def __init__(
        self,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_top_k: Max results from vector search. Defaults to config.
            bm25_top_k: Max results from BM25 search. Defaults to config.
            rrf_k: RRF smoothing constant. Defaults to config.
            final_top_k: Final result count after fusion. Defaults to config.
        """
        config = _get_hybrid_config()

        self.hybrid_enabled = config["enabled"]
        self.rrf_k = rrf_k if rrf_k is not None else config["rrf_k"]
        self.final_top_k = final_top_k if final_top_k is not None else config["final_top_k"]

        vector_k = vector_top_k if vector_top_k is not None else config["vector_top_k"]
        bm25_k = bm25_top_k if bm25_top_k is not None else config["bm25_top_k"]

        self.vector_retriever = ChunksRetriever(top_k=vector_k)
        self.bm25_retriever = BM25Retriever(top_k=bm25_k, with_scores=True)

        logger.info(
            "HybridRetriever initialized: hybrid_enabled=%s, rrf_k=%d, "
            "vector_top_k=%d, bm25_top_k=%d, final_top_k=%d",
            self.hybrid_enabled,
            self.rrf_k,
            vector_k,
            bm25_k,
            self.final_top_k,
        )

    async def get_context(self, query: str) -> List[Any]:
        """
        Retrieve relevant chunks using hybrid search.

        If hybrid mode is enabled, runs both vector and BM25 search in parallel
        and fuses results with RRF. Otherwise, falls back to vector-only search.

        Args:
            query: The search query string.

        Returns:
            List of chunk payloads ranked by combined relevance.
        """
        if not self.hybrid_enabled:
            logger.info("Hybrid mode disabled, falling back to vector-only search")
            return await self.vector_retriever.get_context(query)

        logger.info("Running hybrid search for query: '%s'", query[:100])

        # Run both retrievers in parallel
        vector_task = self._get_vector_results_with_ids(query)
        bm25_task = self.bm25_retriever.get_context(query)

        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task, return_exceptions=True
        )

        # Handle exceptions gracefully
        if isinstance(vector_results, Exception):
            logger.error("Vector search failed: %s", vector_results)
            vector_results = []

        if isinstance(bm25_results, Exception):
            logger.error("BM25 search failed: %s", bm25_results)
            bm25_results = []

        # If both failed, return empty
        if not vector_results and not bm25_results:
            logger.warning("Both retrievers returned no results")
            return []

        # Convert to (id, score, payload) format for RRF
        vector_tuples = [
            (self._get_chunk_id(payload), rank, payload)
            for rank, payload in enumerate(vector_results)
        ]

        bm25_tuples = [
            (self._get_chunk_id(payload), score, payload)
            for payload, score in bm25_results
        ]

        logger.info(
            "Retrieved %d vector results and %d BM25 results",
            len(vector_tuples),
            len(bm25_tuples),
        )

        # Apply RRF fusion
        fused = rrf_fusion_with_payloads(
            [vector_tuples, bm25_tuples],
            k=self.rrf_k,
            final_top_k=self.final_top_k,
        )

        logger.info("RRF fusion produced %d results", len(fused))

        # Return payloads only
        return [payload for _, _, payload in fused]

    async def _get_vector_results_with_ids(self, query: str) -> List[Any]:
        """
        Get vector search results with payload preservation.

        Args:
            query: Search query.

        Returns:
            List of chunk payloads from vector search.
        """
        return await self.vector_retriever.get_context(query)

    def _get_chunk_id(self, payload: Any) -> str:
        """
        Extract unique ID from chunk payload.

        Args:
            payload: Chunk payload dictionary.

        Returns:
            String ID for the chunk.
        """
        if isinstance(payload, dict):
            # Try common ID fields
            for key in ("id", "chunk_id", "_id"):
                if key in payload:
                    return str(payload[key])
            # Fallback to text hash if no ID
            if "text" in payload:
                return str(hash(payload["text"]))
        return str(id(payload))

    async def get_completion(
        self,
        query: str,
        context: Optional[Any] = None,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ) -> List[Any]:
        """
        Get completion using hybrid context retrieval.

        Args:
            query: The query string.
            context: Optional pre-fetched context.
            session_id: Optional session identifier.
            response_model: Response model type (unused, for interface compatibility).

        Returns:
            List containing the context (hybrid retrieval doesn't generate completions).
        """
        if context is None:
            context = await self.get_context(query)
        return context
