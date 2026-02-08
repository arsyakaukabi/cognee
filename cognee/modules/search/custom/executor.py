"""
Search executor for Graph Completion and Chunk searches.
"""

from typing import Any, List, Literal, Optional

import cognee

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("CustomSearchExecutor")


class SearchExecutor:
    """Executor for search operations."""

    def __init__(self, initial_top_k: int = 50):
        """
        Initialize search executor.

        Args:
            initial_top_k: Default top_k for retrievers (default: 50, increased to reduce expansion)
        """
        self.graph_retriever = GraphCompletionRetriever(top_k=initial_top_k, node_type=NodeSet)
        self.chunk_retriever = ChunksRetriever(top_k=initial_top_k)

    async def execute_graph_completion_search(
        self, question: str, top_k: int = 10, node_name: Optional[List[str]] = None
    ) -> List[Edge]:
        """
        Execute Graph Completion search.

        Args:
            question: Query text
            top_k: Number of results to retrieve
            node_name: List of node names (Dataset IDs) to filter by

        Returns:
            List of Edge objects (triplets)
        """
        logger.debug(
            f"Executing Graph Completion search: query='{question[:50]}...', top_k={top_k}, node_name={node_name}"
        )

        self.graph_retriever.top_k = top_k
        triplets = await self.graph_retriever.get_context(question, node_name=node_name)

        logger.debug(f"Graph Completion search returned {len(triplets)} triplets")
        return triplets

    async def execute_chunk_search(
        self, question: str, top_k: int = 10, node_name: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Execute Chunk search.

        Args:
            question: Query text
            top_k: Number of results to retrieve
            node_name: List of node names (Dataset IDs) to filter by

        Returns:
            List of chunk results (ScoredResult or dict)
        """
        logger.debug(
            f"Executing Chunk search: query='{question[:50]}...', top_k={top_k}, node_name={node_name}"
        )

        self.chunk_retriever.top_k = top_k
        results = await cognee.search(
            query_type=SearchType.CHUNKS,
            query_text=question,
            top_k=top_k,
            node_name=node_name,
        )

        logger.debug(f"Chunk search returned {len(results)} results")
        return results

    async def get_search_results(
        self,
        question: str,
        method: Literal["graph", "chunks"],
        top_k: int = 10,
        node_name: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Unified search interface.

        Args:
            question: Query text
            method: Search method ("graph" or "chunks")
            top_k: Number of results to retrieve
            node_name: Optional node set filter

        Returns:
            List of search results
        """
        if method == "graph":
            return await self.execute_graph_completion_search(question, top_k, node_name=node_name)
        if method == "chunks":
            return await self.execute_chunk_search(question, top_k, node_name=node_name)
        raise ValueError(f"Unknown method: {method}")
