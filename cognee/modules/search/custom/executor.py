"""
Search executor for Graph Completion and Chunk searches.
"""

from typing import List, Literal, Any
import cognee
from cognee.modules.search.types import SearchType
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
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
        self.graph_retriever = GraphCompletionRetriever(top_k=initial_top_k)
        self.chunk_retriever = ChunksRetriever(top_k=initial_top_k)
    
    async def execute_graph_completion_search(self, question: str, top_k: int = 10) -> List[Edge]:
        """
        Execute Graph Completion search.
        
        Args:
            question: Query text
            top_k: Number of results to retrieve
            
        Returns:
            List of Edge objects (triplets)
        """
        logger.debug(f"Executing Graph Completion search: query='{question[:50]}...', top_k={top_k}")
        
        # Update retriever top_k
        self.graph_retriever.top_k = top_k
        
        # Get triplets using retriever
        triplets = await self.graph_retriever.get_context(question)
        
        logger.debug(f"Graph Completion search returned {len(triplets)} triplets")
        return triplets
    
    async def execute_chunk_search(self, question: str, top_k: int = 10) -> List[Any]:
        """
        Execute Chunk search.
        
        Args:
            question: Query text
            top_k: Number of results to retrieve
            
        Returns:
            List of chunk results (ScoredResult or dict)
        """
        logger.debug(f"Executing Chunk search: query='{question[:50]}...', top_k={top_k}")
        
        # Update retriever top_k
        self.chunk_retriever.top_k = top_k
        
        # Use cognee.search for chunk search
        results = await cognee.search(
            query_type=SearchType.CHUNKS,
            query_text=question,
            top_k=top_k,
        )
        
        logger.debug(f"Chunk search returned {len(results)} results")
        return results
    
    async def get_search_results(
        self,
        question: str,
        method: Literal["graph", "chunks"],
        top_k: int = 10
    ) -> List[Any]:
        """
        Unified search interface.
        
        Args:
            question: Query text
            method: Search method ("graph" or "chunks")
            top_k: Number of results to retrieve
            
        Returns:
            List of search results
        """
        if method == "graph":
            return await self.execute_graph_completion_search(question, top_k)
        elif method == "chunks":
            return await self.execute_chunk_search(question, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
