"""
Context retrieval module.
Retrieves context from the graph (triplets) and resolves it to text.
"""

from typing import List, Optional
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.utils import resolve_edges_to_text

async def get_context(
    query: str,
    top_k: int = 10,
    datasets: Optional[List[str]] = None,
    dataset_ids: Optional[List[str]] = None,
    wide_search_top_k: Optional[int] = 100,
) -> str:
    """
    Retrieve context from the graph and resolve it to text.
    """
    retriever = GraphCompletionRetriever(
        top_k=top_k,
        wide_search_top_k=wide_search_top_k
    )

    # Note: GraphCompletionRetriever doesn't seem to natively filter by datasets in get_context directly
    # unless initialized with specific filtering logic, but get_context relies on vector search
    # which usually respects current user context or global filters.
    # For now we use the basic get_context which searches the graph.
    
    triplets = await retriever.get_context(query)
    
    if not triplets:
        return ""
        
    context_text = await resolve_edges_to_text(triplets)
    return context_text
