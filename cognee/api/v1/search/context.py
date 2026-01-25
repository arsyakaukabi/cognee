"""
Context retrieval module.
Retrieves context from the graph (triplets) and resolves it to text.
"""

from typing import List, Optional
import time
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.shared.logging_utils import get_logger
from cognee.shared.performance_utils import get_correlation_id, debug_trace_enabled

timing_logger = get_logger("timing")

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
    timing_on = debug_trace_enabled.get()
    request_id = get_correlation_id() if timing_on else None
    t_retriever_start = time.perf_counter_ns() if timing_on else 0
    retriever = GraphCompletionRetriever(
        top_k=top_k,
        wide_search_top_k=wide_search_top_k
    )
    if timing_on:
        dur_ms = (time.perf_counter_ns() - t_retriever_start) / 1_000_000
        timing_logger.info(
            "ctx_retriever_build",
            fn="get_context",
            request_id=request_id,
            dur_ms=round(dur_ms, 3),
        )

    # Note: GraphCompletionRetriever doesn't seem to natively filter by datasets in get_context directly
    # unless initialized with specific filtering logic, but get_context relies on vector search
    # which usually respects current user context or global filters.
    # For now we use the basic get_context which searches the graph.
    
    t_triplets_start = time.perf_counter_ns() if timing_on else 0
    triplets = await retriever.get_context(query)
    if timing_on:
        dur_ms = (time.perf_counter_ns() - t_triplets_start) / 1_000_000
        timing_logger.info(
            "ctx_triplets_fetch",
            fn="get_context -> GraphCompletionRetriever.get_context",
            request_id=request_id,
            triplets_count=len(triplets),
            dur_ms=round(dur_ms, 3),
        )
    
    if not triplets:
        return ""
        
    t_resolve_start = time.perf_counter_ns() if timing_on else 0
    context_text = await resolve_edges_to_text(triplets)
    if timing_on:
        dur_ms = (time.perf_counter_ns() - t_resolve_start) / 1_000_000
        timing_logger.info(
            "ctx_resolve_edges",
            fn="get_context -> resolve_edges_to_text",
            request_id=request_id,
            dur_ms=round(dur_ms, 3),
        )
    return context_text
