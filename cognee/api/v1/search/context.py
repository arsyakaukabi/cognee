"""
Context retrieval module.
Retrieves context from the graph (triplets) and resolves it to text.
"""

import asyncio
from typing import List, Optional
from uuid import UUID
import time

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.performance_utils import get_correlation_id, debug_trace_enabled

timing_logger = get_logger("timing")

async def get_context(
    query: str,
    top_k: int = 10,
    datasets: Optional[List[str]] = None,
    dataset_ids: Optional[List[UUID]] = None,
    wide_search_top_k: Optional[int] = 100,
    user: Optional[User] = None,
) -> str:
    """
    Retrieve context from the graph and resolve it to text.
    """
    timing_on = debug_trace_enabled.get()
    request_id = get_correlation_id() if timing_on else None

    async def _context_for_current_db() -> str:
        t_retriever_start = time.perf_counter_ns() if timing_on else 0
        retriever = GraphCompletionRetriever(top_k=top_k, wide_search_top_k=wide_search_top_k)
        if timing_on:
            dur_ms = (time.perf_counter_ns() - t_retriever_start) / 1_000_000
            timing_logger.info(
                "ctx_retriever_build",
                fn="get_context",
                request_id=request_id,
                dur_ms=round(dur_ms, 3),
            )

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

    # In access-control mode, dataset-specific DB context is selected via contextvars.
    # If dataset_ids is provided, mirror the behavior of other search endpoints and run
    # the retrieval within each dataset's DB context.
    if backend_access_control_enabled() and user is not None and dataset_ids:
        search_datasets = await get_authorized_existing_datasets(
            datasets=dataset_ids, permission_type="read", user=user
        )
        if not search_datasets:
            return ""

        async def _context_for_dataset(dataset) -> str:
            await set_database_global_context_variables(dataset.id, dataset.owner_id)
            return await _context_for_current_db()

        texts = await asyncio.gather(*[_context_for_dataset(d) for d in search_datasets])
        return "\n\n".join([t for t in texts if t])

    # Fallback: use the current DB context (existing behavior).
    return await _context_for_current_db()
