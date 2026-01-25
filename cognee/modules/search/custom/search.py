"""
Simple search script to get top 10 knowledge IDs from a query.
"""

from typing import List, Literal, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

from .executor import SearchExecutor
from .mapper import (
    map_triplet_results_to_knowledge_ids,
    map_chunk_results_to_knowledge_ids
)
from .expander import expand_to_n_unique_knowledge_ids
from cognee.shared.performance_utils import trace_perf, TraceSpan

logger = get_logger("CustomSearch")


@trace_perf("custom_search.search_knowledge_ids", "search")
async def search_knowledge_ids(
    question: str,
    method: str = "graph",  # "graph" or "chunks"
    target_n: int = 10,
    initial_top_k: int = 50
) -> List[str]:
    """
    Search and return top N knowledge IDs (TextDocument.name).
    
    Args:
        question: Query text
        method: Search method ("graph" for GRAPH_COMPLETION, "chunks" for CHUNKS)
        target_n: Target number of unique knowledge IDs (default: 10)
        initial_top_k: Initial search top_k (default: 50)
    
    Returns:
        List of knowledge IDs (TextDocument.name), ordered by retrieval rank
    """
    # Setup
    graph_engine = await get_graph_engine()
    search_executor = SearchExecutor(initial_top_k=initial_top_k)
    
    # Execute search
    if method == "graph":
        logger.info(f"🔍 Searching using GRAPH_COMPLETION...")
        async with TraceSpan("search.execute_graph_completion_search", "search"):
            search_results = await search_executor.execute_graph_completion_search(
                question, 
                top_k=initial_top_k
            )
        
        # Map triplets to knowledge IDs with early stopping
        async with TraceSpan("search.map_triplet_results", "search"):
            knowledge_ids = await map_triplet_results_to_knowledge_ids(
                search_results, 
                graph_engine,
                target_n=target_n
            )
        
        # Expand if needed to reach target_n
        if len(set(knowledge_ids)) < target_n:
            logger.info(f"   Initial search: {len(set(knowledge_ids))} unique IDs (target: {target_n})")
            logger.info(f"   Expanding search...")
            async with TraceSpan("search.expand_to_n_unique", "search"):
                knowledge_ids = await expand_to_n_unique_knowledge_ids(
                    search_results=search_results,
                    method="graph",
                    target_n=target_n,
                    graph_engine=graph_engine,
                    search_executor=search_executor,
                    query=question,
                    initial_top_k=initial_top_k,
                    max_top_k=200
                )
        else:
            # Take only first target_n unique IDs
            unique_ids = []
            seen = set()
            for kid in knowledge_ids:
                if kid not in seen:
                    unique_ids.append(kid)
                    seen.add(kid)
                    if len(unique_ids) >= target_n:
                        break
            knowledge_ids = unique_ids
            
    elif method == "chunks":
        logger.info(f"🔍 Searching using CHUNKS...")
        search_results = await search_executor.execute_chunk_search(
            question,
            top_k=initial_top_k
        )
        
        # Map chunks to knowledge IDs with early stopping
        knowledge_ids = await map_chunk_results_to_knowledge_ids(
            search_results, 
            graph_engine,
            target_n=target_n
        )
        
        # Expand if needed to reach target_n
        if len(set(knowledge_ids)) < target_n:
            logger.info(f"   Initial search: {len(set(knowledge_ids))} unique IDs (target: {target_n})")
            logger.info(f"   Expanding search...")
            knowledge_ids = await expand_to_n_unique_knowledge_ids(
                search_results=search_results,
                method="chunks",
                target_n=target_n,
                graph_engine=graph_engine,
                search_executor=search_executor,
                query=question,
                initial_top_k=initial_top_k,
                max_top_k=200
            )
        else:
            # Take only first target_n unique IDs
            unique_ids = []
            seen = set()
            for kid in knowledge_ids:
                if kid not in seen:
                    unique_ids.append(kid)
                    seen.add(kid)
                    if len(unique_ids) >= target_n:
                        break
            knowledge_ids = unique_ids
    else:
        raise ValueError(f"Unknown method: {method}. Use 'graph' or 'chunks'.")
    
    # Return exactly target_n (or all if less than target_n)
    unique_ids = list(dict.fromkeys(knowledge_ids))  # Preserve order, remove duplicates
    return unique_ids[:target_n]
