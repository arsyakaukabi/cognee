"""
Expand search results until we have exactly N unique knowledge IDs.
"""

from typing import List, Literal, Any, TYPE_CHECKING
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.shared.logging_utils import get_logger
from .result_mapper import map_triplet_results_to_knowledge_ids, map_chunk_results_to_knowledge_ids

if TYPE_CHECKING:
    from .search_executor import SearchExecutor

logger = get_logger("ResultExpander")


async def expand_graph_completion_to_n_ids(
    triplets: List[Edge],
    target_n: int,
    graph_engine: GraphDBInterface,
    search_executor: "SearchExecutor",
    query: str,
    initial_top_k: int = 10,
    max_top_k: int = 200
) -> List[str]:
    """
    Expand Graph Completion results until we have exactly N unique knowledge IDs.
    
    Strategy:
    1. Map initial triplets to knowledge IDs
    2. If not enough, increase top_k and fetch more triplets
    3. Continue until we have >= target_n unique knowledge IDs
    4. Return exactly target_n (or all if we couldn't reach target_n)
    
    Args:
        triplets: Initial list of triplets from search
        target_n: Target number of unique knowledge IDs (e.g., 10)
        graph_engine: Graph database interface
        search_executor: SearchExecutor instance for fetching more results
        query: Original query string
        initial_top_k: Initial search top_k
        max_top_k: Maximum top_k for expansion
        
    Returns:
        Exactly N unique knowledge IDs (ordered by retrieval rank)
    """
    # Map initial triplets to knowledge IDs
    knowledge_ids = await map_triplet_results_to_knowledge_ids(triplets, graph_engine, target_n=target_n)
    unique_ids = list(dict.fromkeys(knowledge_ids))  # Preserve order, remove duplicates
    
    logger.info(f"Initial mapping: {len(triplets)} triplets → {len(unique_ids)} unique knowledge IDs (target: {target_n})")
    
    if len(unique_ids) >= target_n:
        logger.info(f"✅ Already have {len(unique_ids)} unique knowledge IDs, returning top {target_n}")
        return unique_ids[:target_n]
    
    # Need to expand: increase top_k and fetch more triplets
    # Start with larger increments since initial_top_k is already high (50)
    current_top_k = initial_top_k
    expansion_factor = 1.5  # Smaller factor since we start high: 50 → 75 → 112 → 168 → 200
    max_iterations = 5  # Reduced since we start with higher initial_top_k
    iteration = 0
    
    while len(unique_ids) < target_n and current_top_k < max_top_k and iteration < max_iterations:
        iteration += 1
        
        # Increase top_k (smaller factor since initial_top_k is already high)
        previous_top_k = current_top_k
        current_top_k = min(int(current_top_k * expansion_factor), max_top_k)
        
        logger.info(f"Expanding Graph Completion search (iteration {iteration}): "
                   f"top_k={previous_top_k} → {current_top_k}, "
                   f"current_unique_ids={len(unique_ids)}, target={target_n}")
        
        # Fetch more triplets with increased top_k
        new_triplets = await search_executor.execute_graph_completion_search(query, top_k=current_top_k)
        
        if not new_triplets:
            logger.warning(f"No more triplets found at top_k={current_top_k}, stopping expansion")
            break
        
        # Map new triplets to knowledge IDs
        # Note: We can't use simple target_n here because we need to count *new* unique IDs against the *total* target
        # But we can optimize by only mapping until we potentially fill the gap
        remaining_needed = target_n - len(unique_ids)
        new_knowledge_ids = await map_triplet_results_to_knowledge_ids(new_triplets, graph_engine, target_n=None)
        
        # Count new unique IDs before adding
        new_unique_count = sum(1 for kid in new_knowledge_ids if kid not in unique_ids)
        
        # Add new unique IDs (preserve order)
        for kid in new_knowledge_ids:
            if kid not in unique_ids:
                unique_ids.append(kid)
        
        logger.info(f"   → Retrieved {len(new_triplets)} triplets → {len(new_knowledge_ids)} knowledge IDs "
                   f"→ {new_unique_count} new unique IDs → Total: {len(unique_ids)} unique IDs")
        
        # If we have enough, break
        if len(unique_ids) >= target_n:
            logger.info(f"✅ Reached target: {len(unique_ids)} unique knowledge IDs (target: {target_n})")
            break
        
        # If no new unique IDs were added, we're stuck
        if new_unique_count == 0:
            logger.warning(f"No new unique knowledge IDs found at top_k={current_top_k}, stopping expansion")
            break
    
    # Final check
    if len(unique_ids) < target_n:
        logger.warning(f"⚠️  Could only retrieve {len(unique_ids)} unique knowledge IDs (target: {target_n})")
    
    # Return exactly N (or all if we couldn't reach N)
    result = unique_ids[:target_n]
    logger.info(f"Final result: {len(result)} unique knowledge IDs")
    return result


async def expand_chunk_search_to_n_ids(
    chunk_results: List[Any],
    target_n: int,
    graph_engine: GraphDBInterface,
    search_executor: "SearchExecutor",
    query: str,
    initial_top_k: int = 10,
    max_top_k: int = 200
) -> List[str]:
    """
    Expand Chunk search results until we have exactly N unique knowledge IDs.
    
    Strategy:
    1. Map initial chunks to knowledge IDs
    2. If not enough, increase top_k and fetch more chunks
    3. Continue until we have >= target_n unique knowledge IDs
    4. Return exactly target_n (or all if we couldn't reach target_n)
    
    Args:
        chunk_results: Initial list of chunk results from search
        target_n: Target number of unique knowledge IDs (e.g., 10)
        graph_engine: Graph database interface
        search_executor: SearchExecutor instance for fetching more results
        query: Original query string
        initial_top_k: Initial search top_k
        max_top_k: Maximum top_k for expansion
        
    Returns:
        Exactly N unique knowledge IDs (ordered by retrieval rank)
    """
    # Map initial chunks to knowledge IDs
    knowledge_ids = await map_chunk_results_to_knowledge_ids(chunk_results, graph_engine, target_n=target_n)
    unique_ids = list(dict.fromkeys(knowledge_ids))  # Preserve order, remove duplicates
    
    logger.info(f"Initial mapping: {len(chunk_results)} chunks → {len(unique_ids)} unique knowledge IDs (target: {target_n})")
    
    if len(unique_ids) >= target_n:
        logger.info(f"✅ Already have {len(unique_ids)} unique knowledge IDs, returning top {target_n}")
        return unique_ids[:target_n]
    
    # Need to expand: increase top_k and fetch more chunks
    # Start with larger increments since initial_top_k is already high (50)
    current_top_k = initial_top_k
    expansion_factor = 1.5  # Smaller factor since we start high: 50 → 75 → 112 → 168 → 200
    max_iterations = 5  # Reduced since we start with higher initial_top_k
    iteration = 0
    
    while len(unique_ids) < target_n and current_top_k < max_top_k and iteration < max_iterations:
        iteration += 1
        
        # Increase top_k (smaller factor since initial_top_k is already high)
        previous_top_k = current_top_k
        current_top_k = min(int(current_top_k * expansion_factor), max_top_k)
        
        logger.info(f"Expanding Chunk search (iteration {iteration}): "
                   f"top_k={previous_top_k} → {current_top_k}, "
                   f"current_unique_ids={len(unique_ids)}, target={target_n}")
        
        # Fetch more chunks with increased top_k
        new_chunk_results = await search_executor.execute_chunk_search(query, top_k=current_top_k)
        
        if not new_chunk_results:
            logger.warning(f"No more chunks found at top_k={current_top_k}, stopping expansion")
            break
        
        # Map new chunks to knowledge IDs
        remaining_needed = target_n - len(unique_ids)
        new_knowledge_ids = await map_chunk_results_to_knowledge_ids(new_chunk_results, graph_engine, target_n=None)
        
        # Count new unique IDs before adding
        new_unique_count = sum(1 for kid in new_knowledge_ids if kid not in unique_ids)
        
        # Add new unique IDs (preserve order)
        for kid in new_knowledge_ids:
            if kid not in unique_ids:
                unique_ids.append(kid)
        
        logger.info(f"   → Retrieved {len(new_chunk_results)} chunks → {len(new_knowledge_ids)} knowledge IDs "
                   f"→ {new_unique_count} new unique IDs → Total: {len(unique_ids)} unique IDs")
        
        # If we have enough, break
        if len(unique_ids) >= target_n:
            logger.info(f"✅ Reached target: {len(unique_ids)} unique knowledge IDs (target: {target_n})")
            break
        
        # If no new unique IDs were added, we're stuck
        if new_unique_count == 0:
            logger.warning(f"No new unique knowledge IDs found at top_k={current_top_k}, stopping expansion")
            break
    
    # Final check
    if len(unique_ids) < target_n:
        logger.warning(f"⚠️  Could only retrieve {len(unique_ids)} unique knowledge IDs (target: {target_n})")
    
    # Return exactly N (or all if we couldn't reach N)
    result = unique_ids[:target_n]
    logger.info(f"Final result: {len(result)} unique knowledge IDs")
    return result


async def expand_to_n_unique_knowledge_ids(
    search_results: List[Any],
    method: Literal["graph", "chunks"],
    target_n: int,
    graph_engine: GraphDBInterface,
    search_executor: "SearchExecutor",
    query: str,
    initial_top_k: int = 10,
    max_top_k: int = 200
) -> List[str]:
    """
    Expand search results until we have exactly N unique knowledge IDs.
    
    Args:
        search_results: Initial search results (triplets or chunks)
        method: Search method ("graph" or "chunks")
        target_n: Target number of unique knowledge IDs
        graph_engine: Graph database interface
        search_executor: SearchExecutor instance
        query: Original query string
        initial_top_k: Initial search top_k
        max_top_k: Maximum top_k for expansion
        
    Returns:
        Exactly N unique knowledge IDs (ordered by retrieval rank)
    """
    if method == "graph":
        return await expand_graph_completion_to_n_ids(
            search_results, target_n, graph_engine, search_executor, query, initial_top_k, max_top_k
        )
    elif method == "chunks":
        return await expand_chunk_search_to_n_ids(
            search_results, target_n, graph_engine, search_executor, query, initial_top_k, max_top_k
        )
    else:
        raise ValueError(f"Unknown method: {method}")
