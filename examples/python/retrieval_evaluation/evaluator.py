"""
Main evaluator orchestrator for retrieval evaluation.
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.shared.logging_utils import get_logger

from .config import EvaluationConfig
from .csv_loader import load_evaluation_data
from .document_manager import add_documents, cognify_documents, verify_graph_ready
from .search_executor import SearchExecutor
from .result_expander import expand_to_n_unique_knowledge_ids
from .chunk_extractor import extract_chunks_from_knowledge_ids
from .chunk_reranker import rerank_chunks_by_similarity
from .metrics_calculator import calculate_metrics_for_query

logger = get_logger("Evaluator")


@dataclass
class QueryResults:
    """Results for a single query."""
    query: str
    ground_truth_ids: List[str]
    graph_completion_retrieved_ids: List[str]  # Exactly N, ordered
    chunk_retrieved_ids: List[str]  # Exactly N, ordered
    graph_completion_metrics: Dict[str, float]
    chunk_metrics: Dict[str, float]
    # Context (for LLM, not evaluation)
    reranked_chunks: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


@dataclass
class AggregatedResults:
    """Aggregated results across all queries."""
    total_queries: int
    graph_completion_avg_metrics: Dict[str, float]
    chunk_avg_metrics: Dict[str, float]
    per_query_results: List[QueryResults]


async def evaluate_single_query(
    query: str,
    ground_truth_ids: List[str],
    config: EvaluationConfig,
    search_executor: SearchExecutor,
    graph_engine,
    vector_engine
) -> QueryResults:
    """
    Evaluate a single query.
    
    Args:
        query: Query text
        ground_truth_ids: Ground truth knowledge IDs
        config: Evaluation configuration
        search_executor: SearchExecutor instance
        graph_engine: Graph database interface
        vector_engine: Vector database interface
        
    Returns:
        QueryResults object
    """
    logger.info(f"Evaluating query: '{query[:50]}...'")
    
    # Execute Graph Completion search
    gc_triplets = await search_executor.execute_graph_completion_search(
        query, top_k=config.graph_completion_initial_top_k
    )
    
    # Expand to N unique knowledge IDs
    gc_knowledge_ids = await expand_to_n_unique_knowledge_ids(
        gc_triplets,
        method="graph",
        target_n=config.target_n_unique_ids,
        graph_engine=graph_engine,
        search_executor=search_executor,
        query=query,
        initial_top_k=config.graph_completion_initial_top_k,
        max_top_k=config.max_expansion_top_k
    )
    
    # Execute Chunk search
    chunk_results = await search_executor.execute_chunk_search(
        query, top_k=config.chunk_initial_top_k
    )
    
    # Expand to N unique knowledge IDs
    chunk_knowledge_ids = await expand_to_n_unique_knowledge_ids(
        chunk_results,
        method="chunks",
        target_n=config.target_n_unique_ids,
        graph_engine=graph_engine,
        search_executor=search_executor,
        query=query,
        initial_top_k=config.chunk_initial_top_k,
        max_top_k=config.max_expansion_top_k
    )
    
    # Calculate metrics
    gc_metrics = calculate_metrics_for_query(gc_knowledge_ids, ground_truth_ids)
    chunk_metrics = calculate_metrics_for_query(chunk_knowledge_ids, ground_truth_ids)
    
    # Generate context (reranked chunks) if enabled
    reranked_chunks = None
    if config.generate_context_output:
        # Extract all chunks from top-N knowledge IDs (use Graph Completion results)
        context_chunks = await extract_chunks_from_knowledge_ids(gc_knowledge_ids, graph_engine)
        
        # Rerank chunks by similarity
        reranked_chunks = await rerank_chunks_by_similarity(context_chunks, query)
    
    return QueryResults(
        query=query,
        ground_truth_ids=ground_truth_ids,
        graph_completion_retrieved_ids=gc_knowledge_ids,
        chunk_retrieved_ids=chunk_knowledge_ids,
        graph_completion_metrics=gc_metrics,
        chunk_metrics=chunk_metrics,
        reranked_chunks=reranked_chunks
    )


def aggregate_results(all_query_results: List[QueryResults]) -> AggregatedResults:
    """
    Aggregate results across all queries.
    
    Args:
        all_query_results: List of QueryResults for all queries
        
    Returns:
        AggregatedResults object
    """
    if not all_query_results:
        return AggregatedResults(
            total_queries=0,
            graph_completion_avg_metrics={},
            chunk_avg_metrics={},
            per_query_results=[]
        )
    
    # Calculate average metrics
    gc_metric_keys = list(all_query_results[0].graph_completion_metrics.keys())
    chunk_metric_keys = list(all_query_results[0].chunk_metrics.keys())
    
    gc_avg_metrics = {
        key: sum(r.graph_completion_metrics[key] for r in all_query_results) / len(all_query_results)
        for key in gc_metric_keys
    }
    
    chunk_avg_metrics = {
        key: sum(r.chunk_metrics[key] for r in all_query_results) / len(all_query_results)
        for key in chunk_metric_keys
    }
    
    return AggregatedResults(
        total_queries=len(all_query_results),
        graph_completion_avg_metrics=gc_avg_metrics,
        chunk_avg_metrics=chunk_avg_metrics,
        per_query_results=all_query_results
    )


async def run_evaluation(config: EvaluationConfig) -> AggregatedResults:
    """
    Run the complete evaluation.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        AggregatedResults object
    """
    logger.info("=" * 80)
    logger.info("STARTING RETRIEVAL EVALUATION")
    logger.info("=" * 80)
    
    # Load CSV data
    logger.info(f"Loading evaluation data from: {config.csv_path}")
    evaluation_rows = load_evaluation_data(
        config.csv_path,
        question_column=config.question_column,
        gt_column=config.gt_column
    )
    logger.info(f"Loaded {len(evaluation_rows)} evaluation rows")
    
    # Setup documents (if not in read-only mode)
    if not config.read_only_mode:
        if config.document_paths:
            logger.info("Adding documents...")
            await add_documents(config.document_paths)
            
            logger.info("Cognifying documents...")
            await cognify_documents()
        else:
            logger.info("No document paths provided, assuming documents already added")
            if not await verify_graph_ready():
                raise RuntimeError("Graph is not ready and no documents provided to add")
    else:
        logger.info("Read-only mode: skipping document add/cognify")
        if not await verify_graph_ready():
            raise RuntimeError("Graph is not ready in read-only mode")
    
    # Initialize engines
    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()
    # Initialize search executor with higher initial_top_k to reduce expansion iterations
    search_executor = SearchExecutor(initial_top_k=config.graph_completion_initial_top_k)
    
    # Evaluate each query
    all_query_results = []
    for idx, row in enumerate(evaluation_rows, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing query {idx}/{len(evaluation_rows)}")
        logger.info(f"{'=' * 80}")
        
        try:
            query_results = await evaluate_single_query(
                query=row.question,
                ground_truth_ids=row.ground_truth_ids,
                config=config,
                search_executor=search_executor,
                graph_engine=graph_engine,
                vector_engine=vector_engine
            )
            query_results.metadata = row.metadata
            all_query_results.append(query_results)
        except Exception as e:
            logger.error(f"Error evaluating query {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    aggregated_results = aggregate_results(all_query_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total queries evaluated: {aggregated_results.total_queries}")
    logger.info(f"Graph Completion avg Hit@10: {aggregated_results.graph_completion_avg_metrics.get('hit_at_10', 0):.3f}")
    logger.info(f"Chunk avg Hit@10: {aggregated_results.chunk_avg_metrics.get('hit_at_10', 0):.3f}")
    
    return aggregated_results
