"""
Report generator for evaluation results.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

from .evaluator import AggregatedResults, QueryResults
from .config import EvaluationConfig
from cognee.shared.logging_utils import get_logger

logger = get_logger("ReportGenerator")


def generate_console_summary(aggregated_results: AggregatedResults) -> None:
    """
    Print console summary of evaluation results.
    
    Args:
        aggregated_results: AggregatedResults object
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total queries: {aggregated_results.total_queries}")
    print("\nGraph Completion Metrics:")
    for key, value in aggregated_results.graph_completion_avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nChunk Metrics:")
    for key, value in aggregated_results.chunk_avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)


def generate_csv_report(
    aggregated_results: AggregatedResults,
    output_path: str,
    config: EvaluationConfig
) -> None:
    """
    Generate CSV report with per-query results.
    
    Args:
        aggregated_results: AggregatedResults object
        output_path: Path to output CSV file
        config: Evaluation configuration
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV rows
    rows = []
    for query_result in aggregated_results.per_query_results:
        # Graph Completion row
        row_gc = {
            "query": query_result.query,
            "ground_truth_ids": ",".join(query_result.ground_truth_ids),
            "method": "graph_completion",
            "retrieved_ids_top10": ",".join(query_result.graph_completion_retrieved_ids),
            "hit_at_5": query_result.graph_completion_metrics["hit_at_5"],
            "hit_at_10": query_result.graph_completion_metrics["hit_at_10"],
            "hit_rate_at_5": query_result.graph_completion_metrics["hit_rate_at_5"],
            "hit_rate_at_10": query_result.graph_completion_metrics["hit_rate_at_10"],
            "precision_at_5": query_result.graph_completion_metrics["precision_at_5"],
            "precision_at_10": query_result.graph_completion_metrics["precision_at_10"],
            "recall_at_5": query_result.graph_completion_metrics["recall_at_5"],
            "recall_at_10": query_result.graph_completion_metrics["recall_at_10"],
        }
        # Add metadata columns
        if query_result.metadata:
            row_gc.update(query_result.metadata)
        rows.append(row_gc)
        
        # Chunk row
        row_chunk = {
            "query": query_result.query,
            "ground_truth_ids": ",".join(query_result.ground_truth_ids),
            "method": "chunk",
            "retrieved_ids_top10": ",".join(query_result.chunk_retrieved_ids),
            "hit_at_5": query_result.chunk_metrics["hit_at_5"],
            "hit_at_10": query_result.chunk_metrics["hit_at_10"],
            "hit_rate_at_5": query_result.chunk_metrics["hit_rate_at_5"],
            "hit_rate_at_10": query_result.chunk_metrics["hit_rate_at_10"],
            "precision_at_5": query_result.chunk_metrics["precision_at_5"],
            "precision_at_10": query_result.chunk_metrics["precision_at_10"],
            "recall_at_5": query_result.chunk_metrics["recall_at_5"],
            "recall_at_10": query_result.chunk_metrics["recall_at_10"],
        }
        # Add metadata columns
        if query_result.metadata:
            row_chunk.update(query_result.metadata)
        rows.append(row_chunk)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV report saved to: {output_file}")


def generate_context_output(
    aggregated_results: AggregatedResults,
    output_path: str
) -> None:
    """
    Generate context output (reranked chunks) for LLM consumption.
    
    Args:
        aggregated_results: AggregatedResults object
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    context_data = []
    for query_result in aggregated_results.per_query_results:
        if query_result.reranked_chunks:
            context_data.append({
                "query": query_result.query,
                "knowledge_ids": query_result.graph_completion_retrieved_ids,
                "reranked_chunks": query_result.reranked_chunks
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Context output saved to: {output_path}")


def generate_json_report(
    aggregated_results: AggregatedResults,
    output_path: str
) -> None:
    """
    Generate JSON report with full results.
    
    Args:
        aggregated_results: AggregatedResults object
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict (handle dataclasses)
    report_data = {
        "total_queries": aggregated_results.total_queries,
        "graph_completion_avg_metrics": aggregated_results.graph_completion_avg_metrics,
        "chunk_avg_metrics": aggregated_results.chunk_avg_metrics,
        "per_query_results": [
            {
                "query": r.query,
                "ground_truth_ids": r.ground_truth_ids,
                "graph_completion_retrieved_ids": r.graph_completion_retrieved_ids,
                "chunk_retrieved_ids": r.chunk_retrieved_ids,
                "graph_completion_metrics": r.graph_completion_metrics,
                "chunk_metrics": r.chunk_metrics,
                "metadata": r.metadata or {}
            }
            for r in aggregated_results.per_query_results
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON report saved to: {output_file}")


def generate_reports(
    aggregated_results: AggregatedResults,
    config: EvaluationConfig
) -> None:
    """
    Generate all reports.
    
    Args:
        aggregated_results: AggregatedResults object
        config: Evaluation configuration
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Console summary
    generate_console_summary(aggregated_results)
    
    # CSV report
    csv_path = output_dir / "evaluation_results.csv"
    generate_csv_report(aggregated_results, str(csv_path), config)
    
    # JSON report
    json_path = output_dir / "evaluation_results.json"
    generate_json_report(aggregated_results, str(json_path))
    
    # Context output (if enabled)
    if config.generate_context_output:
        context_path = output_dir / config.context_output_file
        generate_context_output(aggregated_results, str(context_path))
