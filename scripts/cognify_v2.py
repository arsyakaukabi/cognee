#!/usr/bin/env python3
"""
Cognify script using extract_graph_from_data_v2 (cascade extraction).

This script runs the full cognify pipeline with the V2 graph extraction
which uses a multi-step cascade approach.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Disable authentication before importing cognee
os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

# Add cognee to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cognee
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph.extract_graph_from_data_v2 import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.cognify.config import get_cognify_config


async def get_v2_cognify_tasks(
    chunk_size: Optional[int] = None,
    chunks_per_batch: int = 10,
    n_rounds: int = 2,
):
    """
    Build cognify tasks using V2 graph extraction (cascade extraction).
    
    Args:
        chunk_size: Maximum chunk size in tokens (auto-calculated if None)
        chunks_per_batch: Number of chunks to process per batch
        n_rounds: Number of extraction rounds for cascade extraction
    
    Returns:
        List of Task objects for the cognify pipeline
    """
    cognify_config = get_cognify_config()
    
    tasks = [
        Task(classify_documents),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=TextChunker,
        ),
        Task(
            extract_graph_from_data,
            n_rounds=n_rounds,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(
            summarize_text,
            summarization_model=cognify_config.summarization_model,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(
            add_data_points,
            task_config={"batch_size": chunks_per_batch},
        ),
    ]
    
    return tasks


async def cognify_v2(
    datasets: list[str] = None,
    chunk_size: Optional[int] = None,
    chunks_per_batch: int = 10,
    n_rounds: int = 2,
):
    """
    Run cognify with V2 graph extraction.
    
    Args:
        datasets: List of dataset names to process (None = all)
        chunk_size: Maximum chunk size in tokens
        chunks_per_batch: Chunks per batch for LLM processing
        n_rounds: Number of cascade extraction rounds
    """
    print("=" * 60)
    print("Cognify V2 (Cascade Graph Extraction)")
    print("=" * 60)
    print(f"Datasets:         {datasets or 'all'}")
    print(f"Chunk Size:       {chunk_size or 'auto'}")
    print(f"Chunks per Batch: {chunks_per_batch}")
    print(f"Extraction Rounds: {n_rounds}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    tasks = await get_v2_cognify_tasks(
        chunk_size=chunk_size,
        chunks_per_batch=chunks_per_batch,
        n_rounds=n_rounds,
    )
    
    print(f"\nPipeline tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task.run.__name__}")
    
    print("\nStarting cognify pipeline...")
    
    results = []
    try:
        async for run_info in run_pipeline(
            tasks=tasks,
            datasets=datasets,
        ):
            print(f"  Pipeline progress: {run_info}")
            results.append(run_info)
        
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("COGNIFY COMPLETED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Results: {len(results)} pipeline runs")
        
        return results
        
    except Exception as e:
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("COGNIFY FAILED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Cognify with V2 cascade graph extraction"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=None,
        help="Dataset names to process (default: all)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=None,
        help="Maximum chunk size in tokens (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Chunks per batch for LLM processing (default: 10)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=2,
        help="Number of cascade extraction rounds (default: 2)"
    )
    
    args = parser.parse_args()
    
    await cognify_v2(
        datasets=args.datasets,
        chunk_size=args.chunk_size,
        chunks_per_batch=args.batch_size,
        n_rounds=args.rounds,
    )


if __name__ == "__main__":
    asyncio.run(main())
