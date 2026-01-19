#!/usr/bin/env python3
"""
Evaluate retrieval results using SearchType.CHUNKS against ground-truth CSV.
Similar to eval_retrieval.py but uses CHUNKS search instead of GraphCompletionRetriever.
"""

# uv run python testing/eval_chunks.py --input testing/data_test_new.csv --output testing/eval_chunks_output.csv --log testing/eval_chunks.log --top-k 15
# tail -n 200 -F /home/usr_00345779_hq_bri_co_id/cognee/testing/eval_chunks.log
import argparse
import asyncio
import ast
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

import cognee
from cognee.modules.search.types import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine


def setup_logging(log_file: str) -> logging.Logger:
    """Setup file logging for retrieval evaluation."""
    logger = logging.getLogger("eval_chunks")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


async def get_document_name_from_chunk(graph_engine, chunk_id: str) -> Optional[str]:
    """Get the TextDocument name that a chunk belongs to via is_part_of relationship."""
    try:
        connections = await graph_engine.get_connections(str(chunk_id))
        for source, relationship, target in connections:
            relationship_name = relationship.get("relationship_name")
            if relationship_name != "is_part_of":
                continue
            if target.get("type") == "TextDocument":
                return target.get("name")
            if source.get("type") == "TextDocument":
                return source.get("name")
    except Exception:
        pass
    return None


async def extract_document_names_from_chunks(
    chunks: List, top_k: int, graph_engine=None
) -> List[str]:
    """Extract unique TextDocument names from chunk search results."""
    if not chunks:
        return []
    
    graph_engine = graph_engine or await get_graph_engine()
    doc_scores: Dict[str, float] = {}
    cache: Dict[str, Optional[str]] = {}
    
    for i, chunk in enumerate(chunks):
        # Extract chunk ID from result
        chunk_id = None
        score = i  # Use index as score (lower = better, earlier in results)
        
        if hasattr(chunk, 'id'):
            chunk_id = str(chunk.id)
        elif isinstance(chunk, dict):
            chunk_id = str(chunk.get('id', chunk.get('chunk_id', '')))
            # Try to get document name directly if available
            doc_name = chunk.get('document_name') or chunk.get('text_document_name')
            if doc_name:
                if doc_name not in doc_scores or score < doc_scores[doc_name]:
                    doc_scores[doc_name] = score
                continue
        
        if not chunk_id:
            continue
            
        # Look up document name from graph
        if chunk_id not in cache:
            cache[chunk_id] = await get_document_name_from_chunk(graph_engine, chunk_id)
        
        name = cache[chunk_id]
        if name and (name not in doc_scores or score < doc_scores[name]):
            doc_scores[name] = score
    
    # Sort by score (lower = better)
    ranked = sorted(doc_scores.items(), key=lambda x: x[1])
    return [name for name, _ in ranked[:top_k]]


def parse_list(value) -> List[str]:
    """Parse list from CSV cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    
    value = value.replace('\n', '').replace('\r', '').strip()
    if not value:
        return []
    
    # Try JSON
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    
    # Try Python literal
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    
    # Handle weird format: ['val1'], ['val2']
    bracket_pattern = r"\['([^']+)'\]"
    matches = re.findall(bracket_pattern, value)
    if matches:
        return matches
    
    return []


def normalize_category_prefix(category: str) -> str:
    """Map category name to its standard prefix."""
    category_lower = category.strip().lower()
    mapping = {
        "working instruction": "wi",
        "wi": "wi",
        "helpdesk": "helpdesk",
        "produk": "produk",
        "product": "produk",
        "promo": "promo",
        "program": "program",
    }
    return mapping.get(category_lower, category_lower)


def build_prefixed_ground_truth(context_ids: List[str], categories: List[str]) -> List[str]:
    """Build prefixed ground truth by pairing doc_ids with categories by index."""
    if not context_ids:
        return []
    if not categories:
        return context_ids

    prefixed = []
    for i, doc_id in enumerate(context_ids):
        if "__" in doc_id:
            prefixed.append(doc_id)
            continue
        
        if i < len(categories):
            category = categories[i]
        else:
            category = categories[0] if categories else ""
        
        if category:
            prefix = normalize_category_prefix(category)
            prefixed.append(f"{prefix}__{doc_id}")
        else:
            prefixed.append(doc_id)
    
    return prefixed


async def evaluate(input_csv: str, output_csv: str, log_file: str, top_k: int) -> None:
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Starting CHUNKS Retrieval Evaluation")
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Top-K: {top_k}")
    logger.info("=" * 60)
    
    df = pd.read_csv(input_csv)
    if "case_type" in df.columns:
        df = df[~df["case_type"].fillna("").str.lower().eq("negative")]
    
    graph_engine = await get_graph_engine()
    total_cases = len(df)
    
    logger.info(f"Total cases to process: {total_cases}")
    
    # Define columns for CSV
    columns = ["question", "id_ground_truth"] + [
        f"context_{i + 1}" for i in range(top_k)
    ] + ["id_eval"]
    
    # Write header if file doesn't exist or is empty
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    
    hit_count = 0
    
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        start_time = datetime.now()
        question = getattr(row, "question", "")
        question_preview = question[:80] + "..." if len(question) > 80 else question
        
        logger.info(f"[{idx}/{total_cases}] Processing: {question_preview}")
        
        # Parse ground truth
        gt_prefixed = parse_list(getattr(row, "id_gt_categories", "")) if hasattr(row, "id_gt_categories") else []
        if gt_prefixed:
            ground_truth = gt_prefixed
        else:
            context_ids = parse_list(getattr(row, "context_ground_truth", ""))
            categories = parse_list(getattr(row, "gt_categories", ""))
            ground_truth = build_prefixed_ground_truth(context_ids, categories)

        try:
            # Use cognee.search with SearchType.CHUNKS
            chunks = await cognee.search(
                query_type=SearchType.CHUNKS,
                query_text=question
            )
            
            doc_names = []
            if chunks:
                doc_names = await extract_document_names_from_chunks(
                    chunks, top_k=top_k, graph_engine=graph_engine
                )
            
            id_eval = bool(ground_truth) and any(doc_id in ground_truth for doc_id in doc_names)
            if id_eval:
                hit_count += 1
            
            elapsed = (datetime.now() - start_time).total_seconds()
            status = "HIT" if id_eval else "MISS"
            logger.info(f"[{idx}/{total_cases}] {status} | Chunks: {len(chunks) if chunks else 0} | Docs: {doc_names[:3]} | Time: {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"[{idx}/{total_cases}] ERROR: {str(e)}")
            doc_names = []
            id_eval = False

        # Build record
        record = {
            "question": question,
            "id_ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            "id_eval": id_eval,
        }
        for i in range(top_k):
            record[f"context_{i + 1}"] = doc_names[i] if i < len(doc_names) else ""
        
        # Append to CSV immediately
        record_df = pd.DataFrame([record], columns=columns)
        record_df.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False
        
        # Log running accuracy
        accuracy = (hit_count / idx) * 100
        logger.info(f"[{idx}/{total_cases}] Running accuracy: {hit_count}/{idx} = {accuracy:.1f}%")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Total: {total_cases} | Hits: {hit_count} | Accuracy: {(hit_count/total_cases)*100:.1f}%")
    logger.info(f"Results saved to: {output_csv}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CHUNKS retrieval results against ground-truth CSV."
    )
    parser.add_argument("--input", default="testing/data_test_new.csv")
    parser.add_argument("--output", default="testing/eval_chunks_output.csv")
    parser.add_argument("--log", default="testing/eval_chunks.log")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(evaluate(args.input, args.output, args.log, args.top_k))


if __name__ == "__main__":
    main()
