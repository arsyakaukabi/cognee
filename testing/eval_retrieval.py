import argparse
import asyncio
import ast
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


# cd /home/usr_00345779_hq_bri_co_id/cognee
# uv run python testing/eval_retrieval.py --input testing/data_test_new.csv --output testing/eval_retrieval_new_output.csv --log testing/eval_retrieval_new.log --top-k 15
# tail -n 200 -F /home/usr_00345779_hq_bri_co_id/cognee/testing/eval_retrieval_new.log
def setup_logging(log_file: str) -> logging.Logger:
    """Setup file logging for retrieval evaluation."""
    logger = logging.getLogger("eval_retrieval")
    logger.setLevel(logging.INFO)
    
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


async def _get_text_document_name_from_chunk(graph_engine, chunk_id: str) -> Optional[str]:
    connections = await graph_engine.get_connections(str(chunk_id))
    for source, relationship, target in connections:
        relationship_name = relationship.get("relationship_name")
        if relationship_name != "is_part_of":
            continue
        if target.get("type") == "TextDocument":
            return target.get("name")
        if source.get("type") == "TextDocument":
            return source.get("name")
    return None


def _triplet_score(edge) -> float:
    # Same scoring used in CogneeGraph.calculate_top_triplet_importances (lower = better).
    n1 = edge.node1.attributes.get("vector_distance", 1)
    n2 = edge.node2.attributes.get("vector_distance", 1)
    e = edge.attributes.get("vector_distance", 1)
    return n1 + n2 + e


async def get_text_document_names_from_triplets(
    triplets, top_k: int, graph_engine=None
) -> List[str]:
    """Return unique TextDocument names ordered by best triplet score."""
    doc_scores: Dict[str, float] = {}
    graph_engine = graph_engine or await get_graph_engine()
    cache: Dict[str, Optional[str]] = {}

    for triplet in triplets:
        score = _triplet_score(triplet)
        for node in (triplet.node1, triplet.node2):
            if node.attributes.get("type") != "DocumentChunk":
                continue
            chunk_id = str(node.id)
            if chunk_id not in cache:
                cache[chunk_id] = await _get_text_document_name_from_chunk(
                    graph_engine, chunk_id
                )
            name = cache[chunk_id]
            if not name:
                continue
            if name not in doc_scores or score < doc_scores[name]:
                doc_scores[name] = score

    ranked: List[Tuple[str, float]] = sorted(doc_scores.items(), key=lambda x: x[1])
    return [name for name, _ in ranked[:top_k]]


def _parse_list(value) -> List[str]:
    """Parse list from CSV cell.
    
    Handles these formats:
    - ["id1", "id2"] - standard JSON
    - ['id1', 'id2'] - Python list literal
    - ['helpdesk'], ['working instruction'] - comma-separated brackets (weird format)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    
    # Clean up: remove newlines, extra whitespace
    value = value.replace('\n', '').replace('\r', '').strip()
    if not value:
        return []
    
    # Try standard JSON first
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    
    # Try Python literal (e.g., ['id1', 'id2'])
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    
    # Handle weird format: ['helpdesk'], ['working instruction']
    # Pattern: multiple bracketed items separated by commas
    import re
    bracket_pattern = r"\['([^']+)'\]"
    matches = re.findall(bracket_pattern, value)
    if matches:
        return matches
    
    return []


def _normalize_category_prefix(category: str) -> str:
    """Map category name to its standard prefix."""
    category_lower = category.strip().lower()
    mapping = {
        "working instruction": "wi",
        "wi": "wi",
        "helpdesk": "helpdesk",
        "produk": "produk",
        "product": "produk",
    }
    return mapping.get(category_lower, category_lower)


def _build_prefixed_ground_truth(context_ids: List[str], categories: List[str]) -> List[str]:
    """Build prefixed ground truth by pairing doc_ids with categories by index.
    
    Examples:
        context_ids: ["id1", "id2"]
        categories: ["helpdesk", "working instruction"]
        result: ["helpdesk__id1", "wi__id2"]
    """
    if not context_ids:
        return []
    if not categories:
        return context_ids

    prefixed = []
    for i, doc_id in enumerate(context_ids):
        # Skip if already prefixed
        if "__" in doc_id:
            prefixed.append(doc_id)
            continue
        
        # Get category by index, or use first category as fallback
        if i < len(categories):
            category = categories[i]
        else:
            category = categories[0] if categories else ""
        
        if category:
            prefix = _normalize_category_prefix(category)
            prefixed.append(f"{prefix}__{doc_id}")
        else:
            prefixed.append(doc_id)
    
    return prefixed


async def evaluate(input_csv: str, output_csv: str, log_file: str, top_k: int) -> None:
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info(f"Starting evaluation run")
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Top-K: {top_k}")
    logger.info("=" * 60)
    
    df = pd.read_csv(input_csv)
    if "case_type" in df.columns:
        df = df[~df["case_type"].fillna("").str.lower().eq("negative")]

    retriever = GraphCompletionRetriever(top_k=top_k)
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
        
        gt_prefixed = _parse_list(getattr(row, "id_gt_categories", ""))
        if gt_prefixed:
            ground_truth = gt_prefixed
        else:
            context_ids = _parse_list(getattr(row, "context_ground_truth", ""))
            categories = _parse_list(getattr(row, "gt_categories", ""))
            ground_truth = _build_prefixed_ground_truth(context_ids, categories)

        try:
            triplets = await retriever.get_context(question)
            doc_names = []
            if triplets:
                doc_names = await get_text_document_names_from_triplets(
                    triplets, top_k=top_k, graph_engine=graph_engine
                )
            
            id_eval = bool(ground_truth) and any(doc_id in ground_truth for doc_id in doc_names)
            if id_eval:
                hit_count += 1
            
            elapsed = (datetime.now() - start_time).total_seconds()
            status = "HIT" if id_eval else "MISS"
            logger.info(f"[{idx}/{total_cases}] {status} | Retrieved: {doc_names[:3]} | Time: {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"[{idx}/{total_cases}] ERROR: {str(e)}")
            doc_names = []
            id_eval = False

        # Build record
        record = {
            "question": question,
            "id_ground_truth": json.dumps(ground_truth, ensure_ascii=True),
            "id_eval": id_eval,
        }
        for i in range(top_k):
            record[f"context_{i + 1}"] = doc_names[i] if i < len(doc_names) else ""
        
        # Append to CSV immediately
        record_df = pd.DataFrame([record], columns=columns)
        record_df.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False  # Only write header once
        
        # Log running accuracy
        accuracy = (hit_count / idx) * 100
        logger.info(f"[{idx}/{total_cases}] Running accuracy: {hit_count}/{idx} = {accuracy:.1f}%")
    
    # Final summary
    logger.info("=" * 60)
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"Total: {total_cases} | Hits: {hit_count} | Accuracy: {(hit_count/total_cases)*100:.1f}%")
    logger.info(f"Results saved to: {output_csv}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results against ground-truth CSV."
    )
    parser.add_argument("--input", default="testing/data_test.csv")
    parser.add_argument("--output", default="testing/eval_output.csv")
    parser.add_argument("--log", default="testing/eval_retrieval.log")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(evaluate(args.input, args.output, args.log, args.top_k))


if __name__ == "__main__":
    main()
