"""
Compatibility script to run retrieval evaluation using the new framework
but ensuring input/output compatibility with testing/eval_retrieval_v2.py.
"""

import argparse
import asyncio
import ast
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Any

import pandas as pd
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

# Import new framework components
from examples.python.retrieval_evaluation.simple_search import search_knowledge_ids

# Use cognee logger or setup local logger
logger = logging.getLogger("eval_v2_compat")

def setup_logging(log_file: str) -> logging.Logger:
    """Setup file logging for retrieval evaluation."""
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# ==============================================================================
# Helper functions migrated from testing/eval_retrieval_v2.py
# ==============================================================================

def parse_list(value) -> List[str]:
    """Parse list from CSV cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []

    value = value.replace("\n", "").replace("\r", "").strip()
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


def normalize_doc_name(name: str) -> str:
    """Normalize doc name to use double-underscore prefix if applicable."""
    if "__" in name:
        return name

    known_prefixes = ("helpdesk", "wi", "produk", "promo", "program")
    for prefix in known_prefixes:
        marker = f"{prefix}_"
        if name.startswith(marker):
            return f"{prefix}__{name[len(marker):]}"

    return name

# ==============================================================================
# Main Evaluation Logic
# ==============================================================================

async def evaluate(
    input_csv: str, output_csv: str, log_file: str, top_k: int, triplets_json: str
) -> None:
    # Setup logging
    logger = setup_logging(log_file)
    
    # We use initial_top_k=50 as default to cast a wider net, then expand/contract
    initial_top_k = max(100, top_k * 2) 

    logger.info("=" * 60)
    logger.info("Starting GRAPH_COMPLETION Retrieval Evaluation (COMPATIBILITY MODE)")
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Triplets JSON: {triplets_json}")
    logger.info(f"Target Unique IDs (Top-K): {top_k}")
    logger.info(f"Initial Search Top-K: {initial_top_k}")
    logger.info("Using search_knowledge_ids() directly")
    logger.info("=" * 60)

    # Read CSV
    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if "case_type" in df.columns:
        df = df[~df["case_type"].fillna("").str.lower().eq("negative")]
    
    total_cases = len(df)
    
    # Prepare output columns
    columns = ["question", "id_ground_truth"] + [
        f"context_{i + 1}" for i in range(top_k)
    ] + ["id_eval"]
    
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    hit_count = 0
    triplets_dump: List[Dict[str, object]] = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        start_time = datetime.now()
        question = getattr(row, "question", "")
        question_preview = question[:80] + "..." if len(question) > 80 else question
        logger.info(f"[{idx}/{total_cases}] Processing: {question_preview}")

        # -------------------------------------------------------------
        # 1. Parse Ground Truth (Matching v2 logic)
        # -------------------------------------------------------------
        gt_prefixed = (
            parse_list(getattr(row, "id_gt_categories", ""))
            if hasattr(row, "id_gt_categories")
            else []
        )
        if gt_prefixed:
            ground_truth = gt_prefixed
        else:
            context_ids = parse_list(getattr(row, "context_ground_truth", ""))
            categories = parse_list(getattr(row, "gt_categories", ""))
            ground_truth = build_prefixed_ground_truth(context_ids, categories)
            
        normalized_ground_truth = [normalize_doc_name(doc_id) for doc_id in ground_truth]

        # -------------------------------------------------------------
        # 2. Execute Search using search_knowledge_ids directly
        # -------------------------------------------------------------
        try:
            # Direct call to the simplified search function
            retrieved_doc_names = await search_knowledge_ids(
                question=question,
                method="graph",
                target_n=top_k,
                initial_top_k=initial_top_k
            )
            
            # -------------------------------------------------------------
            # 3. Triplets Dump Placeholder
            # -------------------------------------------------------------
            # search_knowledge_ids does not return triplets, so we add a placeholder.
            triplets_dump.append(
                {
                    "question": question,
                    "source": "search_knowledge_ids",
                    "triplet_count": -1, # Indieates not available
                    "triplets": [], 
                    "note": "Triplets not available when using simplified search_knowledge_ids wrapper"
                }
            )

            # -------------------------------------------------------------
            # 4. Evaluate (HIT/MISS)
            # -------------------------------------------------------------
            doc_names_normalized = [normalize_doc_name(doc_name) for doc_name in retrieved_doc_names]
            
            id_eval = bool(normalized_ground_truth) and any(
                doc_id in normalized_ground_truth for doc_id in doc_names_normalized
            )
            
            if id_eval:
                hit_count += 1

            elapsed = (datetime.now() - start_time).total_seconds()
            status = "HIT" if id_eval else "MISS"
            logger.info(
                f"[{idx}/{total_cases}] {status} | "
                f"Docs (Top {len(retrieved_doc_names)}): {retrieved_doc_names} | "
                f"Time: {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"[{idx}/{total_cases}] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            triplets_dump.append(
                {
                    "question": question,
                    "source": "error",
                    "error": str(e),
                    "triplet_count": 0,
                    "triplets": [],
                }
            )
            retrieved_doc_names = []
            id_eval = False

        # -------------------------------------------------------------
        # 5. Write to Output
        # -------------------------------------------------------------
        record = {
            "question": question,
            "id_ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            "id_eval": id_eval,
        }
        for i in range(top_k):
            record[f"context_{i + 1}"] = retrieved_doc_names[i] if i < len(retrieved_doc_names) else ""

        record_df = pd.DataFrame([record], columns=columns)
        record_df.to_csv(output_csv, mode="a", header=write_header, index=False)
        write_header = False

        accuracy = (hit_count / idx) * 100
        logger.info(f"[{idx}/{total_cases}] Running accuracy: {hit_count}/{idx} = {accuracy:.1f}%")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Total: {total_cases} | Hits: {hit_count} | Accuracy: {(hit_count/total_cases)*100:.1f}%")
    logger.info(f"Results saved to: {output_csv}")
    logger.info("=" * 60)

    with open(triplets_json, "w", encoding="utf-8") as handle:
        json.dump(triplets_dump, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GRAPH_COMPLETION retrieval context against ground-truth CSV (Compatibility Mode)."
    )
    # Default values match what the user might be used to
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="eval_output.csv", help="Output CSV file path")
    parser.add_argument("--log", default="eval.log", help="Log file path")
    parser.add_argument(
        "--triplets-json", default="eval_triplets.json", help="Triplets dump JSON path"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of unique knowledge IDs to retrieve")
    
    args = parser.parse_args()

    # Ensure output directory exists if path has directory components
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    asyncio.run(
        evaluate(args.input, args.output, args.log, args.top_k, args.triplets_json)
    )


if __name__ == "__main__":
    main()
