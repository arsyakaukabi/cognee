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
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
try:
    import requests
except Exception:  # pragma: no cover - fallback when requests isn't available
    requests = None

# Import new framework components
from examples.python.retrieval_evaluation.result_mapper import (
    get_textdocument_for_chunk,
    get_documentchunks_for_entity,
    get_entities_for_entitytype,
)

# Use local logger
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


def _normalize_id_knowledge(id_knowledge: Optional[str]) -> str:
    """Normalize idKnowledge value coming from retrieval endpoint."""
    if not id_knowledge:
        return ""
    if id_knowledge.startswith("_") and not id_knowledge.startswith("__"):
        return id_knowledge[1:]
    return id_knowledge


def result_to_doc_name(result: Dict[str, Any]) -> str:
    """Convert retrieval endpoint result to prefixed doc name."""
    if not isinstance(result, dict):
        return ""
    id_knowledge = result.get("idKnowledge") or result.get("id_knowledge") or ""
    knowledge_type = result.get("knowledgeType") or result.get("knowledge_type") or ""
    id_knowledge = _normalize_id_knowledge(str(id_knowledge)) if id_knowledge is not None else ""
    knowledge_type = str(knowledge_type) if knowledge_type is not None else ""
    if knowledge_type:
        return f"{knowledge_type}__{id_knowledge}" if id_knowledge else knowledge_type
    return id_knowledge


def call_retrieval_endpoint(
    api_url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Tuple[int, Dict[str, Any]]:
    """Call retrieval endpoint and return (status_code, response_json)."""
    if requests is None:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            api_url, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            raw = response.read().decode("utf-8")
            return status_code, json.loads(raw) if raw else {}
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    status_code = response.status_code
    response.raise_for_status()
    return status_code, response.json() if response.content else {}

# ==============================================================================
# Triplet Serialization Helpers
# ==============================================================================

def to_jsonable(value: Any) -> Any:
    """Convert values to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def serialize_node(node: Any) -> Dict[str, Any]:
    """Serialize a graph node to a JSON-friendly dict."""
    node_id = getattr(node, "id", None)
    attributes = getattr(node, "attributes", {}) or {}
    return {
        "id": str(node_id) if node_id is not None else None,
        "type": attributes.get("type"),
        "name": attributes.get("name"),
        "attributes": to_jsonable(attributes),
    }


def serialize_edge(edge: Any) -> Dict[str, Any]:
    """Serialize a graph edge to a JSON-friendly dict."""
    attributes = getattr(edge, "attributes", {}) or {}
    return {
        "directed": getattr(edge, "directed", True),
        "relationship_name": attributes.get("relationship_name"),
        "attributes": to_jsonable(attributes),
    }


def serialize_textdocument(doc: Any) -> Dict[str, Any]:
    """Serialize a TextDocument node to a small JSON-friendly dict."""
    if isinstance(doc, dict):
        return {
            "id": str(doc.get("id")) if doc.get("id") is not None else None,
            "name": doc.get("name"),
            "type": doc.get("type", "TextDocument"),
        }
    return {
        "id": str(getattr(doc, "id", None)) if getattr(doc, "id", None) is not None else None,
        "name": getattr(doc, "name", None),
        "type": getattr(doc, "type", "TextDocument"),
    }


def count_triplet_nodes(triplets: List[Any]) -> Dict[str, int]:
    """Count unique node IDs by type within the triplet list."""
    counts = {
        "N_triplets": len(triplets),
        "N_entity": 0,
        "N_entity_type": 0,
        "N_document_chunk": 0,
    }
    seen_entity = set()
    seen_entity_type = set()
    seen_doc_chunk = set()

    for triplet in triplets:
        for node in (getattr(triplet, "node1", None), getattr(triplet, "node2", None)):
            if not node or not hasattr(node, "attributes"):
                continue
            node_id = getattr(node, "id", None)
            node_type = node.attributes.get("type", "")
            if not node_id or not node_type:
                continue
            if node_type == "Entity":
                seen_entity.add(str(node_id))
            elif node_type == "EntityType":
                seen_entity_type.add(str(node_id))
            elif node_type == "DocumentChunk":
                seen_doc_chunk.add(str(node_id))

    counts["N_entity"] = len(seen_entity)
    counts["N_entity_type"] = len(seen_entity_type)
    counts["N_document_chunk"] = len(seen_doc_chunk)
    return counts


async def map_node_to_textdocuments(
    node_id: str, node_type: str, graph_engine, cache: Dict[Tuple[str, str], Dict[str, Any]]
) -> Dict[str, Any]:
    """Map a node to TextDocument nodes, including intermediate path info."""
    cache_key = (node_id, node_type)
    if cache_key in cache:
        return cache[cache_key]

    mapping: Dict[str, Any] = {
        "node_id": node_id,
        "node_type": node_type,
        "knowledge_ids": [],
        "paths": [],
    }

    def add_document(doc: Any, path: Dict[str, Any]) -> None:
        doc_info = serialize_textdocument(doc)
        mapping["paths"].append({**path, "document": doc_info})
        name = doc_info.get("name")
        if name and name not in mapping["knowledge_ids"]:
            mapping["knowledge_ids"].append(name)

    if node_type == "DocumentChunk":
        doc = await get_textdocument_for_chunk(node_id, graph_engine)
        if doc:
            add_document(doc, {"chunk_id": node_id})
    elif node_type == "Entity":
        chunk_ids = await get_documentchunks_for_entity(node_id, graph_engine)
        for chunk_id in chunk_ids:
            doc = await get_textdocument_for_chunk(chunk_id, graph_engine)
            if doc:
                add_document(doc, {"entity_id": node_id, "chunk_id": chunk_id})
    elif node_type == "EntityType":
        entity_ids = await get_entities_for_entitytype(node_id, graph_engine)
        for entity_id in entity_ids:
            chunk_ids = await get_documentchunks_for_entity(entity_id, graph_engine)
            for chunk_id in chunk_ids:
                doc = await get_textdocument_for_chunk(chunk_id, graph_engine)
                if doc:
                    add_document(
                        doc,
                        {"entitytype_id": node_id, "entity_id": entity_id, "chunk_id": chunk_id},
                    )
    else:
        mapping["note"] = "unsupported_node_type"

    cache[cache_key] = mapping
    return mapping


async def serialize_triplets(triplets: List[Any], graph_engine) -> List[Dict[str, Any]]:
    """Serialize triplets with node-to-TextDocument mappings."""
    serialized: List[Dict[str, Any]] = []
    node_mapping_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rank, triplet in enumerate(triplets, start=1):
        node1 = getattr(triplet, "node1", None)
        node2 = getattr(triplet, "node2", None)

        node1_id = str(getattr(node1, "id", "")) if node1 else ""
        node2_id = str(getattr(node2, "id", "")) if node2 else ""
        node1_type = (
            node1.attributes.get("type", "") if node1 and hasattr(node1, "attributes") else ""
        )
        node2_type = (
            node2.attributes.get("type", "") if node2 and hasattr(node2, "attributes") else ""
        )

        serialized.append(
            {
                "rank": rank,
                "node1": serialize_node(node1) if node1 else {},
                "edge": serialize_edge(triplet),
                "node2": serialize_node(node2) if node2 else {},
                "node1_textdocuments": await map_node_to_textdocuments(
                    node1_id, node1_type, graph_engine, node_mapping_cache
                )
                if node1_id and node1_type
                else {},
                "node2_textdocuments": await map_node_to_textdocuments(
                    node2_id, node2_type, graph_engine, node_mapping_cache
                )
                if node2_id and node2_type
                else {},
            }
        )

    return serialized

# ==============================================================================
# Main Evaluation Logic
# ==============================================================================

async def evaluate(
    input_csv: str,
    output_csv: str,
    log_file: str,
    top_k: int,
    triplets_json: str,
    start_index: int,
    api_url: str,
    search_type: str,
    timeout: float,
    debug_trace: bool,
    api_token: str,
) -> None:
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("Starting RETRIEVAL Evaluation via HTTP Endpoint (COMPATIBILITY MODE)")
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Triplets JSON: {triplets_json}")
    logger.info(f"Target Top-K: {top_k}")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Search Type: {search_type}")
    logger.info(f"Debug Trace: {debug_trace}")
    logger.info("Using /api/v1/search/retrieval endpoint")
    logger.info("=" * 60)

    # Read CSV
    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    total_cases_all = len(df)
    if start_index < 1:
        start_index = 1
    if start_index > total_cases_all:
        logger.warning(f"start-index {start_index} is past total rows {total_cases_all}; nothing to do.")
        return
    df = df.iloc[start_index - 1 :]
    total_cases = len(df)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if debug_trace:
        headers["X-Debug-Trace"] = "true"
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Prepare output columns
    columns = ["question", "case_type", "id_ground_truth"] + [
        f"context_{i + 1}" for i in range(top_k)
    ] + ["id_eval", "N_triplets", "N_entity", "N_entity_type", "N_document_chunk", "elapsed_s"]
    
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    hit_count = 0
    elapsed_times: List[float] = []
    triplets_dump: List[Dict[str, object]] = []
    if os.path.exists(triplets_json) and os.path.getsize(triplets_json) > 0 and start_index > 1:
        try:
            with open(triplets_json, "r", encoding="utf-8") as handle:
                existing_dump = json.load(handle)
                if isinstance(existing_dump, list):
                    triplets_dump = existing_dump
        except Exception as exc:
            logger.warning(f"Could not read existing triplets JSON: {exc}")

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        global_idx = start_index + idx - 1
        start_time = datetime.now()
        question = getattr(row, "question", "")
        case_type = getattr(row, "case_type", "")
        is_negative_case = str(case_type).strip().lower() == "negative"
        question_preview = question[:80] + "..." if len(question) > 80 else question
        logger.info(f"[{global_idx}/{total_cases_all}] Processing: {question_preview}")

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
        # 2. Execute Search using retrieval HTTP endpoint
        # -------------------------------------------------------------
        try:
            payload = {
                "query": question,
                "top_k": top_k,
                "search_type": search_type,
            }
            request_start = time.perf_counter()
            status_code, response_json = call_retrieval_endpoint(
                api_url=api_url,
                payload=payload,
                headers=headers,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - request_start

            data = response_json.get("data", []) if isinstance(response_json, dict) else []
            retrieved_doc_names = [
                result_to_doc_name(item) for item in data if item is not None
            ]
            retrieved_doc_names = [name for name in retrieved_doc_names if name]
            if len(retrieved_doc_names) > top_k:
                retrieved_doc_names = retrieved_doc_names[:top_k]

            triplet_counts = {
                "N_triplets": 0,
                "N_entity": 0,
                "N_entity_type": 0,
                "N_document_chunk": 0,
            }
            
            triplets_dump.append(
                {
                    "question": question,
                    "case_type": case_type,
                    "source": "api_retrieval",
                    "payload": payload,
                    "status_code": status_code,
                    "response": response_json,
                    "retrieved_doc_names": retrieved_doc_names,
                    "target_top_k": top_k,
                    "elapsed_s": elapsed,
                }
            )

            # -------------------------------------------------------------
            # 4. Evaluate (HIT/MISS)
            # -------------------------------------------------------------
            doc_names_normalized = [normalize_doc_name(doc_name) for doc_name in retrieved_doc_names]
            
            id_eval = bool(normalized_ground_truth) and any(
                doc_id in normalized_ground_truth for doc_id in doc_names_normalized
            )
            if is_negative_case:
                id_eval = True
            
            if id_eval:
                hit_count += 1

            if is_negative_case:
                status = "FORCED_TRUE"
            else:
                status = "HIT" if id_eval else "MISS"
            logger.info(
                f"[{global_idx}/{total_cases_all}] {status} | "
                f"Docs (Top {len(retrieved_doc_names)}): {retrieved_doc_names} | "
                f"Time: {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"[{global_idx}/{total_cases_all}] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            triplets_dump.append(
                {
                    "question": question,
                    "case_type": case_type,
                    "source": "api_retrieval_error",
                    "error": str(e),
                    "payload": {
                        "query": question,
                        "top_k": top_k,
                        "search_type": search_type,
                    },
                    "target_top_k": top_k,
                    "elapsed_s": elapsed,
                }
            )
            retrieved_doc_names = []
            id_eval = False
            triplet_counts = {
                "N_triplets": 0,
                "N_entity": 0,
                "N_entity_type": 0,
                "N_document_chunk": 0,
            }
            if is_negative_case:
                id_eval = True
                hit_count += 1
        
        elapsed_times.append(elapsed)

        # -------------------------------------------------------------
        # 5. Write to Output
        # -------------------------------------------------------------
        record = {
            "question": question,
            "case_type": case_type,
            "id_ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            "id_eval": id_eval,
            "N_triplets": triplet_counts["N_triplets"],
            "N_entity": triplet_counts["N_entity"],
            "N_entity_type": triplet_counts["N_entity_type"],
            "N_document_chunk": triplet_counts["N_document_chunk"],
            "elapsed_s": elapsed,
        }
        for i in range(top_k):
            record[f"context_{i + 1}"] = retrieved_doc_names[i] if i < len(retrieved_doc_names) else ""

        record_df = pd.DataFrame([record], columns=columns)
        record_df.to_csv(output_csv, mode="a", header=write_header, index=False)
        write_header = False

        accuracy = (hit_count / idx) * 100
        logger.info(
            f"[{global_idx}/{total_cases_all}] Running accuracy (this run): "
            f"{hit_count}/{idx} = {accuracy:.1f}%"
        )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Total: {total_cases} | Hits: {hit_count} | Accuracy: {(hit_count/total_cases)*100:.1f}%")
    if elapsed_times:
        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        logger.info(f"Average time per request: {avg_elapsed:.2f}s")
    logger.info(f"Results saved to: {output_csv}")
    logger.info("=" * 60)

    with open(triplets_json, "w", encoding="utf-8") as handle:
        json.dump(triplets_dump, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval endpoint results against ground-truth CSV (Compatibility Mode)."
    )
    # Default values match what the user might be used to
    parser.add_argument(
        "--input",
        default="testing/data_test_new_800.csv",
        help="Input CSV file path",
    )
    parser.add_argument("--output", default="eval_output.csv", help="Output CSV file path")
    parser.add_argument("--log", default="eval.log", help="Log file path")
    parser.add_argument(
        "--triplets-json", default="eval_triplets.json", help="Triplets dump JSON path"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of unique knowledge IDs to retrieve",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/search/retrieval",
        help="Retrieval API endpoint",
    )
    parser.add_argument(
        "--search-type",
        default="graph_completion_custom",
        help="Search type: chunks, graph_completion, graph_completion_custom",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--api-token",
        default="",
        help="Optional bearer token for Authorization header",
    )
    parser.add_argument(
        "--debug-trace",
        dest="debug_trace",
        action="store_true",
        default=True,
        help="Send X-Debug-Trace: true header",
    )
    parser.add_argument(
        "--no-debug-trace",
        dest="debug_trace",
        action="store_false",
        help="Disable X-Debug-Trace header",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based row index to start from (useful for resuming runs)",
    )
    
    args = parser.parse_args()

    # Ensure output directory exists if path has directory components
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    asyncio.run(
        evaluate(
            args.input,
            args.output,
            args.log,
            args.top_k,
            args.triplets_json,
            args.start_index,
            args.api_url,
            args.search_type,
            args.timeout,
            args.debug_trace,
            args.api_token,
        )
    )


if __name__ == "__main__":
    main()
