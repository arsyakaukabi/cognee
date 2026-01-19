"""
Reciprocal Rank Fusion (RRF) implementation for combining multiple ranked result lists.

RRF is a simple and effective method for combining results from multiple retrieval
systems. It uses reciprocal ranks to give diminishing importance to lower-ranked
results while avoiding issues with score normalization across different systems.

Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
"Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def rrf_fusion(
    result_lists: List[List[Tuple[str, Any]]],
    k: int = 60,
    final_top_k: int = 50,
) -> List[Tuple[str, float]]:
    """
    Apply Reciprocal Rank Fusion to combine multiple ranked result lists.

    RRF computes a combined score for each document based on its rank in each
    result list using the formula: score(d) = Σ (1 / (k + rank_i(d)))

    Args:
        result_lists: List of ranked results from different retrievers.
            Each result list is a list of (doc_id, score) tuples, where score
            can be any value (only rank position is used).
        k: RRF smoothing constant. Higher values give more weight to lower-ranked
            documents. Default is 60, which is commonly used in literature.
        final_top_k: Maximum number of results to return after fusion.

    Returns:
        Fused list of (doc_id, rrf_score) tuples, sorted by RRF score descending.
        Ties are broken by document ID for determinism.

    Example:
        >>> vector_results = [("doc1", 0.95), ("doc2", 0.87), ("doc3", 0.72)]
        >>> bm25_results = [("doc2", 15.3), ("doc4", 12.1), ("doc1", 8.7)]
        >>> fused = rrf_fusion([vector_results, bm25_results], k=60, final_top_k=10)
        >>> # doc2 ranks high in both, so it will likely be top result
    """
    if not result_lists:
        return []

    rrf_scores: Dict[str, float] = defaultdict(float)

    for results in result_lists:
        if not results:
            continue
        for rank, (doc_id, _score) in enumerate(results, start=1):
            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank)

    if not rrf_scores:
        return []

    # Sort by RRF score descending, then by doc_id ascending for determinism
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return sorted_results[:final_top_k]


def rrf_fusion_with_payloads(
    result_lists: List[List[Tuple[str, Any, Any]]],
    k: int = 60,
    final_top_k: int = 50,
) -> List[Tuple[str, float, Any]]:
    """
    Apply RRF fusion and preserve payload data from original results.

    Similar to rrf_fusion but also tracks and returns payload data for each
    document. If a document appears in multiple result lists, the payload
    from the first occurrence is used.

    Args:
        result_lists: List of ranked results, where each result is
            (doc_id, score, payload).
        k: RRF smoothing constant.
        final_top_k: Maximum number of results to return.

    Returns:
        List of (doc_id, rrf_score, payload) tuples, sorted by RRF score.
    """
    if not result_lists:
        return []

    rrf_scores: Dict[str, float] = defaultdict(float)
    payloads: Dict[str, Any] = {}

    for results in result_lists:
        if not results:
            continue
        for rank, (doc_id, _score, payload) in enumerate(results, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
            # Keep first payload encountered
            if doc_id not in payloads:
                payloads[doc_id] = payload

    if not rrf_scores:
        return []

    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return [
        (doc_id, score, payloads[doc_id])
        for doc_id, score in sorted_results[:final_top_k]
    ]
