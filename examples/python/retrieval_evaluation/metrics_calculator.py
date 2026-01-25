"""
Metrics calculator for retrieval evaluation.
Calculates Hit@k, Precision@k, Recall@k, and HitRate@k.
"""

from typing import List, Dict


def calculate_hit_at_k(retrieved: List[str], gt: List[str], k: int) -> bool:
    """
    Check if ANY ground truth ID is in top-k retrieved.
    
    Args:
        retrieved: Ordered list of retrieved knowledge IDs
        gt: List of ground truth knowledge IDs
        k: Top-k to check
        
    Returns:
        True if any ground truth ID is in top-k
    """
    if not retrieved or not gt or k <= 0:
        return False
    
    top_k_retrieved = retrieved[:k]
    return any(gt_id in top_k_retrieved for gt_id in gt)


def calculate_hit_rate_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    """
    Calculate fraction of ground truth IDs found in top-k.
    
    Args:
        retrieved: Ordered list of retrieved knowledge IDs
        gt: List of ground truth knowledge IDs
        k: Top-k to check
        
    Returns:
        Fraction of ground truth IDs found (0.0 to 1.0)
    """
    if not retrieved or not gt or k <= 0:
        return 0.0
    
    top_k_retrieved = set(retrieved[:k])
    found_count = sum(1 for gt_id in gt if gt_id in top_k_retrieved)
    return found_count / len(gt) if len(gt) > 0 else 0.0


def calculate_precision_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    """
    Calculate precision@k: fraction of top-k that are ground truth.
    
    Args:
        retrieved: Ordered list of retrieved knowledge IDs
        gt: List of ground truth knowledge IDs
        k: Top-k to check
        
    Returns:
        Precision@k (0.0 to 1.0)
    """
    if not retrieved or k <= 0:
        return 0.0
    
    top_k_retrieved = retrieved[:k]
    gt_set = set(gt)
    relevant_count = sum(1 for ret_id in top_k_retrieved if ret_id in gt_set)
    return relevant_count / len(top_k_retrieved) if len(top_k_retrieved) > 0 else 0.0


def calculate_recall_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    """
    Calculate recall@k: fraction of ground truth found in top-k.
    
    Args:
        retrieved: Ordered list of retrieved knowledge IDs
        gt: List of ground truth knowledge IDs
        k: Top-k to check
        
    Returns:
        Recall@k (0.0 to 1.0)
    """
    if not retrieved or not gt or k <= 0:
        return 0.0
    
    top_k_retrieved = set(retrieved[:k])
    found_count = sum(1 for gt_id in gt if gt_id in top_k_retrieved)
    return found_count / len(gt) if len(gt) > 0 else 0.0


def calculate_metrics_for_query(retrieved: List[str], gt: List[str]) -> Dict[str, float]:
    """
    Calculate all metrics for a single query.
    
    Args:
        retrieved: Ordered list of retrieved knowledge IDs (exactly N, e.g., 10)
        gt: List of ground truth knowledge IDs
        
    Returns:
        Dictionary with all metrics:
        - hit_at_5, hit_at_10 (bool as float: 1.0 or 0.0)
        - hit_rate_at_5, hit_rate_at_10
        - precision_at_5, precision_at_10
        - recall_at_5, recall_at_10
    """
    return {
        "hit_at_5": float(calculate_hit_at_k(retrieved, gt, 5)),
        "hit_at_10": float(calculate_hit_at_k(retrieved, gt, 10)),
        "hit_rate_at_5": calculate_hit_rate_at_k(retrieved, gt, 5),
        "hit_rate_at_10": calculate_hit_rate_at_k(retrieved, gt, 10),
        "precision_at_5": calculate_precision_at_k(retrieved, gt, 5),
        "precision_at_10": calculate_precision_at_k(retrieved, gt, 10),
        "recall_at_5": calculate_recall_at_k(retrieved, gt, 5),
        "recall_at_10": calculate_recall_at_k(retrieved, gt, 10),
    }
