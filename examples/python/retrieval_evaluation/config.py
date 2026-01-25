"""
Configuration for retrieval evaluation framework.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    
    csv_path: str
    question_column: str = "question"
    gt_column: str = "context_ground_truth"
    document_paths: Optional[List[str]] = None  # If None, assume documents already added
    target_n_unique_ids: int = 10  # Target number of unique knowledge IDs
    initial_top_k: int = 50  # Initial search top_k (increased to reduce expansion iterations)
    max_expansion_top_k: int = 200  # Maximum top_k for expansion
    graph_completion_initial_top_k: int = 50  # Initial top_k for Graph Completion (increased)
    chunk_initial_top_k: int = 50  # Initial top_k for Chunk search (increased)
    output_dir: str = "evaluation_results"
    generate_context_output: bool = True  # Generate reranked chunks for LLM
    context_output_file: str = "context_output.json"
    read_only_mode: bool = False  # Skip add/cognify if True
