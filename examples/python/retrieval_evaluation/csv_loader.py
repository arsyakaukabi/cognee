"""
CSV loader for evaluation data with ground truth parsing.
"""

import json
import ast
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path


@dataclass
class EvaluationRow:
    """Single evaluation row from CSV."""
    question: str
    ground_truth_ids: List[str]
    metadata: Dict[str, Any]


def parse_ground_truth_ids(gt_string: str) -> List[str]:
    """
    Parse JSON array string to list of IDs.
    
    Handles formats like:
    - '["id1", "id2"]'
    - "[\"id1\", \"id2\"]"
    - Python list format: ['id1', 'id2']
    
    Args:
        gt_string: JSON array string or Python list string
        
    Returns:
        List of ID strings
    """
    if not gt_string or not isinstance(gt_string, str):
        return []
    
    try:
        # Try JSON parsing first
        # Remove outer quotes if present
        cleaned = gt_string.strip().strip('"').strip("'")
        # Parse JSON array
        ids = json.loads(cleaned)
        if isinstance(ids, list):
            return [str(id) for id in ids]
        else:
            return [str(ids)]
    except (json.JSONDecodeError, ValueError):
        # Fallback to ast.literal_eval for Python list format
        try:
            ids = ast.literal_eval(gt_string)
            if isinstance(ids, list):
                return [str(id) for id in ids]
            else:
                return [str(ids)]
        except (ValueError, SyntaxError):
            return []


def load_evaluation_data(csv_path: str, question_column: str = "question", gt_column: str = "context_ground_truth") -> List[EvaluationRow]:
    """
    Load CSV and parse evaluation data.
    
    Args:
        csv_path: Path to CSV file
        question_column: Name of column containing questions
        gt_column: Name of column containing ground truth IDs (JSON array)
        
    Returns:
        List of EvaluationRow objects
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in CSV")
    if gt_column not in df.columns:
        raise ValueError(f"Column '{gt_column}' not found in CSV")
    
    rows = []
    for idx, row in df.iterrows():
        question = str(row[question_column]) if pd.notna(row[question_column]) else ""
        gt_string = str(row[gt_column]) if pd.notna(row[gt_column]) else "[]"
        
        ground_truth_ids = parse_ground_truth_ids(gt_string)
        
        # Store all other columns as metadata
        metadata = {col: row[col] for col in df.columns if col not in [question_column, gt_column]}
        # Convert pandas types to native Python types
        metadata = {k: v if pd.notna(v) else None for k, v in metadata.items()}
        
        rows.append(EvaluationRow(
            question=question,
            ground_truth_ids=ground_truth_ids,
            metadata=metadata
        ))
    
    return rows
