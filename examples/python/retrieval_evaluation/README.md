# Retrieval Evaluation Framework

An isolated evaluation framework to measure retrieval performance for Graph Completion and Chunk-based search methods against ground truth knowledge IDs.

## Features

- **Dual Search Methods**: Evaluates both `SearchType.GRAPH_COMPLETION` and `SearchType.CHUNKS`
- **Ground Truth Comparison**: Compares retrieved results against CSV ground truth knowledge IDs
- **Comprehensive Metrics**: Calculates Hit@k, HitRate@k, Precision@k, and Recall@k (at k=5 and k=10)
- **Result Expansion**: Automatically expands search results until exactly N unique knowledge IDs are retrieved
- **Context Generation**: Generates reranked chunks from top-N knowledge IDs for LLM consumption
- **Multiple Report Formats**: Generates CSV, JSON, and console summaries

## CSV Format

The evaluation CSV should have the following columns:

- `question`: Query text for search
- `context_ground_truth`: JSON array string of knowledge IDs (ground truth)
  - Example: `'["id1", "id2", "id3"]'`
- Additional metadata columns (optional): `generator`, `style`, `case_type`, `gt_categories`, etc.

**Important**: `ground_truth_id` = `TextDocument.name` in the knowledge graph.

## Usage

### Basic Example

```python
import asyncio
from examples.python.retrieval_evaluation.config import EvaluationConfig
from examples.python.retrieval_evaluation.evaluator import run_evaluation
from examples.python.retrieval_evaluation.report_generator import generate_reports

async def main():
    # Configure evaluation
    config = EvaluationConfig(
        csv_path="path/to/evaluation_data.csv",
        question_column="question",
        gt_column="context_ground_truth",
        document_paths=None,  # If None, assumes documents already added
        target_n_unique_ids=10,  # Target number of unique knowledge IDs
        initial_top_k=10,
        max_expansion_top_k=200,
        output_dir="evaluation_results",
        generate_context_output=True,  # Generate reranked chunks for LLM
        read_only_mode=False  # Set to True if documents already added/cognified
    )
    
    # Run evaluation
    results = await run_evaluation(config)
    
    # Generate reports
    generate_reports(results, config)

if __name__ == "__main__":
    asyncio.run(main())
```

### With Document Addition

```python
config = EvaluationConfig(
    csv_path="evaluation_data.csv",
    document_paths=[
        "path/to/doc1.md",
        "path/to/doc2.md",
        # ... more documents
    ],
    target_n_unique_ids=10,
    read_only_mode=False  # Will add and cognify documents
)
```

### Read-Only Mode (Documents Already Added)

```python
config = EvaluationConfig(
    csv_path="evaluation_data.csv",
    document_paths=None,
    read_only_mode=True  # Skip add/cognify, use existing graph
)
```

## Configuration Options

- `csv_path`: Path to CSV file with evaluation data
- `question_column`: Column name for questions (default: "question")
- `gt_column`: Column name for ground truth IDs (default: "context_ground_truth")
- `document_paths`: List of document paths to add (None = assume already added)
- `target_n_unique_ids`: Target number of unique knowledge IDs to retrieve (default: 10)
- `initial_top_k`: Initial search top_k (default: 10)
- `max_expansion_top_k`: Maximum top_k for expansion (default: 200)
- `output_dir`: Output directory for reports (default: "evaluation_results")
- `generate_context_output`: Generate reranked chunks for LLM (default: True)
- `context_output_file`: Filename for context output (default: "context_output.json")
- `read_only_mode`: Skip add/cognify if True (default: False)

## Output Files

The framework generates the following output files in `output_dir`:

1. **evaluation_results.csv**: Per-query results with all metrics
2. **evaluation_results.json**: Full results in JSON format
3. **context_output.json** (if enabled): Reranked chunks for LLM consumption

## Metrics Explained

- **Hit@k**: Boolean (1.0 or 0.0) - Is ANY ground truth ID in top-k?
- **HitRate@k**: Fraction of ground truth IDs found in top-k (0.0 to 1.0)
- **Precision@k**: Fraction of top-k that are ground truth (0.0 to 1.0)
- **Recall@k**: Fraction of ground truth found in top-k (0.0 to 1.0)

All metrics are calculated at k=5 and k=10.

## Result Expansion

The framework automatically expands search results until exactly N unique knowledge IDs are retrieved:

1. Start with initial `top_k` results
2. Map to knowledge IDs (may get < N unique IDs)
3. If unique IDs < N:
   - Increase `top_k` (exponential backoff: 10 → 20 → 40 → 80...)
   - Fetch more results
   - Map to knowledge IDs
   - Continue until ≥ N unique IDs
4. Return exactly N unique knowledge IDs (ordered by retrieval rank)

## Context Generation

For LLM consumption (separate from evaluation):

1. Extract all chunks from top-N retrieved knowledge IDs
2. Rerank chunks using pgvector similarity with query
3. Return reranked chunks ordered by similarity score

This is saved to `context_output.json` for later LLM use.

## Architecture

```
retrieval_evaluation/
├── __init__.py
├── config.py                  # Configuration
├── csv_loader.py              # CSV loader + ground truth parsing
├── document_manager.py        # Add & cognify documents
├── search_executor.py         # Search execution wrapper
├── result_mapper.py           # Map results to knowledge IDs
├── result_expander.py         # Expand results to N unique IDs
├── chunk_extractor.py         # Extract chunks from knowledge IDs
├── chunk_reranker.py          # Rerank chunks by similarity
├── metrics_calculator.py     # Calculate metrics
├── evaluator.py              # Main orchestrator
├── report_generator.py        # Generate reports
└── README.md                  # This file
```

## Notes

- The framework assumes `ground_truth_id` = `TextDocument.name`
- Results are expanded to exactly N unique knowledge IDs (not just top-N triplets/chunks)
- Context generation is separate from evaluation (for LLM consumption only)
- All chunks from top-N knowledge IDs are extracted and reranked
- The framework handles all node types: DocumentChunk, Entity, and EntityType
