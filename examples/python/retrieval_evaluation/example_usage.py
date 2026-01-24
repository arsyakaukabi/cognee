"""
Example usage of the retrieval evaluation framework.
"""

import asyncio
import os
from pathlib import Path

# Load environment variables
import dotenv
dotenv.load_dotenv(override=True)

# Disable backend access control for local testing
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

from examples.python.retrieval_evaluation.config import EvaluationConfig
from examples.python.retrieval_evaluation.evaluator import run_evaluation
from examples.python.retrieval_evaluation.report_generator import generate_reports


async def main():
    """Run evaluation example."""
    
    # Example 1: With document paths (will add and cognify)
    config = EvaluationConfig(
        csv_path="path/to/your/evaluation_data.csv",  # Update this path
        question_column="question",
        gt_column="context_ground_truth",
        document_paths=[
            # Add your document paths here
            # "path/to/doc1.md",
            # "path/to/doc2.md",
        ],
        target_n_unique_ids=10,
        initial_top_k=10,
        max_expansion_top_k=200,
        output_dir="evaluation_results",
        generate_context_output=True,
        read_only_mode=False  # Will add and cognify documents
    )
    
    # Example 2: Read-only mode (documents already added)
    # config = EvaluationConfig(
    #     csv_path="path/to/your/evaluation_data.csv",
    #     document_paths=None,
    #     target_n_unique_ids=10,
    #     read_only_mode=True  # Skip add/cognify
    # )
    
    print("=" * 80)
    print("RETRIEVAL EVALUATION FRAMEWORK")
    print("=" * 80)
    print(f"CSV Path: {config.csv_path}")
    print(f"Target N unique IDs: {config.target_n_unique_ids}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Read-only mode: {config.read_only_mode}")
    print("=" * 80)
    
    # Run evaluation
    try:
        results = await run_evaluation(config)
        
        # Generate reports
        generate_reports(results, config)
        
        print("\n✅ Evaluation complete!")
        print(f"Results saved to: {config.output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
