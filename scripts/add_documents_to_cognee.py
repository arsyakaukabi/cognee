#!/usr/bin/env python3
"""
Script to add all documents from /data/* to Cognee.

Supports batching for better memory management and progress tracking.
Runs without authentication.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Disable authentication before importing cognee
os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

# Add cognee to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cognee


async def add_all_documents(
    data_dir: str = "./data",
    dataset_name: str = "bribrain_docs",
    batch_size: int = 20,
    file_pattern: str = "*.txt",
    dry_run: bool = False
):
    """
    Add all documents from data directory to Cognee.
    
    Args:
        data_dir: Directory containing documents
        dataset_name: Name of the dataset in Cognee
        batch_size: Number of files to add per batch
        file_pattern: Glob pattern for files to add
        dry_run: If True, only show what would be added without actually adding
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist", file=sys.stderr)
        return False
    
    # Get all files matching pattern
    files = sorted(data_path.glob(file_pattern))
    
    if not files:
        print(f"No files matching '{file_pattern}' found in {data_dir}", file=sys.stderr)
        return False
    
    print(f"=" * 60)
    print(f"Cognee Document Ingestion (No Auth)")
    print(f"=" * 60)
    print(f"Data Directory: {data_path.absolute()}")
    print(f"Dataset Name:   {dataset_name}")
    print(f"Total Files:    {len(files)}")
    print(f"Batch Size:     {batch_size}")
    print(f"Dry Run:        {dry_run}")
    print(f"=" * 60)
    
    # Categorize files
    categories = {}
    for f in files:
        prefix = f.stem.split("_")[0] if "_" in f.stem else "other"
        categories.setdefault(prefix, []).append(f)
    
    print("\nFiles by category:")
    for cat, cat_files in sorted(categories.items()):
        print(f"  {cat}: {len(cat_files)} files")
    
    if dry_run:
        print("\n[DRY RUN] Would add the following files:")
        for i, f in enumerate(files, 1):
            print(f"  {i:3d}. {f.name}")
        return True
    
    print("\nStarting document ingestion...")
    start_time = datetime.now()
    
    # Process in batches
    total_batches = (len(files) + batch_size - 1) // batch_size
    successful = 0
    failed = 0
    errors = []
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(files))
        batch_files = files[batch_start:batch_end]
        
        print(f"\n[Batch {batch_num + 1}/{total_batches}] Processing {len(batch_files)} files...")
        
        for i, file_path in enumerate(batch_files, 1):
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Add to Cognee with dataset_name (no user parameter)
                await cognee.add(content, dataset_name=dataset_name)
                
                successful += 1
                progress = batch_start + i
                print(f"  [{progress}/{len(files)}] ✓ {file_path.name}")
                
            except Exception as e:
                failed += 1
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                print(f"  [{batch_start + i}/{len(files)}] ✗ {file_path.name} - {str(e)}", file=sys.stderr)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total Files:     {len(files)}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {failed}")
    print(f"Duration:        {duration}")
    if len(files) > 0:
        print(f"Avg per file:    {duration.total_seconds() / len(files):.2f}s")
    
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
    
    return failed == 0


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Add documents to Cognee (no auth)")
    parser.add_argument(
        "--data-dir", "-d",
        default="./data",
        help="Directory containing documents (default: ./data)"
    )
    parser.add_argument(
        "--dataset", "-n",
        default="bribrain_docs",
        help="Dataset name in Cognee (default: bribrain_docs)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=20,
        help="Number of files per batch (default: 20)"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*.txt",
        help="File pattern to match (default: *.txt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without actually adding"
    )
    
    args = parser.parse_args()
    
    # Add documents
    success = await add_all_documents(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        file_pattern=args.pattern,
        dry_run=args.dry_run
    )
    
    if not success:
        sys.exit(1)
    
    print("\n✓ All documents added successfully!")
    print("  Run 'cognee.cognify()' separately to process them.")


if __name__ == "__main__":
    asyncio.run(main())
