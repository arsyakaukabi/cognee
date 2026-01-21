#!/usr/bin/env python3
"""
Simple Cognify script - runs the standard cognee.cognify() function.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Disable authentication
os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))

import cognee


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cognee cognify")
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=None,
        help="Dataset names to process (default: all)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=None,
        help="Maximum chunk size in tokens (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Chunks per batch (default: 100)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cognify - Standard Pipeline")
    print("=" * 60)
    print(f"Datasets:         {args.datasets or 'all'}")
    print(f"Chunk Size:       {args.chunk_size or 'auto'}")
    print(f"Batch Size:       {args.batch_size}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        result = await cognee.cognify(
            datasets=args.datasets,
            chunk_size=args.chunk_size,
            chunks_per_batch=args.batch_size,
        )
        
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("COGNIFY COMPLETED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Result: {result}")
        
    except Exception as e:
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("COGNIFY FAILED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
