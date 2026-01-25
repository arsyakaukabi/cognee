#!/usr/bin/env python3
"""
Script to verify that ground truth document IDs from data_test.csv exist in the database.
Checks both relational DB (Data table) and graph DB (TextDocument nodes).
"""

import argparse
import asyncio
import ast
import json
import re
from typing import List, Dict, Set, Tuple

import pandas as pd

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.models import Data
from sqlalchemy import select


def parse_list(value) -> List[str]:
    """Parse list from CSV cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    
    value = value.replace('\n', '').replace('\r', '').strip()
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


async def get_all_document_names_from_graph() -> Set[str]:
    """Get all TextDocument names from graph database."""
    graph_engine = await get_graph_engine()
    
    # Query all TextDocument nodes
    query = """
    MATCH (doc)
    WHERE doc.type = 'TextDocument' OR doc.type = 'PdfDocument'
    RETURN doc.name AS name
    """
    
    try:
        results = await graph_engine.query(query)
        names = set()
        for row in results:
            name = row.get("name") if isinstance(row, dict) else row[0]
            if name:
                names.add(str(name))
        return names
    except Exception as e:
        print(f"Warning: Could not query graph DB: {e}")
        return set()


async def get_all_data_names_from_relational() -> Set[str]:
    """Get all Data names from relational database."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data.name))
        names = set()
        for row in result.all():
            if row[0]:
                names.add(str(row[0]))
        return names


async def check_ground_truth_ids(input_csv: str, output_csv: str) -> None:
    """Check if ground truth IDs exist in database."""
    print("=" * 70)
    print("Checking Ground Truth Document IDs")
    print("=" * 70)
    
    # Load CSV
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} rows from {input_csv}")
    
    # Get all document names from databases
    print("\nQuerying graph database for TextDocument names...")
    graph_names = await get_all_document_names_from_graph()
    print(f"  Found {len(graph_names)} documents in graph DB")
    
    print("\nQuerying relational database for Data names...")
    relational_names = await get_all_data_names_from_relational()
    print(f"  Found {len(relational_names)} documents in relational DB")
    
    # Combine all known names
    all_db_names = graph_names | relational_names
    print(f"\nTotal unique document names in DB: {len(all_db_names)}")
    
    # Check each test case
    results = []
    all_gt_ids: Set[str] = set()
    found_ids: Set[str] = set()
    missing_ids: Set[str] = set()
    
    print("\n" + "-" * 70)
    print("Checking each test case...")
    print("-" * 70)
    
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        question = getattr(row, "question", "")[:60]
        context_ids = parse_list(getattr(row, "context_ground_truth", ""))
        
        # Skip if no ground truth
        if not context_ids:
            continue
        
        case_found = []
        case_missing = []
        
        for doc_id in context_ids:
            all_gt_ids.add(doc_id)
            
            # Check direct match
            if doc_id in all_db_names:
                found_ids.add(doc_id)
                case_found.append(doc_id)
            else:
                # Check with category prefix (e.g., helpdesk__id, wi__id, produk__id)
                found_with_prefix = False
                for prefix in ["helpdesk__", "wi__", "produk__", "promo__", "program__"]:
                    prefixed_name = f"{prefix}{doc_id}"
                    if prefixed_name in all_db_names:
                        found_ids.add(doc_id)
                        case_found.append(f"{doc_id} (as {prefixed_name})")
                        found_with_prefix = True
                        break
                
                if not found_with_prefix:
                    missing_ids.add(doc_id)
                    case_missing.append(doc_id)
        
        results.append({
            "row": idx,
            "question": question,
            "gt_ids": json.dumps(context_ids),
            "found": json.dumps(case_found),
            "missing": json.dumps(case_missing),
            "all_found": len(case_missing) == 0
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal unique ground truth IDs: {len(all_gt_ids)}")
    print(f"  ✅ Found in DB:     {len(found_ids)}")
    print(f"  ❌ Missing from DB: {len(missing_ids)}")
    
    if missing_ids:
        print("\n" + "-" * 70)
        print("Missing IDs (first 20):")
        print("-" * 70)
        for i, doc_id in enumerate(sorted(missing_ids)[:20]):
            print(f"  - {doc_id}")
        if len(missing_ids) > 20:
            print(f"  ... and {len(missing_ids) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Check if ground truth document IDs exist in database"
    )
    parser.add_argument("--input", default="testing/data_test.csv")
    parser.add_argument("--output", default="testing/gt_id_check_results.csv")
    args = parser.parse_args()
    
    asyncio.run(check_ground_truth_ids(args.input, args.output))


if __name__ == "__main__":
    main()
