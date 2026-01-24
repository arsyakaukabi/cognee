"""
Simple search script untuk mendapatkan top 10 knowledge IDs dari sebuah query.

Script ini sangat sederhana:
- Input: question (query)
- Output: top 10 TextDocument.name (knowledge IDs)

Tidak perlu evaluasi, tidak perlu CSV, hanya search dan return knowledge IDs.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment
import dotenv
dotenv.load_dotenv(override=True)

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import setup_logging

from examples.python.retrieval_evaluation.search_executor import SearchExecutor
from examples.python.retrieval_evaluation.result_mapper import (
    map_triplet_results_to_knowledge_ids,
    map_chunk_results_to_knowledge_ids
)
from examples.python.retrieval_evaluation.result_expander import expand_to_n_unique_knowledge_ids


async def search_knowledge_ids(
    question: str,
    method: str = "graph",  # "graph" or "chunks"
    target_n: int = 10,
    initial_top_k: int = 50
) -> list[str]:
    """
    Search dan return top N knowledge IDs (TextDocument.name).
    
    Args:
        question: Query text
        method: Search method ("graph" untuk GRAPH_COMPLETION, "chunks" untuk CHUNKS)
        target_n: Target jumlah unique knowledge IDs (default: 10)
        initial_top_k: Initial search top_k (default: 50)
    
    Returns:
        List of knowledge IDs (TextDocument.name), ordered by retrieval rank
    """
    # Setup
    graph_engine = await get_graph_engine()
    search_executor = SearchExecutor(initial_top_k=initial_top_k)
    
    # Execute search
    if method == "graph":
        print(f"🔍 Searching using GRAPH_COMPLETION...")
        search_results = await search_executor.execute_graph_completion_search(
            question, 
            top_k=initial_top_k
        )
        
        # Map triplets to knowledge IDs with early stopping
        knowledge_ids = await map_triplet_results_to_knowledge_ids(
            search_results, 
            graph_engine,
            target_n=target_n
        )
        
        # Expand if needed to reach target_n
        if len(set(knowledge_ids)) < target_n:
            print(f"   Initial search: {len(set(knowledge_ids))} unique IDs (target: {target_n})")
            print(f"   Expanding search...")
            knowledge_ids = await expand_to_n_unique_knowledge_ids(
                search_results=search_results,
                method="graph",
                target_n=target_n,
                graph_engine=graph_engine,
                search_executor=search_executor,
                query=question,
                initial_top_k=initial_top_k,
                max_top_k=200
            )
        else:
            # Take only first target_n unique IDs
            unique_ids = []
            seen = set()
            for kid in knowledge_ids:
                if kid not in seen:
                    unique_ids.append(kid)
                    seen.add(kid)
                    if len(unique_ids) >= target_n:
                        break
            knowledge_ids = unique_ids
            
    elif method == "chunks":
        print(f"🔍 Searching using CHUNKS...")
        search_results = await search_executor.execute_chunk_search(
            question,
            top_k=initial_top_k
        )
        
        # Map chunks to knowledge IDs with early stopping
        knowledge_ids = await map_chunk_results_to_knowledge_ids(
            search_results,
            graph_engine,
            target_n=target_n
        )
        
        # Expand if needed to reach target_n
        if len(set(knowledge_ids)) < target_n:
            print(f"   Initial search: {len(set(knowledge_ids))} unique IDs (target: {target_n})")
            print(f"   Expanding search...")
            knowledge_ids = await expand_to_n_unique_knowledge_ids(
                search_results=search_results,
                method="chunks",
                target_n=target_n,
                graph_engine=graph_engine,
                search_executor=search_executor,
                query=question,
                initial_top_k=initial_top_k,
                max_top_k=200
            )
        else:
            # Take only first target_n unique IDs
            unique_ids = []
            seen = set()
            for kid in knowledge_ids:
                if kid not in seen:
                    unique_ids.append(kid)
                    seen.add(kid)
                    if len(unique_ids) >= target_n:
                        break
            knowledge_ids = unique_ids
    else:
        raise ValueError(f"Unknown method: {method}. Use 'graph' or 'chunks'.")
    
    # Return exactly target_n (or all if less than target_n)
    unique_ids = list(dict.fromkeys(knowledge_ids))  # Preserve order, remove duplicates
    return unique_ids[:target_n]


async def main():
    """Main function."""
    
    # ============================================
    # KONFIGURASI - EDIT BAGIAN INI
    # ============================================
    
    # Opsi 1: Query dari command line argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        # Opsi 2: Query hardcoded (edit di sini)
        question = "Bagaimana cara mengaktifkan notifikasi?"
    
    # Search method: "graph" (GRAPH_COMPLETION) atau "chunks" (CHUNKS)
    method = "graph"  # Ganti ke "chunks" jika ingin pakai CHUNKS search
    
    # Target jumlah knowledge IDs
    target_n = 10
    
    # ============================================
    # JALANKAN SEARCH
    # ============================================
    
    logger = setup_logging()
    
    print("=" * 80)
    print("SIMPLE SEARCH - TOP 10 KNOWLEDGE IDs")
    print("=" * 80)
    print(f"Query: {question}")
    print(f"Method: {method.upper()}")
    print(f"Target: {target_n} unique knowledge IDs")
    print("=" * 80)
    
    try:
        # Check graph status
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        
        if len(nodes) == 0:
            print("\n❌ ERROR: Graph is empty!")
            print("   Pastikan dokumen sudah di-add dan cognify terlebih dahulu.")
            print("   Contoh:")
            print("   ```python")
            print("   await cognee.add(['path/to/doc1.md'])")
            print("   await cognee.cognify()")
            print("   ```")
            return
        
        print(f"\n📊 Graph status: {len(nodes)} nodes, {len(edges)} edges")
        
        # Search
        print()
        knowledge_ids = await search_knowledge_ids(
            question=question,
            method=method,
            target_n=target_n,
            initial_top_k=50
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ SEARCH RESULTS")
        print("=" * 80)
        print(f"Found {len(knowledge_ids)} unique knowledge IDs:\n")
        
        for i, kid in enumerate(knowledge_ids, 1):
            print(f"  {i}. {kid}")
        
        print("\n" + "=" * 80)
        
        # Return as list (bisa digunakan untuk programmatic access)
        return knowledge_ids
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # Exit dengan code 0 jika berhasil, 1 jika error
    sys.exit(0 if result else 1)
