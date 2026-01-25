"""
Test script for result_mapper.py
Tests mapping triplets (from Graph Completion search) to knowledge IDs.
"""

import asyncio
import os
from pathlib import Path

# Load environment
import dotenv
dotenv.load_dotenv(override=True)

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import setup_logging

from examples.python.retrieval_evaluation.result_mapper import (
    map_triplet_results_to_knowledge_ids,
    map_chunk_results_to_knowledge_ids
)


async def test_triplet_mapping():
    """Test mapping triplets to knowledge IDs."""
    
    logger = setup_logging()
    print("=" * 80)
    print("TEST RESULT MAPPER - TRIPLETS TO KNOWLEDGE IDs")
    print("=" * 80)
    
    # Setup: Ensure graph is ready
    print("\n📊 Checking graph status...")
    from cognee.infrastructure.databases.graph import get_graph_engine
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    print(f"   Graph has {len(nodes)} nodes and {len(edges)} edges")
    
    if len(nodes) == 0:
        print("\n⚠️  Graph is empty! Please add and cognify documents first.")
        print("   Example:")
        print("   ```python")
        print("   await cognee.add(['path/to/doc1.md', 'path/to/doc2.md'])")
        print("   await cognee.cognify()")
        print("   ```")
        return
    
    # Test queries
    test_queries = [
        "Bagaimana cara mengaktifkan notifikasi?",
        "Apa langkah-langkah yang perlu dilakukan?",
        "Bagaimana proses ini bekerja?",
    ]
    
    # Initialize Graph Completion retriever
    # Use higher top_k (50) to match the optimized config and reduce expansion iterations
    graph_retriever = GraphCompletionRetriever(top_k=50)
    
    print("\n" + "=" * 80)
    print("TESTING TRIPLET MAPPING")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print("─" * 80)
        
        try:
            # Step 1: Get triplets from Graph Completion search
            print("\n🔍 Step 1: Getting triplets from Graph Completion search...")
            triplets = await graph_retriever.get_context(query)
            print(f"   ✅ Retrieved {len(triplets)} triplets")
            
            if not triplets:
                print("   ⚠️  No triplets found. Skipping this query.")
                continue
            
            # Show first triplet structure
            if triplets:
                first_triplet = triplets[0]
                print(f"\n   First triplet structure:")
                print(f"   - Node1 ID: {first_triplet.node1.id if hasattr(first_triplet.node1, 'id') else 'N/A'}")
                print(f"   - Node1 Type: {first_triplet.node1.attributes.get('type', 'N/A') if hasattr(first_triplet.node1, 'attributes') else 'N/A'}")
                print(f"   - Node1 Name: {first_triplet.node1.attributes.get('name', 'N/A') if hasattr(first_triplet.node1, 'attributes') else 'N/A'}")
                print(f"   - Node2 ID: {first_triplet.node2.id if hasattr(first_triplet.node2, 'id') else 'N/A'}")
                print(f"   - Node2 Type: {first_triplet.node2.attributes.get('type', 'N/A') if hasattr(first_triplet.node2, 'attributes') else 'N/A'}")
                print(f"   - Node2 Name: {first_triplet.node2.attributes.get('name', 'N/A') if hasattr(first_triplet.node2, 'attributes') else 'N/A'}")
                print(f"   - Relationship: {first_triplet.relationship_name if hasattr(first_triplet, 'relationship_name') else 'N/A'}")
            
            # Step 2: Map triplets to knowledge IDs
            print("\n🗺️  Step 2: Mapping triplets to knowledge IDs...")
            knowledge_ids = await map_triplet_results_to_knowledge_ids(triplets, graph_engine)
            print(f"   ✅ Mapped to {len(knowledge_ids)} knowledge IDs")
            
            # Show results
            print(f"\n   Knowledge IDs found:")
            unique_ids = list(dict.fromkeys(knowledge_ids))  # Preserve order, remove duplicates
            for idx, kid in enumerate(unique_ids, 1):
                print(f"   {idx}. {kid}")
            
            # Show statistics
            print(f"\n   Statistics:")
            print(f"   - Total knowledge IDs (with duplicates): {len(knowledge_ids)}")
            print(f"   - Unique knowledge IDs: {len(unique_ids)}")
            print(f"   - Duplicates: {len(knowledge_ids) - len(unique_ids)}")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


async def test_chunk_mapping():
    """Test mapping chunks to knowledge IDs (bonus test)."""
    
    print("\n" + "=" * 80)
    print("BONUS: TESTING CHUNK MAPPING")
    print("=" * 80)
    
    graph_engine = await get_graph_engine()
    
    test_query = "Bagaimana cara mengaktifkan notifikasi?"
    
    print(f"\nQuery: {test_query}")
    
    try:
        # Get chunks from CHUNKS search
        print("\n🔍 Getting chunks from CHUNKS search...")
        chunk_results = await cognee.search(
            query_type=SearchType.CHUNKS,
            query_text=test_query,
            top_k=50,  # Use higher top_k to match optimized config
        )
        print(f"   ✅ Retrieved {len(chunk_results)} chunks")
        
        if not chunk_results:
            print("   ⚠️  No chunks found. Skipping.")
            return
        
        # Map chunks to knowledge IDs
        print("\n🗺️  Mapping chunks to knowledge IDs...")
        knowledge_ids = await map_chunk_results_to_knowledge_ids(chunk_results, graph_engine)
        print(f"   ✅ Mapped to {len(knowledge_ids)} knowledge IDs")
        
        unique_ids = list(dict.fromkeys(knowledge_ids))
        print(f"\n   Knowledge IDs found:")
        for idx, kid in enumerate(unique_ids, 1):
            print(f"   {idx}. {kid}")
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    try:
        # Test triplet mapping (main test)
        await test_triplet_mapping()
        
        # Test chunk mapping (bonus)
        await test_chunk_mapping()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
