"""
Tracing script untuk melihat detail alur search_knowledge_ids step by step.
Menggunakan fungsi-fungsi asli dari result_mapper.py.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Load environment
import dotenv
dotenv.load_dotenv(override=True)
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge

# Import fungsi asli dari result_mapper
from examples.python.retrieval_evaluation.result_mapper import (
    get_documentchunks_for_entity,
    get_textdocument_for_chunk,
    get_knowledge_id_from_textdocument,
    map_node_to_knowledge_ids
)

# =====================================================================
# TRACE DATA STORAGE
# =====================================================================
trace_log = []

def log_trace(step: str, function_name: str, input_data: any, output_data: any, notes: str = ""):
    """Log a trace entry."""
    entry = {
        "step": step,
        "function": function_name,
        "input": input_data,
        "output": output_data,
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    trace_log.append(entry)
    print(f"  ✓ [{step}] {function_name}")


# =====================================================================
# TRACED FUNCTIONS
# =====================================================================

async def traced_execute_graph_completion_search(question: str, top_k: int, graph_retriever: GraphCompletionRetriever):
    """
    STEP 1: Execute Graph Completion Search
    Uses GraphCompletionRetriever to get triplets (Edges).
    """
    input_data = {"question": question, "top_k": top_k}
    
    graph_retriever.top_k = top_k
    triplets = await graph_retriever.get_context(question)
    
    # Serialize triplets for logging
    triplets_serialized = []
    for triplet in triplets:
        node1_id = str(triplet.node1.id) if hasattr(triplet.node1, "id") else "N/A"
        node2_id = str(triplet.node2.id) if hasattr(triplet.node2, "id") else "N/A"
        node1_type = triplet.node1.attributes.get("type", "Unknown") if hasattr(triplet.node1, "attributes") else "Unknown"
        node2_type = triplet.node2.attributes.get("type", "Unknown") if hasattr(triplet.node2, "attributes") else "Unknown"
        node1_name = triplet.node1.attributes.get("name", "")[:50] if hasattr(triplet.node1, "attributes") else ""
        node2_name = triplet.node2.attributes.get("name", "")[:50] if hasattr(triplet.node2, "attributes") else ""
        relationship = triplet.attributes.get("relationship_name", "related_to") if hasattr(triplet, "attributes") else "related_to"
        
        triplets_serialized.append({
            "node1_id": node1_id,
            "node1_type": node1_type,
            "node1_name": node1_name,
            "relationship": relationship,
            "node2_id": node2_id,
            "node2_type": node2_type,
            "node2_name": node2_name
        })
    
    output_data = {
        "triplet_count": len(triplets),
        "triplets_sample": triplets_serialized[:5],
        "all_triplets": triplets_serialized
    }
    
    log_trace("1", "GraphCompletionRetriever.get_context()", input_data, output_data,
              f"Retrieved {len(triplets)} triplets from vector + graph search")
    
    return triplets, triplets_serialized


async def traced_map_node_to_knowledge_ids(node_id: str, node_type: str, graph_engine, step_prefix: str):
    """
    STEP 2a/2b: Map a single node to knowledge IDs.
    MENGGUNAKAN FUNGSI ASLI dari result_mapper.py
    """
    input_data = {"node_id": node_id, "node_type": node_type}
    
    # Panggil fungsi asli
    knowledge_ids = await map_node_to_knowledge_ids(node_id, node_type, graph_engine)
    
    output_data = {
        "knowledge_ids": knowledge_ids,
    }
    
    log_trace(step_prefix, f"map_node_to_knowledge_ids({node_type})", input_data, output_data,
              f"Found {len(knowledge_ids)} TextDocument(s)")
    
    return knowledge_ids


async def traced_map_triplet_results_to_knowledge_ids(triplets, triplets_serialized, graph_engine, target_n=None):
    """
    STEP 2: Map all triplets to knowledge IDs.
    Iterates through each triplet and maps both node1 and node2.
    """
    input_data = {"triplet_count": len(triplets)}
    
    all_knowledge_ids = []
    all_traversal_details = []
    seen_ids = set()
    
    for i, (triplet, triplet_ser) in enumerate(zip(triplets[:10], triplets_serialized[:10])):  # Limit to first 10
        # Early stopping trace
        if target_n and len(all_knowledge_ids) >= target_n:
            print(f"  🛑 Early stopping: reached {len(all_knowledge_ids)} IDs")
            break
            
        node1_id = str(triplet.node1.id) if hasattr(triplet.node1, "id") else None
        node2_id = str(triplet.node2.id) if hasattr(triplet.node2, "id") else None
        node1_type = triplet.node1.attributes.get("type", "") if hasattr(triplet.node1, "attributes") else ""
        node2_type = triplet.node2.attributes.get("type", "") if hasattr(triplet.node2, "attributes") else ""
        
        detail = {
            "triplet_index": i,
            "triplet": triplet_ser,
            "node1_mapping": None,
            "node2_mapping": None
        }
        
        # Map node1 using REAL function
        if node1_id and node1_type:
            kids = await traced_map_node_to_knowledge_ids(node1_id, node1_type, graph_engine, f"2.{i}.a")
            for kid in kids:
                if kid not in seen_ids:
                    all_knowledge_ids.append(kid)
                    seen_ids.add(kid)
            detail["node1_mapping"] = {"knowledge_ids": kids}
        
        # Map node2 using REAL function
        if node2_id and node2_type:
            kids = await traced_map_node_to_knowledge_ids(node2_id, node2_type, graph_engine, f"2.{i}.b")
            for kid in kids:
                if kid not in seen_ids:
                    all_knowledge_ids.append(kid)
                    seen_ids.add(kid)
            detail["node2_mapping"] = {"knowledge_ids": kids}
            
            # Additional check after node2 mapping
            if target_n and len(all_knowledge_ids) >= target_n:
                 print(f"  🛑 Early stopping: reached {len(all_knowledge_ids)} IDs after node2 mapping")
                 all_traversal_details.append(detail)
                 break
        
        all_traversal_details.append(detail)
    
    output_data = {
        "unique_knowledge_ids_count": len(all_knowledge_ids),
        "knowledge_ids": all_knowledge_ids,
        "mapping_details": all_traversal_details
    }
    
    log_trace("2", "map_triplet_results_to_knowledge_ids()", input_data, output_data,
              f"Mapped {len(triplets)} triplets → {len(all_knowledge_ids)} unique knowledge IDs")
    
    return all_knowledge_ids


async def traced_search_knowledge_ids(question: str, target_n: int = 10, initial_top_k: int = 50):
    """
    Main traced function mirroring search_knowledge_ids.
    """
    print("\n" + "=" * 80)
    print("🔍 TRACING: search_knowledge_ids")
    print("=" * 80)
    print(f"Query: {question}")
    print(f"Target: {target_n} unique knowledge IDs")
    print(f"Initial top_k: {initial_top_k}")
    print("=" * 80 + "\n")
    
    # Setup
    graph_engine = await get_graph_engine()
    graph_retriever = GraphCompletionRetriever(top_k=initial_top_k)
    
    # Log initial input
    log_trace("0", "search_knowledge_ids()", 
              {"question": question, "method": "graph", "target_n": target_n, "initial_top_k": initial_top_k},
              "Starting...", "Entry point")
    
    # STEP 1: Execute Graph Completion Search
    print("\n📌 STEP 1: Execute Graph Completion Search")
    print("-" * 60)
    triplets, triplets_serialized = await traced_execute_graph_completion_search(question, initial_top_k, graph_retriever)
    
    # STEP 2: Map Triplets to Knowledge IDs
    print("\n📌 STEP 2: Map Triplets to Knowledge IDs (Graph Traversal)")
    print("-" * 60)
    knowledge_ids = await traced_map_triplet_results_to_knowledge_ids(triplets, triplets_serialized, graph_engine, target_n=target_n)
    
    # STEP 3: Deduplicate and take top N
    print("\n📌 STEP 3: Deduplicate and Return Top N")
    print("-" * 60)
    unique_ids = list(dict.fromkeys(knowledge_ids))
    final_result = unique_ids[:target_n]
    
    log_trace("3", "deduplicate_and_limit()", 
              {"all_unique_ids": len(unique_ids), "target_n": target_n},
              {"final_result": final_result, "count": len(final_result)},
              f"Returning top {len(final_result)} knowledge IDs")
    
    print("\n" + "=" * 80)
    print("✅ FINAL RESULT")
    print("=" * 80)
    for i, kid in enumerate(final_result, 1):
        print(f"  {i}. {kid}")
    print("=" * 80)
    
    return final_result, trace_log


async def main():
    """Main function."""
    # Query dari user
    question = "Bagaimana cara mengaktifkan notifikasi transaksi di aplikasi BRImo?"
    
    print("\n" + "=" * 80)
    print("TRACE SEARCH KNOWLEDGE IDS - DETAILED FLOW")
    print("(Using actual functions from result_mapper.py)")
    print("=" * 80)
    
    # Run traced search
    result, trace = await traced_search_knowledge_ids(
        question=question,
        target_n=5,
        initial_top_k=50
    )
    
    # Save trace to file
    output_path = Path(__file__).parent.parent.parent.parent / "docs" / "search_knowledge_ids_trace.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📄 Trace saved to: {output_path}")
    
    return result, trace


if __name__ == "__main__":
    asyncio.run(main())
