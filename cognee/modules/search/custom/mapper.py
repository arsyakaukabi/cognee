"""
Map search results (triplets/chunks) to knowledge IDs (TextDocument.name).
"""

import time
from typing import List, Dict, Any, Set, Optional
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger("CustomMapper")


async def get_textdocument_for_chunk(chunk_id: str, graph_engine: GraphDBInterface) -> Optional[Dict[str, Any]]:
    """
    Get TextDocument node for a given DocumentChunk.
    OPTIMIZED: Trusts edge semantics, skips node verification for speed.
    """
    try:
        edges = await graph_engine.get_edges(chunk_id)
        
        for edge in edges:
            doc_id = None
            doc_name = None
            
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    source_id, target_id, relationship_name, _ = edge
                    if relationship_name == "is_part_of" and source_id == chunk_id:
                        doc_id = target_id
                elif len(edge) == 3:
                    node1_dict, relationship_name, node2_dict = edge
                    if relationship_name == "is_part_of":
                        node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                        node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                        
                        # Extract document info directly from edge data (skip get_node)
                        if node1_id == chunk_id and isinstance(node2_dict, dict):
                            doc_id = node2_id
                            doc_name = node2_dict.get("name")
                        elif node2_id == chunk_id and isinstance(node1_dict, dict):
                            doc_id = node1_id
                            doc_name = node1_dict.get("name")
            
            if doc_id:
                # Return directly without verification
                return {"id": doc_id, "name": doc_name, "type": "TextDocument"}
                    
    except Exception as e:
        logger.warning(f"Error querying edges for chunk {chunk_id}: {e}")
    
    return None


async def get_knowledge_id_from_textdocument(doc: Dict[str, Any]) -> str:
    """
    Extract knowledge ID (TextDocument.name) from TextDocument node.
    
    Args:
        doc: TextDocument node data
        
    Returns:
        Knowledge ID (TextDocument.name)
    """
    if isinstance(doc, dict):
        return doc.get("name", "")
    else:
        return getattr(doc, "name", "")


async def get_documentchunks_for_entity(entity_id: str, graph_engine: GraphDBInterface) -> List[str]:
    """
    Get all DocumentChunk IDs that contain a given Entity.
    OPTIMIZED: Trusts edge semantics, skips node verification for speed.
    """
    chunk_ids = []
    try:
        edges = await graph_engine.get_edges(entity_id)
        
        for edge in edges:
            chunk_id = None
            
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    source_id, target_id, relationship_name, _ = edge
                    if relationship_name == "contains" and target_id == entity_id:
                        chunk_id = source_id
                elif len(edge) == 3:
                    node1_dict, relationship_name, node2_dict = edge
                    if relationship_name == "contains":
                        node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                        node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                        
                        if node2_id == entity_id:
                            chunk_id = node1_id
                        elif node1_id == entity_id:
                            chunk_id = node2_id
            
            if chunk_id:
                chunk_ids.append(chunk_id)
                    
    except Exception as e:
        logger.warning(f"Error querying edges for entity {entity_id}: {e}")
    
    return chunk_ids


async def get_entities_for_entitytype(entitytype_id: str, graph_engine: GraphDBInterface) -> List[str]:
    """
    Get all Entity IDs that have a given EntityType.
    OPTIMIZED: Trusts edge semantics, skips node verification for speed.
    """
    entity_ids = []
    try:
        edges = await graph_engine.get_edges(entitytype_id)
        
        for edge in edges:
            entity_id = None
            
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    source_id, target_id, relationship_name, _ = edge
                    if relationship_name == "is_a" and target_id == entitytype_id:
                        entity_id = source_id
                elif len(edge) == 3:
                    node1_dict, relationship_name, node2_dict = edge
                    if relationship_name == "is_a":
                        node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                        node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                        
                        if node2_id == entitytype_id:
                            entity_id = node1_id
                        elif node1_id == entitytype_id:
                            entity_id = node2_id
            
            if entity_id:
                entity_ids.append(entity_id)
                    
    except Exception as e:
        logger.warning(f"Error querying edges for entitytype {entitytype_id}: {e}")
    
    return entity_ids


async def map_node_to_knowledge_ids(node_id: str, node_type: str, graph_engine: GraphDBInterface) -> List[str]:
    """
    Map a single node to knowledge IDs (TextDocument.name).
    
    Handles DocumentChunk, Entity, and EntityType nodes.
    
    Args:
        node_id: Node ID
        node_type: Node type ("DocumentChunk", "Entity", "EntityType")
        graph_engine: Graph database interface
        
    Returns:
        List of knowledge IDs (may be empty or have multiple IDs)
    """
    knowledge_ids = []
    
    if node_type == "DocumentChunk":
        doc = await get_textdocument_for_chunk(node_id, graph_engine)
        if doc:
            knowledge_id = await get_knowledge_id_from_textdocument(doc)
            if knowledge_id:
                knowledge_ids.append(knowledge_id)
    
    elif node_type == "Entity":
        # Entity → DocumentChunk → TextDocument
        chunk_ids = await get_documentchunks_for_entity(node_id, graph_engine)
        for chunk_id in chunk_ids:
            doc = await get_textdocument_for_chunk(chunk_id, graph_engine)
            if doc:
                knowledge_id = await get_knowledge_id_from_textdocument(doc)
                if knowledge_id:
                    knowledge_ids.append(knowledge_id)
    
    elif node_type == "EntityType":
        # EntityType → Entity → DocumentChunk → TextDocument
        entity_ids = await get_entities_for_entitytype(node_id, graph_engine)
        for entity_id in entity_ids:
            chunk_ids = await get_documentchunks_for_entity(entity_id, graph_engine)
            for chunk_id in chunk_ids:
                doc = await get_textdocument_for_chunk(chunk_id, graph_engine)
                if doc:
                    knowledge_id = await get_knowledge_id_from_textdocument(doc)
                    if knowledge_id:
                        knowledge_ids.append(knowledge_id)
    
    return knowledge_ids


async def map_triplet_results_to_knowledge_ids(
    triplets: List[Edge],
    graph_engine: GraphDBInterface,
    target_n: int = None
) -> List[str]:
    """
    Map triplet results to knowledge IDs (TextDocument.name).
    
    Preserves order from triplet ranking (first occurrence of knowledge ID wins).
    Supports early stopping when target_n unique IDs are found.
    """
    func_start = time.time()
    
    knowledge_ids = []
    seen_ids = set()
    
    triplet_count = len(triplets)
    node_map_count = 0
    graph_query_time = 0.0
    
    for triplet in triplets:
        # Early stopping
        if target_n and len(knowledge_ids) >= target_n:
            break
            
        # Extract nodes
        node1_id = str(triplet.node1.id) if hasattr(triplet.node1, "id") else None
        node2_id = str(triplet.node2.id) if hasattr(triplet.node2, "id") else None
        
        node1_type = triplet.node1.attributes.get("type", "") if hasattr(triplet.node1, "attributes") else ""
        node2_type = triplet.node2.attributes.get("type", "") if hasattr(triplet.node2, "attributes") else ""
        
        # Map node1
        if node1_id and node1_type:
            node_start = time.time()
            mapped_ids = await map_node_to_knowledge_ids(node1_id, node1_type, graph_engine)
            graph_query_time += (time.time() - node_start)
            node_map_count += 1
            
            for kid in mapped_ids:
                if kid not in seen_ids:
                    knowledge_ids.append(kid)
                    seen_ids.add(kid)
        
        # Map node2
        if target_n and len(knowledge_ids) >= target_n:
            continue
            
        if node2_id and node2_type:
            node_start = time.time()
            mapped_ids = await map_node_to_knowledge_ids(node2_id, node2_type, graph_engine)
            graph_query_time += (time.time() - node_start)
            node_map_count += 1
            
            for kid in mapped_ids:
                if kid not in seen_ids:
                    knowledge_ids.append(kid)
                    seen_ids.add(kid)
    
    func_duration = (time.time() - func_start) * 1000
    logger.info(
        f"⏱️ [MAP_TRIPLETS] Total: {func_duration:.2f}ms | "
        f"Triplets: {triplet_count} | Nodes mapped: {node_map_count} | "
        f"Graph queries: {graph_query_time*1000:.2f}ms | Results: {len(knowledge_ids)}"
    )
    
    return knowledge_ids


async def map_chunk_results_to_knowledge_ids(
    chunk_results: List[Any],
    graph_engine: GraphDBInterface,
    target_n: int = None
) -> List[str]:
    """
    Map chunk search results to knowledge IDs (TextDocument.name).
    
    Preserves order from chunk ranking (first occurrence of knowledge ID wins).
    Supports early stopping when target_n unique IDs are found.
    
    Args:
        chunk_results: List of ScoredResult or dict from CHUNKS search
        graph_engine: Graph database interface
        target_n: Optional target number of unique IDs. If set, stops early once reached.
        
    Returns:
        List of knowledge IDs (may have duplicates, ordered by chunk rank)
    """
    knowledge_ids = []
    seen_ids = set()  # For deduplication while preserving order
    
    for result in chunk_results:
        # Early stopping: check if we already have enough unique IDs
        if target_n and len(knowledge_ids) >= target_n:
            logger.debug(f"Early stopping: reached {len(knowledge_ids)} unique IDs (target: {target_n})")
            break
        # Extract chunk_id from result
        chunk_id = None
        if hasattr(result, "id"):
            chunk_id = str(result.id)
        elif isinstance(result, dict):
            chunk_id = result.get("id")
        elif isinstance(result, dict) and "payload" in result:
            # Handle DocumentChunk payload
            payload = result["payload"]
            if isinstance(payload, dict):
                chunk_id = payload.get("id")
            elif hasattr(payload, "id"):
                chunk_id = str(payload.id)
        
        if chunk_id:
            doc = await get_textdocument_for_chunk(chunk_id, graph_engine)
            if doc:
                knowledge_id = await get_knowledge_id_from_textdocument(doc)
                if knowledge_id and knowledge_id not in seen_ids:
                    knowledge_ids.append(knowledge_id)
                    seen_ids.add(knowledge_id)
    
    return knowledge_ids
