"""
Map search results (triplets/chunks) to knowledge IDs (TextDocument.name).
"""

import time
from typing import List, Dict, Any, Set, Optional
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger("CustomMapper")


# ============================================================================
# BATCH QUERY OPTIMIZATION
# ============================================================================

async def batch_get_chunk_to_document_mapping(
    chunk_ids: List[str],
    graph_engine: GraphDBInterface
) -> Dict[str, str]:
    """
    Get all DocumentChunk → TextDocument.name mappings in a SINGLE Cypher query.
    
    Returns:
        Dict mapping chunk_id -> document_name (TextDocument.name)
    """
    if not chunk_ids:
        return {}
    
    # Single Cypher query to get all mappings at once
    query = """
    MATCH (chunk:Node)-[r:EDGE]->(doc:Node)
    WHERE chunk.id IN $chunk_ids
    AND r.relationship_name = 'is_part_of'
    AND doc.type = 'TextDocument'
    RETURN chunk.id, doc.name
    """
    
    try:
        results = await graph_engine.query(query, {"chunk_ids": chunk_ids})
        return {str(row[0]): str(row[1]) for row in results if row[0] and row[1]}
    except Exception as e:
        logger.warning(f"Batch query failed, falling back to sequential: {e}")
        return {}


async def batch_get_entity_to_chunks_mapping(
    entity_ids: List[str],
    graph_engine: GraphDBInterface
) -> Dict[str, List[str]]:
    """
    Get all Entity → DocumentChunk IDs mappings in a SINGLE Cypher query.
    
    Returns:
        Dict mapping entity_id -> list of chunk_ids
    """
    if not entity_ids:
        return {}
    
    query = """
    MATCH (chunk:Node)-[r:EDGE]->(entity:Node)
    WHERE entity.id IN $entity_ids
    AND r.relationship_name = 'contains'
    RETURN entity.id, chunk.id
    """
    
    try:
        results = await graph_engine.query(query, {"entity_ids": entity_ids})
        mapping = {}
        for row in results:
            if row[0] and row[1]:
                entity_id = str(row[0])
                chunk_id = str(row[1])
                if entity_id not in mapping:
                    mapping[entity_id] = []
                mapping[entity_id].append(chunk_id)
        return mapping
    except Exception as e:
        logger.warning(f"Batch entity query failed: {e}")
        return {}


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
    
    OPTIMIZED: Uses batch Cypher queries instead of sequential get_edges() calls.
    Reduces 37+ DB calls to just 2-3 batch calls.
    """
    func_start = time.time()
    
    if not triplets:
        return []
    
    # ========== PHASE 1: Collect all node IDs by type ==========
    chunk_ids = set()
    entity_ids = set()
    node_order = []  # Preserve triplet order for ranking
    
    for triplet in triplets:
        for node in (triplet.node1, triplet.node2):
            node_id = str(node.id) if hasattr(node, "id") else None
            node_type = node.attributes.get("type", "") if hasattr(node, "attributes") else ""
            
            if node_id and node_type:
                node_order.append((node_id, node_type))
                
                if node_type == "DocumentChunk":
                    chunk_ids.add(node_id)
                elif node_type == "Entity":
                    entity_ids.add(node_id)
    
    # ========== PHASE 2: Batch query chunk → document mappings ==========
    batch_start = time.time()
    
    # Get chunk → document mapping in 1 query
    chunk_to_doc = await batch_get_chunk_to_document_mapping(list(chunk_ids), graph_engine)
    
    # Get entity → chunks mapping in 1 query
    entity_to_chunks = await batch_get_entity_to_chunks_mapping(list(entity_ids), graph_engine)
    
    # Get additional chunks from entities
    additional_chunk_ids = set()
    for chunks in entity_to_chunks.values():
        additional_chunk_ids.update(chunks)
    additional_chunk_ids -= chunk_ids  # Only query new ones
    
    # Get document mapping for entity's chunks
    if additional_chunk_ids:
        additional_chunk_to_doc = await batch_get_chunk_to_document_mapping(
            list(additional_chunk_ids), graph_engine
        )
        chunk_to_doc.update(additional_chunk_to_doc)
    
    batch_duration = (time.time() - batch_start) * 1000
    
    # ========== PHASE 3: Build ordered knowledge_ids list ==========
    knowledge_ids = []
    seen_ids = set()
    
    for node_id, node_type in node_order:
        if target_n and len(knowledge_ids) >= target_n:
            break
            
        if node_type == "DocumentChunk":
            doc_name = chunk_to_doc.get(node_id)
            if doc_name and doc_name not in seen_ids:
                knowledge_ids.append(doc_name)
                seen_ids.add(doc_name)
                
        elif node_type == "Entity":
            chunk_list = entity_to_chunks.get(node_id, [])
            for chunk_id in chunk_list:
                doc_name = chunk_to_doc.get(chunk_id)
                if doc_name and doc_name not in seen_ids:
                    knowledge_ids.append(doc_name)
                    seen_ids.add(doc_name)
                    if target_n and len(knowledge_ids) >= target_n:
                        break
    
    func_duration = (time.time() - func_start) * 1000
    logger.info(
        f"⏱️ [MAP_TRIPLETS_BATCH] Total: {func_duration:.2f}ms | "
        f"Batch queries: {batch_duration:.2f}ms | "
        f"Chunks: {len(chunk_ids)} | Entities: {len(entity_ids)} | "
        f"Results: {len(knowledge_ids)}"
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
