"""
Map search results (triplets/chunks) to knowledge IDs (TextDocument.name).
"""

from typing import List, Dict, Any, Set, Optional
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger("ResultMapper")


async def get_textdocument_for_chunk(chunk_id: str, graph_engine: GraphDBInterface) -> Optional[Dict[str, Any]]:
    """
    Get TextDocument node for a given DocumentChunk.
    
    Args:
        chunk_id: DocumentChunk node ID
        graph_engine: Graph database interface
        
    Returns:
        TextDocument node data (dict) or None if not found
    """
    try:
        edges = await graph_engine.get_edges(chunk_id)
        
        for edge in edges:
            relationship_name = None
            potential_doc_id = None
            
            # Handle different edge formats
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    # EdgeData format: (source_id, target_id, relationship_name, properties)
                    source_id, target_id, relationship_name, _ = edge
                    # For "is_part_of" edge: source=DocumentChunk, target=TextDocument
                    if relationship_name == "is_part_of" and source_id == chunk_id:
                        potential_doc_id = target_id
                elif len(edge) == 3:
                    # Kuzu format: (node1_dict, relationship_name, node2_dict)
                    node1_dict, relationship_name, node2_dict = edge
                    node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                    node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                    
                    if relationship_name == "is_part_of":
                        # If chunk is node1, then node2 is the document
                        if node1_id == chunk_id:
                            potential_doc_id = node2_id
                        elif node2_id == chunk_id:
                            potential_doc_id = node1_id
            
            if potential_doc_id and relationship_name == "is_part_of":
                try:
                    doc_node = await graph_engine.get_node(potential_doc_id)
                    if doc_node:
                        node_type = doc_node.get("type", "") if isinstance(doc_node, dict) else getattr(doc_node, "type", "")
                        if node_type == "TextDocument":
                            return doc_node if isinstance(doc_node, dict) else {
                                "id": getattr(doc_node, "id", None),
                                "name": getattr(doc_node, "name", None),
                                "type": node_type,
                            }
                except Exception as e:
                    logger.warning(f"Error verifying document node {potential_doc_id}: {e}")
                    continue
                    
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
    
    Args:
        entity_id: Entity node ID
        graph_engine: Graph database interface
        
    Returns:
        List of DocumentChunk IDs
    """
    chunk_ids = []
    try:
        edges = await graph_engine.get_edges(entity_id)
        
        for edge in edges:
            relationship_name = None
            potential_chunk_id = None
            
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    source_id, target_id, relationship_name, _ = edge
                    # For "contains" edge: source=DocumentChunk, target=Entity
                    if relationship_name == "contains" and target_id == entity_id:
                        potential_chunk_id = source_id
                elif len(edge) == 3:
                    node1_dict, relationship_name, node2_dict = edge
                    node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                    node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                    
                    if relationship_name == "contains":
                        if node1_id == entity_id:
                            potential_chunk_id = node2_id
                        elif node2_id == entity_id:
                            potential_chunk_id = node1_id
            
            if potential_chunk_id and relationship_name == "contains":
                try:
                    chunk_node = await graph_engine.get_node(potential_chunk_id)
                    if chunk_node:
                        node_type = chunk_node.get("type", "") if isinstance(chunk_node, dict) else getattr(chunk_node, "type", "")
                        if node_type == "DocumentChunk":
                            chunk_ids.append(potential_chunk_id)
                except Exception as e:
                    logger.warning(f"Error verifying chunk node {potential_chunk_id}: {e}")
                    continue
                    
    except Exception as e:
        logger.warning(f"Error querying edges for entity {entity_id}: {e}")
    
    return chunk_ids


async def get_entities_for_entitytype(entitytype_id: str, graph_engine: GraphDBInterface) -> List[str]:
    """
    Get all Entity IDs that have a given EntityType.
    
    Args:
        entitytype_id: EntityType node ID
        graph_engine: Graph database interface
        
    Returns:
        List of Entity IDs
    """
    entity_ids = []
    try:
        edges = await graph_engine.get_edges(entitytype_id)
        
        for edge in edges:
            relationship_name = None
            potential_entity_id = None
            
            if isinstance(edge, tuple):
                if len(edge) == 4:
                    source_id, target_id, relationship_name, _ = edge
                    # For "is_a" edge: source=Entity, target=EntityType
                    if relationship_name == "is_a" and target_id == entitytype_id:
                        potential_entity_id = source_id
                elif len(edge) == 3:
                    node1_dict, relationship_name, node2_dict = edge
                    node1_id = str(node1_dict.get("id", "")) if isinstance(node1_dict, dict) else None
                    node2_id = str(node2_dict.get("id", "")) if isinstance(node2_dict, dict) else None
                    
                    if relationship_name == "is_a":
                        if node1_id == entitytype_id:
                            potential_entity_id = node2_id
                        elif node2_id == entitytype_id:
                            potential_entity_id = node1_id
            
            if potential_entity_id and relationship_name == "is_a":
                try:
                    entity_node = await graph_engine.get_node(potential_entity_id)
                    if entity_node:
                        node_type = entity_node.get("type", "") if isinstance(entity_node, dict) else getattr(entity_node, "type", "")
                        if node_type == "Entity":
                            entity_ids.append(potential_entity_id)
                except Exception as e:
                    logger.warning(f"Error verifying entity node {potential_entity_id}: {e}")
                    continue
                    
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
    
    Args:
        triplets: List of Edge objects (triplets) from GraphCompletionRetriever
        graph_engine: Graph database interface
        target_n: Optional target number of unique IDs. If set, stops early once reached.
        
    Returns:
        List of knowledge IDs (may have duplicates, ordered by triplet rank)
    """
    knowledge_ids = []
    seen_ids = set()  # For deduplication while preserving order
    
    for triplet in triplets:
        # Early stopping: check if we already have enough unique IDs
        if target_n and len(knowledge_ids) >= target_n:
            logger.debug(f"Early stopping: reached {len(knowledge_ids)} unique IDs (target: {target_n})")
            break
            
        # Extract nodes from triplet
        node1_id = str(triplet.node1.id) if hasattr(triplet.node1, "id") else None
        node2_id = str(triplet.node2.id) if hasattr(triplet.node2, "id") else None
        
        node1_type = triplet.node1.attributes.get("type", "") if hasattr(triplet.node1, "attributes") else ""
        node2_type = triplet.node2.attributes.get("type", "") if hasattr(triplet.node2, "attributes") else ""
        
        # Map node1
        if node1_id and node1_type:
            mapped_ids = await map_node_to_knowledge_ids(node1_id, node1_type, graph_engine)
            for kid in mapped_ids:
                if kid not in seen_ids:
                    knowledge_ids.append(kid)
                    seen_ids.add(kid)
        
        # Map node2 (skip if we already have enough)
        if target_n and len(knowledge_ids) >= target_n:
            continue
            
        if node2_id and node2_type:
            mapped_ids = await map_node_to_knowledge_ids(node2_id, node2_type, graph_engine)
            for kid in mapped_ids:
                if kid not in seen_ids:
                    knowledge_ids.append(kid)
                    seen_ids.add(kid)
    
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
