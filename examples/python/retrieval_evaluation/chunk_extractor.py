"""
Extract all chunks from a list of knowledge IDs (TextDocument.name).
"""

from typing import List, Dict, Any, Set
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger("ChunkExtractor")


async def get_chunks_for_textdocument(textdocument_name: str, graph_engine: GraphDBInterface) -> List[Dict[str, Any]]:
    """
    Get all DocumentChunks that belong to a TextDocument by name.
    
    Args:
        textdocument_name: TextDocument.name (knowledge ID)
        graph_engine: Graph database interface
        
    Returns:
        List of chunks with structure:
        {
            "chunk_id": str,
            "knowledge_id": str,  # TextDocument.name
            "text": str,
            "chunk_index": Optional[int],
            "metadata": dict
        }
    """
    chunks = []
    
    try:
        # Get all nodes from graph to find TextDocument by name
        nodes, edges = await graph_engine.get_graph_data()
        
        # Find TextDocument by name
        textdocument_id = None
        for node in nodes:
            if isinstance(node, tuple) and len(node) >= 2:
                node_id, properties = node[0], node[1]
                if isinstance(properties, dict):
                    node_type = properties.get("type", "")
                    if node_type == "TextDocument":
                        doc_name = properties.get("name", "")
                        if doc_name == textdocument_name:
                            textdocument_id = str(node_id)
                            break
        
        if not textdocument_id:
            logger.warning(f"TextDocument with name '{textdocument_name}' not found")
            return []
        
        # Find all DocumentChunks with is_part_of relationship to this TextDocument
        chunk_ids = set()
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) >= 3:
                source_id, target_id, relationship_name = edge[0], edge[1], edge[2]
                
                # For "is_part_of" edge: source=DocumentChunk, target=TextDocument
                if relationship_name == "is_part_of" and target_id == textdocument_id:
                    chunk_ids.add(source_id)
        
        # Retrieve chunk nodes
        for chunk_id in chunk_ids:
            try:
                chunk_node = await graph_engine.get_node(chunk_id)
                if not chunk_node:
                    continue
                
                # Extract chunk data
                if isinstance(chunk_node, dict):
                    chunk_text = chunk_node.get("text", "")
                    chunk_index = chunk_node.get("chunk_index")
                    chunk_metadata = chunk_node.get("metadata", {})
                    # Also check properties if text not in top level
                    if not chunk_text and "properties" in chunk_node:
                        props = chunk_node["properties"]
                        if isinstance(props, dict):
                            chunk_text = props.get("text", "")
                            chunk_index = props.get("chunk_index")
                            chunk_metadata = props.get("metadata", {})
                else:
                    chunk_text = getattr(chunk_node, "text", "")
                    chunk_index = getattr(chunk_node, "chunk_index", None)
                    chunk_metadata = getattr(chunk_node, "metadata", {})
                
                if chunk_text:  # Only add chunks with text
                    chunks.append({
                        "chunk_id": chunk_id,
                        "knowledge_id": textdocument_name,
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "metadata": chunk_metadata if isinstance(chunk_metadata, dict) else {},
                    })
            except Exception as e:
                logger.warning(f"Error retrieving chunk {chunk_id}: {e}")
                continue
        
        logger.debug(f"Found {len(chunks)} chunks for TextDocument '{textdocument_name}'")
        
    except Exception as e:
        logger.error(f"Error extracting chunks for TextDocument '{textdocument_name}': {e}")
    
    return chunks


async def extract_chunks_from_knowledge_ids(
    knowledge_ids: List[str],
    graph_engine: GraphDBInterface
) -> List[Dict[str, Any]]:
    """
    Extract all chunks from a list of knowledge IDs (TextDocument.name).
    
    Args:
        knowledge_ids: List of knowledge IDs (TextDocument.name)
        graph_engine: Graph database interface
        
    Returns:
        List of chunks (can be more than len(knowledge_ids) since one knowledge ID can have multiple chunks)
        Structure:
        {
            "chunk_id": str,
            "knowledge_id": str,  # TextDocument.name
            "text": str,
            "chunk_index": Optional[int],
            "metadata": dict
        }
    """
    all_chunks = []
    seen_chunk_ids = set()  # Deduplicate chunks
    
    for knowledge_id in knowledge_ids:
        chunks = await get_chunks_for_textdocument(knowledge_id, graph_engine)
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)
    
    logger.info(f"Extracted {len(all_chunks)} unique chunks from {len(knowledge_ids)} knowledge IDs")
    return all_chunks
