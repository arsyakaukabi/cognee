"""
Retrieval endpoint for testing purposes.
Returns document IDs and their knowledge types from CHUNKS or GRAPH_COMPLETION search.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

import cognee
from cognee.modules.search.types import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.custom import search_knowledge_ids


class RetrievalResult(BaseModel):
    """Single retrieval result with document ID and knowledge type."""
    id_knowledge: str
    knowledge_type: str


def parse_document_name(doc_name: str) -> Tuple[str, str]:
    """
    Parse document name into (id_knowledge, knowledge_type).
    
    Document names are formatted as: category__id
    Examples:
        - "produk__9gSth4cMEicXgesCyi4dcL" -> ("9gSth4cMEicXgesCyi4dcL", "produk")
        - "wi__CXF3s535h4wvcsQ85Pgg9C" -> ("CXF3s535h4wvcsQ85Pgg9C", "wi")
        - "helpdesk__22y7j4AeCpcVQqdZ98bjra" -> ("22y7j4AeCpcVQqdZ98bjra", "helpdesk")
    """
    # # Prefer the canonical delimiter (`category__id`), but accept legacy `category_id`.
    # if "__" in doc_name:
    #     parts = doc_name.split("__", 1)
    #     return parts[1], parts[0]  # (id, category)
    if "_" in doc_name:
        parts = doc_name.split("_", 1)
        return parts[1], parts[0]  # (id, category)
    return doc_name, "unknown"


async def _get_text_document_name_from_chunk(graph_engine, chunk_id: str) -> Optional[str]:
    """Get the TextDocument name that a chunk belongs to via is_part_of relationship."""
    try:
        connections = await graph_engine.get_connections(str(chunk_id))
        for source, relationship, target in connections:
            relationship_name = relationship.get("relationship_name")
            if relationship_name != "is_part_of":
                continue
            if target.get("type") == "TextDocument":
                return target.get("name")
            if source.get("type") == "TextDocument":
                return source.get("name")
    except Exception:
        pass
    return None


def _triplet_score(edge) -> float:
    """Calculate triplet score - lower is better."""
    n1 = edge.node1.attributes.get("vector_distance", 1)
    n2 = edge.node2.attributes.get("vector_distance", 1)
    e = edge.attributes.get("vector_distance", 1)
    return n1 + n2 + e


async def retrieve_chunks(query: str, top_k: int = 10) -> List[RetrievalResult]:
    """
    Retrieve documents using CHUNKS search type.
    
    Returns list of RetrievalResult with id_knowledge and knowledge_type.
    """
    graph_engine = await get_graph_engine()
    
    # Search for chunks
    chunks = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text=query,
        top_k=top_k * 3  # Get more chunks to find unique documents
    )
    
    if not chunks:
        return []
    
    # Extract document names from chunks
    doc_scores: Dict[str, float] = {}
    cache: Dict[str, Optional[str]] = {}
    
    for i, chunk in enumerate(chunks):
        chunk_id = None
        score = i  # Use index as score (lower = better)
        
        if hasattr(chunk, 'id'):
            chunk_id = str(chunk.id)
        elif isinstance(chunk, dict):
            chunk_id = str(chunk.get('id', chunk.get('chunk_id', '')))
            doc_name = chunk.get('document_name') or chunk.get('text_document_name')
            if doc_name:
                if doc_name not in doc_scores or score < doc_scores[doc_name]:
                    doc_scores[doc_name] = score
                continue
        
        if not chunk_id:
            continue
        
        if chunk_id not in cache:
            cache[chunk_id] = await _get_text_document_name_from_chunk(graph_engine, chunk_id)
        
        name = cache[chunk_id]
        if name and (name not in doc_scores or score < doc_scores[name]):
            doc_scores[name] = score
    
    # Sort by score and take top_k
    ranked = sorted(doc_scores.items(), key=lambda x: x[1])[:top_k]
    
    # Convert to RetrievalResult
    results = []
    for doc_name, _ in ranked:
        id_knowledge, knowledge_type = parse_document_name(doc_name)
        results.append(RetrievalResult(
            id_knowledge=id_knowledge,
            knowledge_type=knowledge_type
        ))
    
    return results


async def retrieve_graph_completion(query: str, top_k: int = 10) -> List[RetrievalResult]:
    """
    Retrieve documents using GRAPH_COMPLETION (triplets) search type.
    
    Returns list of RetrievalResult with id_knowledge and knowledge_type.
    """
    graph_engine = await get_graph_engine()
    retriever = GraphCompletionRetriever(top_k=top_k * 3)
    
    # Get triplets from graph completion retriever
    triplets = await retriever.get_context(query)
    
    if not triplets:
        return []
    
    # Extract document names from triplets
    doc_scores: Dict[str, float] = {}
    cache: Dict[str, Optional[str]] = {}
    
    for triplet in triplets:
        score = _triplet_score(triplet)
        for node in (triplet.node1, triplet.node2):
            if node.attributes.get("type") != "DocumentChunk":
                continue
            chunk_id = str(node.id)
            if chunk_id not in cache:
                cache[chunk_id] = await _get_text_document_name_from_chunk(
                    graph_engine, chunk_id
                )
            name = cache[chunk_id]
            if not name:
                continue
            if name not in doc_scores or score < doc_scores[name]:
                doc_scores[name] = score
    
    # Sort by score and take top_k
    ranked = sorted(doc_scores.items(), key=lambda x: x[1])[:top_k]
    
    # Convert to RetrievalResult
    results = []
    for doc_name, _ in ranked:
        id_knowledge, knowledge_type = parse_document_name(doc_name)
        results.append(RetrievalResult(
            id_knowledge=id_knowledge,
            knowledge_type=knowledge_type
        ))
    
    return results


async def retrieve(query: str, top_k: int = 10, search_type: str = "chunks") -> List[RetrievalResult]:
    """
    Main retrieval function.
    
    Args:
        query: Search query text
        top_k: Maximum number of results to return
        search_type: Either "chunks" or "graph_completion"
    
    Returns:
        List of RetrievalResult with id_knowledge and knowledge_type
    """
    if search_type.lower() == "graph_completion":
        return await retrieve_graph_completion(query, top_k)
    elif search_type.lower() == "graph_completion_custom":
        doc_names = await search_knowledge_ids(query, method="graph", target_n=top_k)
        results = []
        for doc_name in doc_names:
            id_knowledge, knowledge_type = parse_document_name(doc_name)
            results.append(RetrievalResult(
                id_knowledge=id_knowledge,
                knowledge_type=knowledge_type
            ))
        return results
    else:
        return await retrieve_chunks(query, top_k)
