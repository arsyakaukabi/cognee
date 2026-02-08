"""
Retrieval endpoint for testing purposes.
Returns document IDs and their knowledge types from CHUNKS or GRAPH_COMPLETION search.
"""

import json
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

import cognee
from cognee.modules.search.types import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.custom import search_knowledge_ids


class RetrievalResult(BaseModel):
    """Single retrieval result with document ID, knowledge type, optional summary, and filename."""
    id_knowledge: str
    knowledge_type: str
    summary: Optional[str] = None
    filename: Optional[str] = None


def parse_document_name(doc_name: str) -> Tuple[str, str]:
    """
    Parse document name into (id_knowledge, knowledge_type).
    
    Document names are formatted as: category_id
    Examples:
        - "produk_9gSth4cMEicXgesCyi4dcL" -> ("9gSth4cMEicXgesCyi4dcL", "produk")
        - "wi_CXF3s535h4wvcsQ85Pgg9C" -> ("CXF3s535h4wvcsQ85Pgg9C", "wi")
        - "helpdesk_22y7j4AeCpcVQqdZ98bjra" -> ("22y7j4AeCpcVQqdZ98bjra", "helpdesk")
    """
    # # Prefer the canonical delimiter (`category_id`), but accept legacy `category_id`.
    # if "_" in doc_name:
    #     parts = doc_name.split("_", 1)
    #     return parts[1], parts[0]  # (id, category)
    if "_" in doc_name:
        parts = doc_name.split("_", 1)
        return parts[1], parts[0]  # (id, category)
    return doc_name, "unknown"


# Known prefixes that should use parse_document_name
KNOWN_PREFIXES = ("produk_", "promo_", "program_", "wi_", "helpdesk_")


def get_knowledge_id_and_type(doc_name: str, doc_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Get id_knowledge and knowledge_type based on doc_name.
    
    If doc_name starts with known prefixes (produk_, promo_, program_, wi_, helpdesk_),
    use parse_document_name to extract id and type.
    
    Otherwise, use the provided doc_id (if available) or fallback to doc_name,
    and set knowledge_type to \"others\".
    
    Args:
        doc_name: Document name
        doc_id: Optional document ID from graph (avoids DB query)
        
    Returns:
        Tuple of (id_knowledge, knowledge_type)
    """
    # Check if doc_name starts with known prefixes
    if doc_name.startswith(KNOWN_PREFIXES):
        return parse_document_name(doc_name)
    
    # For other documents, use provided doc_id or fallback to doc_name
    id_knowledge = doc_id if doc_id else doc_name
    return id_knowledge, "others"


async def _get_text_document_info_from_chunk(graph_engine, chunk_id: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Get the TextDocument name and id that a chunk belongs to via is_part_of relationship.
    
    Returns:
        Tuple of (name, id) or None if not found
    """
    try:
        connections = await graph_engine.get_connections(str(chunk_id))
        for source, relationship, target in connections:
            relationship_name = relationship.get("relationship_name")
            if relationship_name != "is_part_of":
                continue
            if target.get("type") == "TextDocument":
                return target.get("name"), str(target.get("id")) if target.get("id") else None
            if source.get("type") == "TextDocument":
                return source.get("name"), str(source.get("id")) if source.get("id") else None
    except Exception:
        pass
    return None


def _triplet_score(edge) -> float:
    """Calculate triplet score - lower is better."""
    n1 = edge.node1.attributes.get("vector_distance", 1)
    n2 = edge.node2.attributes.get("vector_distance", 1)
    e = edge.attributes.get("vector_distance", 1)
    return n1 + n2 + e


async def _get_summaries_batch(graph_engine, doc_names: List[str]) -> Dict[str, str]:
    """
    Batch fetch TextSummary for multiple documents in a single query.
    
    This is optimized to avoid N+1 query problem by fetching all summaries at once.
    
    Path: TextDocument <- DocumentChunk <- TextSummary
    
    Args:
        graph_engine: Graph database engine
        doc_names: List of document names to fetch summaries for
        
    Returns:
        Dict mapping doc_name -> first summary text (or empty dict if no summaries)
    """
    if not doc_names:
        return {}
    
    # Use a single batch query to get all summaries
    # Query pattern: doc -> chunk -> summary
    query = """
    MATCH (doc:Node)-[e1:EDGE]-(chunk:Node)-[e2:EDGE]-(summary:Node)
    WHERE doc.name IN $doc_names
      AND doc.type IN ['TextDocument', 'PdfDocument', 'AudioDocument', 'ImageDocument', 'UnstructuredDocument']
      AND e1.relationship_name = 'is_part_of'
      AND chunk.type = 'DocumentChunk'
      AND e2.relationship_name = 'made_from'
      AND summary.type = 'TextSummary'
    RETURN doc.name, summary.properties
    """
    
    try:
        results = await graph_engine.query(query, {"doc_names": doc_names})
    except Exception:
        # Fallback: return empty dict if query fails
        return {}
    
    # Build doc_name -> summary mapping (take first summary per document)
    summaries: Dict[str, str] = {}
    for row in results:
        if not row or len(row) < 2:
            continue
        doc_name = row[0]
        props_str = row[1]
        
        # Skip if we already have a summary for this doc
        if doc_name in summaries:
            continue
            
        # Parse properties JSON to extract summary text
        try:
            if isinstance(props_str, str):
                props = json.loads(props_str)
            elif isinstance(props_str, dict):
                props = props_str
            else:
                continue
            
            summary_text = props.get("text", "")
            if summary_text:
                summaries[doc_name] = summary_text
        except (json.JSONDecodeError, TypeError):
            continue
    
    return summaries


async def retrieve_chunks(
    query: str, top_k: int = 10, node_name: Optional[List[str]] = None
) -> List[RetrievalResult]:
    """
    Retrieve documents using CHUNKS search type.
    
    Returns list of RetrievalResult with id_knowledge and knowledge_type.
    """
    graph_engine = await get_graph_engine()
    
    # Search for chunks
    chunks = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text=query,
        top_k=top_k * 3,  # Get more chunks to find unique documents
        node_name=node_name,
    )
    
    if not chunks:
        return []
    
    # Extract document names from chunks
    # doc_scores: doc_name -> (score, doc_id)
    doc_scores: Dict[str, Tuple[float, Optional[str]]] = {}
    cache: Dict[str, Optional[Tuple[str, Optional[str]]]] = {}  # chunk_id -> (name, id)
    
    for i, chunk in enumerate(chunks):
        chunk_id = None
        score = i  # Use index as score (lower = better)
        
        if hasattr(chunk, 'id'):
            chunk_id = str(chunk.id)
        elif isinstance(chunk, dict):
            chunk_id = str(chunk.get('id', chunk.get('chunk_id', '')))
            doc_name = chunk.get('document_name') or chunk.get('text_document_name')
            if doc_name:
                if doc_name not in doc_scores or score < doc_scores[doc_name][0]:
                    # No doc_id available from dict, use None
                    doc_scores[doc_name] = (score, None)
                continue
        
        if not chunk_id:
            continue
        
        if chunk_id not in cache:
            cache[chunk_id] = await _get_text_document_info_from_chunk(graph_engine, chunk_id)
        
        info = cache[chunk_id]
        if info:
            name, doc_id = info
            if name and (name not in doc_scores or score < doc_scores[name][0]):
                doc_scores[name] = (score, doc_id)
    
    # Sort by score and take top_k
    ranked = sorted(doc_scores.items(), key=lambda x: x[1][0])[:top_k]
    
    # Convert to RetrievalResult
    results = []
    for doc_name, (_, doc_id) in ranked:
        id_knowledge, knowledge_type = get_knowledge_id_and_type(doc_name, doc_id)
        results.append(RetrievalResult(
            id_knowledge=id_knowledge,
            knowledge_type=knowledge_type
        ))
    
    return results


async def retrieve_graph_completion(
    query: str, top_k: int = 10, node_name: Optional[List[str]] = None
) -> List[RetrievalResult]:
    """
    Retrieve documents using GRAPH_COMPLETION (triplets) search type.
    
    Returns list of RetrievalResult with id_knowledge and knowledge_type.
    """
    graph_engine = await get_graph_engine()
    retriever = GraphCompletionRetriever(top_k=top_k * 3)
    
    # Get triplets from graph completion retriever
    triplets = await retriever.get_context(query, node_name=node_name)
    
    if not triplets:
        return []
    
    # Extract document names from triplets
    # doc_scores: doc_name -> (score, doc_id)
    doc_scores: Dict[str, Tuple[float, Optional[str]]] = {}
    cache: Dict[str, Optional[Tuple[str, Optional[str]]]] = {}  # chunk_id -> (name, id)
    
    for triplet in triplets:
        score = _triplet_score(triplet)
        for node in (triplet.node1, triplet.node2):
            if node.attributes.get("type") != "DocumentChunk":
                continue
            chunk_id = str(node.id)
            if chunk_id not in cache:
                cache[chunk_id] = await _get_text_document_info_from_chunk(
                    graph_engine, chunk_id
                )
            info = cache[chunk_id]
            if not info:
                continue
            name, doc_id = info
            if not name:
                continue
            if name not in doc_scores or score < doc_scores[name][0]:
                doc_scores[name] = (score, doc_id)
    
    # Sort by score and take top_k
    ranked = sorted(doc_scores.items(), key=lambda x: x[1][0])[:top_k]
    
    # Convert to RetrievalResult
    results = []
    for doc_name, (_, doc_id) in ranked:
        id_knowledge, knowledge_type = get_knowledge_id_and_type(doc_name, doc_id)
        results.append(RetrievalResult(
            id_knowledge=id_knowledge,
            knowledge_type=knowledge_type
        ))
    
    return results


async def retrieve(
    query: str,
    top_k: int = 10,
    search_type: str = "chunks",
    dataset_ids: List[str] = None,
    node_name: Optional[List[str]] = None,
) -> List[RetrievalResult]:
    """
    Main retrieval function.
    
    Args:
        query: Search query text
        top_k: Maximum number of results to return
        search_type: Either "chunks", "graph_completion", or "graph_completion_custom"
    
    Returns:
        List of RetrievalResult with id_knowledge, knowledge_type, and summary (for graph_completion_custom)
    """
    if search_type.lower() == "graph_completion":
        return await retrieve_graph_completion(query, top_k, node_name=node_name)
    elif search_type.lower() == "graph_completion_custom":
        # Get document names from custom search
        doc_names = await search_knowledge_ids(
            query, method="graph", target_n=top_k, node_name=node_name
        )
        
        if not doc_names:
            return []
        
        # Separate docs by prefix type
        docs_with_known_prefix = []
        docs_needing_id_lookup = []
        
        for doc_name in doc_names:
            if doc_name.startswith(KNOWN_PREFIXES):
                docs_with_known_prefix.append(doc_name)
            else:
                docs_needing_id_lookup.append(doc_name)
        
        # Batch fetch doc_ids (for non-prefix docs) and original_extension (for ALL docs) from Data table
        doc_id_map = {}  # doc_name -> doc_id
        doc_ext_map = {}  # doc_name -> original_extension (for ALL docs)
        
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.data.models import Data
            from sqlalchemy import select
            
            db_engine = get_relational_engine()
            async with db_engine.get_async_session() as session:
                # Fetch for ALL doc_names to get extensions
                stmt = select(
                    Data.name, 
                    Data.id, 
                    Data.original_extension,
                    Data.extension
                ).where(Data.name.in_(doc_names))
                result = await session.execute(stmt)
                for row in result:
                    doc_name_row = row[0]
                    if doc_name_row and row[1]:
                        doc_id_map[doc_name_row] = str(row[1])
                    if doc_name_row:
                        # Prefer original_extension, fallback to extension
                        ext = row[2] or row[3]
                        if ext:
                            doc_ext_map[doc_name_row] = ext if ext.startswith('.') else f".{ext}"
        except Exception:
            pass
        
        # Parse all docs and identify which ones need summaries
        parsed_docs = {}  # doc_name -> (id, type)
        docs_needing_summary = []
        excluded_types = {"produk", "promo", "program", "wi", "helpdesk"}
        
        for doc_name in doc_names:
            if doc_name.startswith(KNOWN_PREFIXES):
                id_knowledge, knowledge_type = parse_document_name(doc_name)
            else:
                # Use doc_id from batch lookup, fallback to doc_name
                doc_id = doc_id_map.get(doc_name)
                id_knowledge = doc_id if doc_id else doc_name
                knowledge_type = "others"
            
            parsed_docs[doc_name] = (id_knowledge, knowledge_type)
            
            # Case-insensitive check for exclusion
            if knowledge_type.lower() not in excluded_types:
                docs_needing_summary.append(doc_name)
        
        # Batch fetch summaries only for allowed types
        summaries = {}
        if docs_needing_summary:
            graph_engine = await get_graph_engine()
            summaries = await _get_summaries_batch(graph_engine, docs_needing_summary)
        
        # Build results with summaries and filename
        results = []
        for doc_name in doc_names:
            id_knowledge, knowledge_type = parsed_docs[doc_name]
            summary = summaries.get(doc_name)  # None if not found or excluded
            
            # Build filename from doc_name + extension
            ext = doc_ext_map.get(doc_name, "")
            filename = f"{doc_name}{ext}" if ext else doc_name
            
            results.append(RetrievalResult(
                id_knowledge=id_knowledge,
                knowledge_type=knowledge_type,
                summary=summary,
                filename=filename
            ))
        return results
    else:
        return await retrieve_chunks(query, top_k, node_name=node_name)


async def get_document_details(doc_name: str) -> Dict[str, Union[str, List[str], int, None]]:
    """
    Retrieve document filename and chunk summaries.
    
    Args:
        doc_name: Name of the document (e.g. "helpdesk_...")
        
    Returns:
        Dict with "filename", "file_size", "created_at", "updated_at", and "summaries" keys.
    """
    graph_engine = await get_graph_engine()
    
    filename = None
    file_size = None
    created_at = None
    updated_at = None
    original_extension = None
    
    # 1. Fetch metadata from relational database (Data table) - includes original_extension
    try:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.data.models import Data
        from sqlalchemy import select
        
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            # Query Data table by name - get original_extension for filename
            stmt = select(
                Data.data_size, 
                Data.created_at, 
                Data.updated_at,
                Data.original_extension,
                Data.extension
            ).where(Data.name == doc_name)
            result = await session.execute(stmt)
            row = result.first()
            if row:
                file_size = row[0]
                if row[1]:
                    created_at = int(row[1].timestamp() * 1000) if hasattr(row[1], 'timestamp') else row[1]
                if row[2]:
                    updated_at = int(row[2].timestamp() * 1000) if hasattr(row[2], 'timestamp') else row[2]
                # Prefer original_extension, fallback to extension
                original_extension = row[3] or row[4]
    except Exception as e:
        print(f"DEBUG: Error querying relational DB: {e}")
        pass
    
    # 2. Build filename from doc_name + original_extension
    if original_extension:
        # Ensure extension starts with dot
        ext = original_extension if original_extension.startswith('.') else f".{original_extension}"
        filename = f"{doc_name}{ext}"
    else:
        filename = doc_name
        

    # 2. Fetch chunk summaries
    # Path: TextDocument <- DocumentChunk -> TextSummary
    query_summaries = """
    MATCH (doc:Node)-[e1:EDGE]-(chunk:Node)-[e2:EDGE]-(summary:Node)
    WHERE doc.name = $doc_name
      AND doc.type IN ['TextDocument', 'PdfDocument', 'AudioDocument', 'ImageDocument', 'UnstructuredDocument']
      AND e1.relationship_name = 'is_part_of'
      AND chunk.type = 'DocumentChunk'
      AND e2.relationship_name = 'made_from'
      AND summary.type = 'TextSummary'
    RETURN summary.properties
    """
    
    summaries_list = []
    try:
        results_summaries = await graph_engine.query(query_summaries, {"doc_name": doc_name})
        print(f"DEBUG: Summary results: {results_summaries}")
        if results_summaries:
            for row in results_summaries:
                if not row:
                    continue
                props_str = row[0]
                
                try:
                    props = json.loads(props_str) if isinstance(props_str, str) else props_str
                    if isinstance(props, dict):
                        text = props.get("text")
                        if text:
                            summaries_list.append(text)
                except (json.JSONDecodeError, TypeError):
                    continue
    except Exception as e:
        print(f"DEBUG: Error querying summaries: {e}")
        pass
        
    return {
        "filename": filename,
        "file_size": file_size,
        "created_at": created_at,
        "updated_at": updated_at,
        "summaries": summaries_list
    }
