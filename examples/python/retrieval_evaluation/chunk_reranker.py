"""
Rerank chunks using pgvector similarity with query.
Only calculates similarity for the provided chunk IDs (not the entire collection).
"""

from typing import List, Dict, Any, Optional
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger("ChunkReranker")


async def rerank_chunks_by_similarity(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using pgvector similarity with query.
    
    Only calculates similarity for the provided chunk IDs (not the entire collection).
    This is more efficient than searching the entire collection and filtering.
    
    Args:
        chunks: List of chunks to rerank
        query: Query text for similarity calculation
        top_k: Optional limit on number of chunks to return
        
    Returns:
        Reranked chunks with similarity scores, ordered by similarity (highest first)
        Structure:
        {
            "chunk_id": str,
            "knowledge_id": str,
            "text": str,
            "chunk_index": Optional[int],
            "metadata": dict,
            "similarity_score": float  # 0.0 to 1.0 (higher = more similar)
        }
    """
    if not chunks:
        return []
    
    # Extract chunk IDs
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")]
    
    if not chunk_ids:
        logger.warning("No chunk IDs found in chunks list")
        return []
    
    # Get vector engine
    vector_engine = get_vector_engine()
    
    try:
        # Step 1: Embed the query text once
        query_embedding = (await vector_engine.embedding_engine.embed_text([query]))[0]
        
        # Step 2: Get embeddings for specific chunk IDs only
        # For PGVector, we need to query the database directly to get vectors
        chunk_id_to_embedding = {}
        
        # Check if this is PGVector adapter
        adapter_name = vector_engine.__class__.__name__
        
        if adapter_name == "PGVectorAdapter":
            # Use PGVector's native operator syntax for cosine similarity
            # Reference: https://github.com/pgvector/pgvector
            # <=> is cosine distance operator, 1 - (embedding <=> query) gives cosine similarity
            # Since vectors are already in the database, we query directly by chunk IDs
            from sqlalchemy import select, text
            from sqlalchemy.sql import func
            PGVectorDataPoint = await vector_engine.get_table("DocumentChunk_text")
            
            async with vector_engine.get_async_session() as session:
                # Use PGVector's cosine_distance() method which uses <=> operator internally
                # Formula: 1 - (embedding <=> query_vector) = cosine_similarity
                # This calculates similarity directly in the database without retrieving vectors
                # The vector is already in the database, we just query by chunk IDs
                query_db = select(
                    PGVectorDataPoint.c.id,
                    (1 - PGVectorDataPoint.c.vector.cosine_distance(query_embedding)).label("similarity")
                ).where(PGVectorDataPoint.c.id.in_(chunk_ids))
                
                results = await session.execute(query_db)
                
                for row in results.all():
                    chunk_id = str(row.id)
                    similarity = row.similarity
                    if similarity is not None:
                        # similarity is already calculated as 1 - cosine_distance (which uses <=> operator)
                        # Range: 1.0 = identical, -1.0 = opposite, we clamp to [0.0, 1.0]
                        similarity_score = max(0.0, min(1.0, float(similarity)))
                        chunk_id_to_embedding[chunk_id] = similarity_score
        else:
            # For non-PGVector adapters, use search() which already calculates similarity
            # This uses the adapter's built-in similarity calculation (e.g., ChromaDB's cosine distance)
            logger.warning(f"Using search() method for {adapter_name}. "
                          "Consider using PGVector adapter for optimal performance.")
            
            # Use search with query_vector to get similarity scores directly
            search_results = await vector_engine.search(
                collection_name="DocumentChunk_text",
                query_vector=query_embedding,
                limit=len(chunk_ids) * 10  # Get more results to try to cover our chunks
            )
            
            # Create mapping: chunk_id -> similarity score from search results
            chunk_id_set = set(chunk_ids)
            for result in search_results:
                result_id = str(result.id) if hasattr(result, "id") else None
                if result_id and result_id in chunk_id_set:
                    # Get score from search result (already calculated by adapter)
                    # Note: score might be distance (lower = more similar) or similarity (higher = more similar)
                    # We'll normalize it to similarity score (higher = more similar)
                    score = result.score if hasattr(result, "score") else 0.0
                    
                    # For most adapters, score is normalized distance (0-1, lower = more similar)
                    # Convert to similarity (higher = more similar)
                    # If score is already similarity, this will still work (just inverted)
                    similarity_score = max(0.0, 1.0 - float(score))
                    chunk_id_to_embedding[result_id] = similarity_score
            
            # If we couldn't get scores for any chunks, return without reranking
            if not chunk_id_to_embedding:
                logger.warning("Could not retrieve similarity scores for chunks. Returning chunks without reranking.")
                return chunks
        
        # Step 3: Add similarity scores to chunks
        reranked_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_id_to_embedding:
                # chunk_id_to_embedding already contains similarity scores (calculated by database)
                similarity_score = chunk_id_to_embedding[chunk_id]
                
                chunk_with_score = {
                    **chunk,
                    "similarity_score": similarity_score
                }
                reranked_chunks.append(chunk_with_score)
            else:
                # If chunk not found in vector DB, assign 0.0 similarity
                logger.debug(f"Chunk {chunk_id} not found in vector database or missing embedding")
                chunk_with_score = {
                    **chunk,
                    "similarity_score": 0.0
                }
                reranked_chunks.append(chunk_with_score)
        
        # Step 4: Sort by similarity score (highest first)
        reranked_chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        
        # Step 5: Apply top_k if specified
        if top_k is not None and top_k > 0:
            reranked_chunks = reranked_chunks[:top_k]
        
        logger.info(f"Reranked {len(reranked_chunks)} chunks by similarity "
                   f"(calculated for {len(chunk_ids)} specific chunks only)")
        return reranked_chunks
        
    except Exception as e:
        logger.error(f"Error reranking chunks: {e}")
        import traceback
        traceback.print_exc()
        # Return chunks without scores if reranking fails
        return chunks
