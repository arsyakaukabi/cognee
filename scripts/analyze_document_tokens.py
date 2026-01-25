#!/usr/bin/env python3
"""
Script to analyze document tokens and chunking results.
Outputs JSON with document token counts and chunk-level token counts.

Connects to PostgreSQL for document metadata and Kuzu for chunk data.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional

import tiktoken
import asyncpg


# Database connection settings
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "bribrain_knowledge_base",
    "user": "admin",
    "password": "admin",
}


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """
    Count tokens in text using tiktoken.
    
    Uses text-embedding-ada-002 tokenizer (cl100k_base) to match
    the embedding model used by Cognee for chunking.
    
    Models and their encodings:
    - text-embedding-ada-002: cl100k_base
    - gpt-4o / gpt-4o-mini: o200k_base
    - gpt-4 / gpt-3.5-turbo: cl100k_base
    """
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (text-embedding-ada-002)
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def find_documents_by_pattern(conn, pattern: str) -> List[Dict[str, Any]]:
    """Find documents matching a pattern."""
    query = """
    SELECT id, name, token_count, data_size, raw_data_location
    FROM data
    WHERE name LIKE $1
    ORDER BY name
    """
    rows = await conn.fetch(query, f'%{pattern}%')
    return [dict(r) for r in rows]


async def get_document_info(conn, doc_name: str) -> Optional[Dict[str, Any]]:
    """Get document info from data table."""
    query = """
    SELECT id, name, token_count, data_size, raw_data_location, created_at
    FROM data
    WHERE name = $1 OR name = $2
    LIMIT 1
    """
    row = await conn.fetchrow(query, doc_name, doc_name.replace('.md', ''))
    if row:
        return dict(row)
    return None


async def get_chunks_from_kuzu(graph_engine, doc_name: str) -> List[Dict[str, Any]]:
    """Get chunks from Kuzu graph database using is_part_of relationship."""
    chunks = []
    
    # Clean doc name (remove .md extension)
    clean_name = doc_name.replace('.md', '')
    
    try:
        # Query: Find DocumentChunk nodes that are part of this TextDocument
        # The relationship is: DocumentChunk -> is_part_of -> TextDocument
        query = f'''
            MATCH (chunk:Node)-[r:EDGE]->(doc:Node)
            WHERE doc.name = "{clean_name}" AND chunk.type = "DocumentChunk"
            RETURN chunk.id, chunk.properties
        '''
        
        result = await graph_engine.query(query)
        
        if result:
            for row in result:
                chunk_id = row[0]
                properties_str = row[1]
                
                if properties_str:
                    try:
                        properties = json.loads(properties_str)
                        chunks.append({
                            "chunk_id": str(chunk_id),
                            "chunk_index": properties.get("chunk_index", 0),
                            "text": properties.get("text", ""),
                            "chunk_size": properties.get("chunk_size", 0),
                            "cut_type": properties.get("cut_type", ""),
                        })
                    except json.JSONDecodeError:
                        pass
        
        # Try reverse direction if no results
        if not chunks:
            query2 = f'''
                MATCH (doc:Node)-[r:EDGE]->(chunk:Node)
                WHERE doc.name = "{clean_name}" AND chunk.type = "DocumentChunk"
                RETURN chunk.id, chunk.properties
            '''
            result2 = await graph_engine.query(query2)
            
            if result2:
                for row in result2:
                    chunk_id = row[0]
                    properties_str = row[1]
                    
                    if properties_str:
                        try:
                            properties = json.loads(properties_str)
                            chunks.append({
                                "chunk_id": str(chunk_id),
                                "chunk_index": properties.get("chunk_index", 0),
                                "text": properties.get("text", ""),
                                "chunk_size": properties.get("chunk_size", 0),
                                "cut_type": properties.get("cut_type", ""),
                            })
                        except json.JSONDecodeError:
                            pass
                            
    except Exception as e:
        print(f"Kuzu query error for {doc_name}: {e}", file=sys.stderr)
    
    return sorted(chunks, key=lambda x: x.get("chunk_index", 0))


async def read_raw_file(file_path: str) -> Optional[str]:
    """Read raw file content if exists."""
    try:
        if file_path.startswith("file://"):
            file_path = file_path[7:]
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
    return None


async def analyze_document(conn, graph_engine, doc_name: str) -> Dict[str, Any]:
    """Analyze a single document and its chunks."""
    result = {
        "document_name": doc_name,
        "document_found": False,
        "document_tokens": 0,
        "document_text_length": 0,
        "raw_content_tokens": 0,
        "chunks": [],
        "total_chunk_tokens": 0,
        "chunk_count": 0,
    }
    
    # Get document from data table
    doc = await get_document_info(conn, doc_name)
    
    if doc:
        result["document_found"] = True
        result["document_id"] = str(doc.get('id', ''))
        result["document_tokens"] = doc.get('token_count', 0) or 0
        result["document_text_length"] = doc.get('data_size', 0) or 0
        result["raw_data_location"] = doc.get('raw_data_location', '')
        
        # Read raw file to count actual tokens
        if doc.get('raw_data_location'):
            raw_content = await read_raw_file(doc['raw_data_location'])
            if raw_content:
                result["raw_content_tokens"] = count_tokens(raw_content)
        
        # Get chunks from Kuzu
        chunks = await get_chunks_from_kuzu(graph_engine, doc_name)
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = count_tokens(chunk_text) if chunk_text else 0
            
            result["chunks"].append({
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text_length": len(chunk_text),
                "tokens": chunk_tokens,
                "chunk_size": chunk.get("chunk_size", 0),
                "cut_type": chunk.get("cut_type", ""),
                # "text_preview": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                "text_preview": chunk_text,
            })
            result["total_chunk_tokens"] += chunk_tokens
        
        result["chunk_count"] = len(result["chunks"])
    
    return result


async def main():
    """Main function to analyze specified documents."""
    
    # Documents to analyze (from user request)
    large_docs = [
        "wi__bPVUU8wEuBwUL5Uq88EKV5.md",
        "wi__hiWR4cGfWER5yNLoXA6BzU.md",
    ]
    
    small_docs = [
        "wi__3znsdbQDToCeQ2k9Ktxxx7.md",
        "wi__reDXFRvp3Ky9tFzm9pZFY7.md",
        "wi__yHmNcLPh5Ps5jwAaKyggFa.md",
    ]
    
    print(f"Connecting to database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}", file=sys.stderr)
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        print("Connected to PostgreSQL successfully!", file=sys.stderr)
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Connect to Kuzu
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cognee.infrastructure.databases.graph import get_graph_engine
    graph_engine = await get_graph_engine()
    print("Connected to Kuzu graph database!", file=sys.stderr)
    
    try:
        # Find all wi__ documents
        wi_docs = await find_documents_by_pattern(conn, 'wi__')
        
        print(f"\n=== All wi__ documents from database ===", file=sys.stderr)
        for doc in wi_docs:
            category = "> 8192" if (doc['token_count'] or 0) > 8192 else "<= 8192"
            print(f"  {doc['name']}: {doc['token_count']} tokens ({category})", file=sys.stderr)
        
        results = {
            "analysis_type": "document_token_analysis",
            "token_threshold": 8192,
            "database": DB_CONFIG['database'],
            "all_wi_documents": [
                {
                    "name": d['name'],
                    "token_count": d['token_count'],
                    "data_size": d['data_size'],
                    "category": "> 8192 tokens" if (d['token_count'] or 0) > 8192 else "<= 8192 tokens"
                }
                for d in wi_docs
            ],
            "target_large_documents": [],
            "target_small_documents": [],
            "summary": {
                "total_wi_documents": len(wi_docs),
                "docs_over_8192": len([d for d in wi_docs if (d['token_count'] or 0) > 8192]),
                "docs_under_8192": len([d for d in wi_docs if (d['token_count'] or 0) <= 8192]),
            }
        }
        
        # Analyze target large documents
        print(f"\n=== Analyzing target large documents ===", file=sys.stderr)
        for doc_name in large_docs:
            print(f"  Analyzing: {doc_name}", file=sys.stderr)
            analysis = await analyze_document(conn, graph_engine, doc_name)
            results["target_large_documents"].append(analysis)
            print(f"    Found {analysis['chunk_count']} chunks, {analysis['total_chunk_tokens']} chunk tokens", file=sys.stderr)
        
        # Analyze target small documents  
        print(f"\n=== Analyzing target small documents ===", file=sys.stderr)
        for doc_name in small_docs:
            print(f"  Analyzing: {doc_name}", file=sys.stderr)
            analysis = await analyze_document(conn, graph_engine, doc_name)
            results["target_small_documents"].append(analysis)
            print(f"    Found {analysis['chunk_count']} chunks, {analysis['total_chunk_tokens']} chunk tokens", file=sys.stderr)
        
        # Output JSON
        print("\n" + "=" * 60, file=sys.stderr)
        print("JSON OUTPUT:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        
        # Save to docs/
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "document_token_analysis.json"
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_path}", file=sys.stderr)
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
