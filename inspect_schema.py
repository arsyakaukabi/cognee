import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine

async def inspect():
    graph_engine = await get_graph_engine()
    
    # query to get a TextDocument and its properties
    query_doc = """
    MATCH (doc:TextDocument)
    RETURN doc
    LIMIT 1
    """
    results_doc = await graph_engine.query(query_doc)
    print("--- TextDocument ---")
    if results_doc:
        for row in results_doc:
            print(row[0].properties if hasattr(row[0], 'properties') else row[0])

    # query to get a DocumentChunk and its properties
    query_chunk = """
    MATCH (chunk:DocumentChunk)
    RETURN chunk
    LIMIT 1
    """
    results_chunk = await graph_engine.query(query_chunk)
    print("\n--- DocumentChunk ---")
    if results_chunk:
        for row in results_chunk:
            print(row[0].properties if hasattr(row[0], 'properties') else row[0])

    # query to check relationship chunk -> summary
    query_summary = """
    MATCH (chunk:DocumentChunk)-[e]-(summary:TextSummary)
    RETURN chunk.id, e.relationship_name, summary
    LIMIT 1
    """
    results_summary = await graph_engine.query(query_summary)
    print("\n--- Summary Relationship ---")
    if results_summary:
        print(results_summary)

if __name__ == "__main__":
    asyncio.run(inspect())
