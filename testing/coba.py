import os, asyncio
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
    NeptuneAnalyticsAdapter,
)
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

# Prereqs: export GRAPH_ID (and AWS credentials/region) so the adapter can connect.
# GRAPH_ID = os.environ["GRAPH_ID"]

# HAPUS ini:
# GRAPH_ID = os.environ["GRAPH_ID"]

# Ganti dengan init graph + vector seperti biasa:
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

async def hybrid_retrieve(query: str, top_k: int = 5):
    graph = await get_graph_engine()   # Kuzu dari .env
    vector = get_vector_engine()       # Vector provider dari .env

    vector_hits = await vector.search("DocumentChunk_text", query, limit=top_k)
    node_ids = [hit.id for hit in vector_hits]

    graph_nodes = await graph.get_nodes(node_ids) if node_ids else []
    # optional: neighbors via graph.query / get_nodeset_subgraph

    return {
        "query": query,
        "vector_results": [hit.payload for hit in vector_hits],
        "graph_nodes": graph_nodes,
    }


if __name__ == "__main__":
    result = asyncio.run(hybrid_retrieve("Berapa bunga briguna karya"))
    print(result)
