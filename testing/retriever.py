import asyncio
from typing import List, Optional, Dict, Tuple

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.infrastructure.databases.graph import get_graph_engine

USER_QUERY = "jelaskan program beasiswa bri untuk mahasiswa"


async def _get_text_document_name_from_chunk(graph_engine, chunk_id: str) -> Optional[str]:
    connections = await graph_engine.get_connections(str(chunk_id))
    for source, relationship, target in connections:
        relationship_name = relationship.get("relationship_name")
        if relationship_name != "is_part_of":
            continue
        if target.get("type") == "TextDocument":
            return target.get("name")
        if source.get("type") == "TextDocument":
            return source.get("name")
    return None


def _triplet_score(edge) -> float:
    # Same scoring used in CogneeGraph.calculate_top_triplet_importances (lower = better).
    n1 = edge.node1.attributes.get("vector_distance", 1)
    n2 = edge.node2.attributes.get("vector_distance", 1)
    e = edge.attributes.get("vector_distance", 1)
    return n1 + n2 + e


async def get_text_document_names_from_triplets(triplets, top_k: int) -> List[str]:
    """Return unique TextDocument names ordered by best triplet score."""
    doc_scores: Dict[str, float] = {}
    graph_engine = await get_graph_engine()
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

    ranked: List[Tuple[str, float]] = sorted(doc_scores.items(), key=lambda x: x[1])
    return [name for name, _ in ranked[:top_k]]


async def main() -> None:
    top_k = 5
    retriever = GraphCompletionRetriever(top_k=top_k)
    triplets = await retriever.get_context(USER_QUERY)

    if not triplets:
        print("No triplets found.")
        return

    doc_names = await get_text_document_names_from_triplets(triplets, top_k=top_k)

    print("=== TextDocument_name (ordered by triplet rank) ===")
    print("doc_names", doc_names)
    # for idx, name in enumerate(doc_names, start=1):
    #     print(f"{idx}. {name}")


if __name__ == "__main__":
    asyncio.run(main())
