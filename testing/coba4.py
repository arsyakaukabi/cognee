import asyncio

from cognee.modules.graph.utils.resolve_edges_to_text import _create_title_from_text, _get_top_n_frequent_words
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever

USER_QUERY = "Berapa bunga briguna karya"

def get_document_chunk_nodes(triplet):
    nodes = [triplet.node1, triplet.node2]
    return [n for n in nodes if n.attributes.get("type") == "DocumentChunk"]

async def main():
    retriever = GraphCompletionRetriever(top_k=5)
    triplets = await retriever.get_context(USER_QUERY)

    if not triplets:
        print("No triplets found.")
        return

    # Ini adalah context yang dipakai sebelum LLM dipanggil
    context_text = await retriever.resolve_edges_to_text(triplets)

    print("=== TRIPLETS ===")
    # for t in triplets:
    #     print(t)
    #     doc_chunks = get_document_chunk_nodes(t)
    #     if doc_chunks:
    #         print("  DocumentChunk nodes:")
    #         for n in doc_chunks:
    #             description = n.attributes.get("description")
    #             text = n.attributes.get("text")
    #             print(f"    id={n.id} name={n.attributes.get('name')}")
    #             if text:
    #                 print(f"    text={text}")
    #             if description:
    #                 print(f"    description={description}")

    print("\n=== CONTEXT_TEXT (pre-LLM) ===")
    print(context_text)
    # nodes = {}

    # for edge in triplets:
    #     for node in (edge.node1, edge.node2):
    #         if node.id  in nodes:
    #             continue
    #         if node.attributes.get("type") != "DocumentChunk":
    #             print("kenaaaaa", node.attributes.get("type"))
    #             continue
                
    #         text = node.attributes.get("text")
    #         text = node.attributes.get("text")

    #         description = node.attributes.get("description")
    #         if text:
    #             name = _create_title_from_text(text)
    #             content = text
    #             if description:
    #                 content = f"{content}\n\nDescription: {description}"
    #         else:
    #             name = node.attributes.get("name", "Unnamed Node")
    #             content = description or name

    #         nodes[node.id] = {"node": node, "name": name, "content": content}
    #         print("nodes mantappp",nodes)

if __name__ == "__main__":
    asyncio.run(main())
