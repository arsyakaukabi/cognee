import asyncio

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever
from cognee.infrastructure.databases.graph import get_graph_engine

USER_QUERY = "jelaskan resiko likuiditas sukuk tabungan"

async def main():
    retriever = NaturalLanguageRetriever()
    graph_engine = await get_graph_engine()

    print("=== STEP 1: CHECK GRAPH EMPTY ===")
    is_empty = await graph_engine.is_empty()
    print("input: (graph_engine)")
    print("output:", is_empty)
    if is_empty:
        print("Graph empty, stop.")
        return

    print("\n=== STEP 2: GET GRAPH SCHEMA ===")
    node_schemas, edge_schemas = await retriever._get_graph_schema(graph_engine)
    print("input: (graph_engine)")
    print("output node_schemas:", node_schemas)
    print("output edge_schemas:", edge_schemas)

# ... setelah ambil edge_schemas dan previous_attempts
    system_prompt = render_prompt(
        retriever.system_prompt_path,
        context={
            "edge_schemas": edge_schemas,
            "previous_attempts": "No attempts yet",
        },
    )
    print("\n=== FULL SYSTEM PROMPT ===")
    print(system_prompt)

    print("\n=== STEP 3: GENERATE CYPHER ===")
    cypher_query = await retriever._generate_cypher_query(
        USER_QUERY, edge_schemas, previous_attempts="No attempts yet"
    )
    print("input query:", USER_QUERY)
    print("input edge_schemas:", edge_schemas)
   
   
    print("\n=== STEP 3.5: LLM OUTPUT (RAW) ===")
    print("output cypher_query:\n", repr(cypher_query))

    print("\n=== STEP 4: EXECUTE CYPHER ===")
    result = await graph_engine.query(cypher_query)
    print("input cypher_query:", cypher_query)
    print("output result:", result)

    print("\n=== STEP 5: FINAL CONTEXT ===")
    print("context:", result)

if __name__ == "__main__":
    asyncio.run(main())
