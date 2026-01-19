# coba_inspect_graph.py
import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine

async def main():
    graph = await get_graph_engine()

    print("=== SAMPLE NODES ===")
    nodes = await graph.query("MATCH (n) RETURN n LIMIT 5;")
    print(nodes)

    print("\n=== SAMPLE EDGES ===")
    edges = await graph.query("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 5;")
    print(edges)

    print("\n=== COUNT NODES ===")
    count_nodes = await graph.query("MATCH (n) RETURN COUNT(n);")
    print(count_nodes)

    print("\n=== COUNT EDGES ===")
    count_edges = await graph.query("MATCH ()-[r]->() RETURN COUNT(r);")
    print(count_edges)

if __name__ == "__main__":
    asyncio.run(main())
