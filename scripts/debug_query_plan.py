import os
import asyncio
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bribrain_knowledge_base_hnsw")
DB_USER = os.getenv("DB_USERNAME", "bribrain_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Bribrainaj4!")

async def debug_query_plan():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"Connected to {DB_NAME}")

        # Use a real vector from the table to make sure dimensions match
        # We pick one existing vector to use as the query vector
        print("Fetching a sample vector to use as query...")
        sample_query = """
        SELECT (vector::text) as v_str
        FROM "EdgeType_relationship_name"
        LIMIT 1;
        """
        v_str = await conn.fetchval(sample_query)
        
        if not v_str:
            print("No data in EdgeType_relationship_name, trying Entity_name")
            v_str = await conn.fetchval('SELECT (vector::text) as v_str FROM "Entity_name" LIMIT 1')
            
        if not v_str:
            print("No data found to test.")
            return

        print("Running EXPLAIN ANALYZE...")
        
        # Exact query pattern we want to verify matches the app and index
        # Casting both sides to halfvec(3072)
        explain_query = f"""
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id
        FROM "EdgeType_relationship_name"
        ORDER BY ("vector"::halfvec(3072)) <=> '{v_str}'::halfvec(3072)
        LIMIT 10;
        """
        
        rows = await conn.fetch(explain_query)
        for row in rows:
            print(row['QUERY PLAN'])

        await conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_query_plan())
