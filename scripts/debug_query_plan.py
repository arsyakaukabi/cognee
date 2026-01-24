import os
import asyncio
import asyncpg
from dotenv import load_dotenv
import random

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "10.213.224.113")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bribrain_knowledge_base")
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

        # Generate a random vector string
        vector = [0.01] * 3072
        vector_str = f"[{','.join(map(str, vector))}]"
        
        # Test on one table
        table = "DocumentChunk_text"
        
        # Query matching the one in PGVectorAdapter
        # CAST("DocumentChunk_text".vector AS halfvec(3072)) <-> $1
        
        print("Forcing index usage...")
        await conn.execute("SET enable_seqscan = off;")
        
        query = f"""
        EXPLAIN ANALYZE
        SELECT id, payload, vector, 
               CAST(vector AS halfvec(3072)) <=> '{vector_str}'::halfvec(3072) AS similarity 
        FROM "{table}" 
        ORDER BY similarity 
        LIMIT 50;
        """
        
        print("Running EXPLAIN ANALYZE...")
        rows = await conn.fetch(query)
        
        for row in rows:
            print(row['QUERY PLAN'])

        await conn.close()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_query_plan())
