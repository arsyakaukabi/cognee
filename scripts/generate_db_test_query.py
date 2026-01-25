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

async def generate_sql_file():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"Connected to {DB_NAME}")

        # Fetch a real vector to use as query
        print("Fetching sample vector from Entity_name...")
        sample_query = """
        SELECT (vector::text) as v_str
        FROM "Entity_name"
        LIMIT 1;
        """
        v_str = await conn.fetchval(sample_query)
        
        if not v_str:
            print("No data in Entity_name.")
            return

        # Create the SQL content
        sql_content = f"""
-- Run this query to benchmark performance on Entity_name
-- using the exact same logic as the app (casting to halfvec)

ANALYZE "Entity_name";
SET enable_seqscan = off;

EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    id, 
    payload, 
    ("vector"::halfvec(3072) <=> '{v_str}'::halfvec(3072)) as similarity
FROM "Entity_name"
ORDER BY similarity
LIMIT 20;
"""
        
        with open("db_test_query.sql", "w") as f:
            f.write(sql_content)
            
        print("✅ Generated db_test_query.sql")
        await conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(generate_sql_file())
