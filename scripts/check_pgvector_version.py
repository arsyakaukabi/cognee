import os
import asyncio
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "10.213.224.113")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bribrain_knowledge_base")
DB_USER = os.getenv("DB_USERNAME", "bribrain_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Bribrainaj4!")

async def check_version():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        print(f"pgvector version: {version}")
        
        # Check max dimensions if possible (indirectly via failure or config)
        # Note: pgvector doesn't expose max dims in a simple config view usually, but version tells us capabilities.
        
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_version())
