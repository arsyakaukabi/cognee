import os
import asyncio
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "10.213.224.113")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bribrain_knowledge_base_hnsw")
DB_USER = os.getenv("DB_USERNAME", "bribrain_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Bribrainaj4!")

# Vector tables identified
VECTOR_TABLES = [
    {"table": "Entity_name", "column": "vector"},
    {"table": "TextDocument_name", "column": "vector"},
    {"table": "EntityType_name", "column": "vector"},
    {"table": "EdgeType_relationship_name", "column": "vector"},
    {"table": "DocumentChunk_text", "column": "vector"},
    {"table": "TextSummary_text", "column": "vector"},
]

DIMENSIONS = 3072

async def create_indices():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"Connected to {DB_NAME} at {DB_HOST}:{DB_PORT}")

        # Ensure halfvec exists (implicit in 0.7.0+)
        
        for item in VECTOR_TABLES:
            table = item["table"]
            col = item["column"]
            index_name = f"idx_{table}_{col}_halfvec_hnsw"
            
            print(f"Creating index {index_name} on {table}({col})...")
            
            # Use expression index with casting to halfvec() to support >2000 dims
            # Note: We must use a valid operator class for halfvec.
            # halfvec_cosine_ops is likely what we want.
            
            query = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" 
            ON "{table}" 
            USING hnsw (("{col}"::halfvec({DIMENSIONS})) halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
            
            try:
                await conn.execute(query)
                print(f"✅ Index {index_name} created successfully.")
            except Exception as e:
                print(f"❌ Failed to create index on {table}: {e}")

        await conn.close()
    
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    asyncio.run(create_indices())
