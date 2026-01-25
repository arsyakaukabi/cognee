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

async def list_vector_tables():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"Connected to {DB_NAME} at {DB_HOST}:{DB_PORT}")

        # Query to find tables with columns of type 'vector'
        query = """
        SELECT 
            t.table_name, 
            c.column_name, 
            c.data_type,
            c.udt_name
        FROM 
            information_schema.tables t
        JOIN 
            information_schema.columns c ON t.table_name = c.table_name
        WHERE 
            t.table_schema = 'public' 
            AND (c.data_type = 'USER-DEFINED' AND c.udt_name = 'vector');
        """
        
        rows = await conn.fetch(query)
        
        if not rows:
            print("No vector tables found.")
        else:
            print(f"Found {len(rows)} vector columns:")
            for row in rows:
                print(f"Table: {row['table_name']}, Column: {row['column_name']}, Type: {row['udt_name']}")

        await conn.close()
    
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    asyncio.run(list_vector_tables())
