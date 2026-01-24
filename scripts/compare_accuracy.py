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

async def compare_results():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"Connected to {DB_NAME}")

        # Use a random vector from the table to ensure we get matches
        table = "DocumentChunk_text"
        rows = await conn.fetch(f'SELECT vector FROM "{table}" LIMIT 1')
        if not rows:
            print("No data found")
            return
            
        # Extract raw vector string/list
        # asyncpg returns it as a string "[...]" usually for custom types if not mapped
        # Or a list if mapped. Let's handle string.
        vec_data = rows[0]['vector']
        
        # We need it as a string literal for the query
        if isinstance(vec_data, str):
            vector_str = vec_data
        else:
            # Reconstruct string if it came as list
            vector_str = f"[{','.join(str(f) for f in vec_data)}]"

        print("Running Exact Search (Seq Scan)...")
        # Force seqscan and cast to vector (32-bit) for ground truth 
        # (Though our index is on col::halfvec, standard op without cast uses 32-bit logic if pgvector defaults apply)
        # Actually, to be strictly fair, we should compare:
        # A. Non-indexed: SELECT ... ORDER BY vector <-> query
        # B. Indexed: SELECT ... ORDER BY vector::halfvec <-> query::halfvec
        
        # 1. GROUND TRUTH (Slow, Exact 32-bit)
        await conn.execute("SET enable_seqscan = on;")
        await conn.execute("SET enable_indexscan = off;")
        await conn.execute("SET enable_bitmapscan = off;")
        
        # Use standard cosine distance operator <=> assuming typical usage
        # We perform exact search without casting to halfvec
        query_exact = f"""
        SELECT id, payload
        FROM "{table}" 
        ORDER BY vector <=> '{vector_str}'
        LIMIT 50;
        """
        rows_exact = await conn.fetch(query_exact)
        ids_exact = [str(r['id']) for r in rows_exact]
        
        # 2. APPROXIMATE (Fast, Halfvec Index)
        print("Running Approximate Search (Index Scan)...")
        await conn.execute("SET enable_seqscan = off;")
        await conn.execute("SET enable_indexscan = on;")
        await conn.execute("SET enable_bitmapscan = on;")
        
        # Use the casting pattern we implemented
        query_approx = f"""
        SELECT id, payload
        FROM "{table}" 
        ORDER BY CAST(vector AS halfvec(3072)) <=> '{vector_str}'::halfvec(3072)
        LIMIT 50;
        """
        
        rows_approx = await conn.fetch(query_approx)
        ids_approx = [str(r['id']) for r in rows_approx]
        
        # COMPARE
        print("\n--- Results ---")
        
        # Calculate Overlap (Intersection / K)
        k = 50
        intersection = set(ids_exact).intersection(set(ids_approx))
        overlap_count = len(intersection)
        overlap_pct = (overlap_count / k) * 100
        
        print(f"Top-{k} Overlap: {overlap_count}/{k} ({overlap_pct}%)")
        
        if overlap_count < k:
            print("\nMissing IDs in approx results (examples):")
            missing = list(set(ids_exact) - set(ids_approx))
            print(missing[:5])
            
        await conn.close()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(compare_results())
