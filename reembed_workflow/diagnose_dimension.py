#!/usr/bin/env python3
"""
One-off diagnostic: same asyncpg path as 02_reembed, probe UPDATE only (no embeddings).
Run: uv run python reembed_workflow/diagnose_dimension.py
"""
import asyncio
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_TABLE_SCHEMA = os.getenv("REEMBED_VECTOR_SCHEMA", "public")
DIM = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
TABLE = "DocumentChunk_text"
QUALIFIED = f'{VECTOR_TABLE_SCHEMA}."{TABLE}"'


async def main():
    import asyncpg
    from cognee.infrastructure.databases.relational import get_relational_config
    r = get_relational_config()
    if (r.db_provider or "").lower() != "postgres":
        print("Not Postgres, skip.")
        return
    pool = await asyncpg.create_pool(
        host=r.db_host, port=int(r.db_port or 5432),
        user=r.db_username, password=r.db_password, database=r.db_name,
        min_size=1, max_size=1, statement_cache_size=0,
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS t
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = 'vector'
            """,
            VECTOR_TABLE_SCHEMA, TABLE,
        )
        conn_dim = row["t"] if row else "unknown"
        print(f"pg_attribute says: {conn_dim}  (EMBEDDING_DIMENSIONS={DIM})")
        # Get one real id
        id_row = await conn.fetchrow(f'SELECT id FROM {QUALIFIED} LIMIT 1')
        if not id_row:
            print("No rows in table.")
            await pool.close()
            return
        probe_id = str(id_row["id"])
        probe_str = "[" + ",".join(["0.0"] * DIM) + "]"
        print(f"Probe UPDATE: SET vector = <{DIM} dims> WHERE id = {probe_id[:8]}...")
        try:
            await conn.execute(
                f"UPDATE {QUALIFIED} SET vector = '{probe_str}'::vector({DIM}) WHERE id = $1::uuid",
                probe_id,
            )
            print("Probe OK — column accepts", DIM, "dimensions.")
        except Exception as e:
            err = str(e)
            print("Probe FAILED:", err)
            if "3072" in err or "dimensions" in err:
                print("\n>>> ROOT CAUSE: Column is still vector(3072). Run alter_dimension_4096.sql on this DB.")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
