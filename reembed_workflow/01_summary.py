"""
01_summary.py - Re-embedding workflow summary script.

Shows summary of tables impacted by re-embedding and current embedding config.
Expected dimension is read from EMBEDDING_DIMENSIONS in .env.

Usage (from repo root):
    uv run python reembed_workflow/01_summary.py
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List

# Resolve project root and load .env from there
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from sqlalchemy import text
from cognee.infrastructure.databases.relational import get_relational_engine, get_relational_config
from cognee.infrastructure.databases.vector import get_vector_engine


# Tables that store vector embeddings (display uses source_field for description)
VECTOR_TABLES = {
    "DocumentChunk_text": {"source_field": "text", "description": "Document chunks untuk RAG"},
    "Entity_name": {"source_field": "name", "description": "Entity names dari knowledge graph"},
    "EntityType_name": {"source_field": "name", "description": "Entity type definitions"},
    "EdgeType_relationship_name": {"source_field": "relationship_name", "description": "Edge/relationship types"},
    "TextSummary_text": {"source_field": "text", "description": "Document summaries"},
    "TextDocument_name": {"source_field": "name", "description": "Original document metadata"},
}


async def get_table_stats(table_name: str) -> Dict:
    """Get statistics for a single table."""
    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        count_query = text(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        result = await session.execute(count_query)
        count = result.scalar()

        dim_query = text("""
            SELECT pg_catalog.format_type(atttypid, atttypmod) as column_type
            FROM pg_attribute
            JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE attname = 'vector' AND relname = :table_name
        """)
        result = await session.execute(dim_query, {"table_name": table_name})
        row = result.first()
        vector_type = row.column_type if row else "N/A"

        return {
            "count": count,
            "vector_type": vector_type,
        }


async def get_embedding_config() -> Dict:
    """Get current embedding configuration."""
    try:
        vector_engine = get_vector_engine()
        test_embed = await vector_engine.embedding_engine.embed_text(["test"])
        actual_dim = len(test_embed[0])
    except Exception as e:
        actual_dim = f"Error: {e}"

    return {
        "model": os.getenv("EMBEDDING_MODEL", "Not set"),
        "endpoint": os.getenv("EMBEDDING_ENDPOINT", "Not set"),
        "configured_dimensions": os.getenv("EMBEDDING_DIMENSIONS", "Not set"),
        "actual_dimensions": actual_dim,
    }


async def main():
    """Main function to display re-embedding summary."""
    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    print("=" * 70)
    print("COGNEE RE-EMBEDDING WORKFLOW - SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get embedding config
    print("\n📋 CURRENT EMBEDDING CONFIGURATION")
    print("-" * 50)
    config = await get_embedding_config()
    print(f"  Model:                 {config['model']}")
    endpoint = config["endpoint"]
    print(f"  Endpoint:              {endpoint[:60]}..." if len(endpoint) > 60 else f"  Endpoint:              {endpoint}")
    print(f"  Configured Dimensions: {config['configured_dimensions']}")
    print(f"  Actual Dimensions:     {config['actual_dimensions']}")

    # DB connection (so we can verify we're pointing at the right place)
    print("\n🗄️  RELATIONAL DB CONNECTION (from .env)")
    print("-" * 50)
    try:
        rconfig = get_relational_config()
        print(f"  Provider:   {rconfig.db_provider}")
        print(f"  Host:       {rconfig.db_host}")
        print(f"  Port:       {rconfig.db_port}")
        print(f"  DB name:    {rconfig.db_name}")
        print(f"  Username:   {rconfig.db_username}")
    except Exception as e:
        print(f"  (Could not load config: {e})")

    # Get table stats
    print("\n📊 IMPACTED TABLES")
    print("-" * 70)
    print(f"{'Table Name':<35} {'Records':>10} {'Vector Type':>15}")
    print("-" * 70)

    total_records = 0
    table_results = []

    for table_name, info in VECTOR_TABLES.items():
        try:
            stats = await get_table_stats(table_name)
            total_records += stats["count"]
            table_results.append({
                "name": table_name,
                "count": stats["count"],
                "vector_type": stats["vector_type"],
                "source_field": info["source_field"],
                "description": info["description"],
                "status": "OK",
            })
            print(f"{table_name:<35} {stats['count']:>10} {stats['vector_type']:>15}")
        except Exception as e:
            table_results.append({
                "name": table_name,
                "count": 0,
                "vector_type": "N/A",
                "source_field": info["source_field"],
                "description": info["description"],
                "status": f"ERROR: {e}",
            })
            print(f"{table_name:<35} {'ERROR':>10} {'N/A':>15}")
            # Surface root cause: print actual exception (first time in full, then brief)
            err_msg = str(e).strip()
            if len(err_msg) > 120:
                err_msg = err_msg[:117] + "..."
            print(f"     └─ {err_msg}")

    print("-" * 70)
    print(f"{'TOTAL':.<35} {total_records:>10}")

    # Summary
    print("\n📝 RE-EMBEDDING SUMMARY")
    print("-" * 50)
    print(f"  Total tables to migrate:  {len(VECTOR_TABLES)}")
    print(f"  Total records to re-embed: {total_records:,}")
    print(f"  Source field mapping (payload->>'text' used by re-embed script):")
    for table_name, info in VECTOR_TABLES.items():
        print(f"    • {table_name}: payload->>'{info['source_field']}'")

    # Dimension status (target from .env)
    current_dim = None
    for t in table_results:
        for dim in ("1536", "3072", "4096", "1024", "768"):
            if dim in t["vector_type"]:
                current_dim = int(dim)
                break
        if current_dim is not None:
            break

    print("\n⚠️  DIMENSION STATUS")
    print("-" * 50)
    print(f"  Target dimensions (from .env): vector({expected_dim})")
    if current_dim is not None:
        print(f"  Current vector columns:      vector({current_dim})")
        if current_dim == expected_dim:
            print("  Status: Schema already matches. Ready for re-embedding.")
        else:
            print("  Status: ALTER TABLE required before re-embedding (run alter_dimension_4096.sql).")
    else:
        print("  Unable to determine current dimension. Check table list above.")

    print("\n" + "=" * 70)
    print("Next steps: Run alter_dimension_4096.sql if needed, then 02_reembed.py")
    print("=" * 70)

    return table_results


if __name__ == "__main__":
    asyncio.run(main())
