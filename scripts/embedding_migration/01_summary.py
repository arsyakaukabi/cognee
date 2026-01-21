"""
01_summary.py - Embedding Migration Summary Script

Menampilkan summary tabel yang terdampak migrasi embedding beserta jumlah record.
Model: text-embedding-ada-002 (1536) → text-embedding-3-large (3072)

Usage:
    python scripts/embedding_migration/01_summary.py
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine


# Konfigurasi tabel vector yang perlu di-migrasi
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
        # Count records
        count_query = text(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        result = await session.execute(count_query)
        count = result.scalar()
        
        # Get vector column info
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
            "vector_type": vector_type
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
    """Main function to display migration summary."""
    print("=" * 70)
    print("COGNEE EMBEDDING MIGRATION - SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Get embedding config
    print("\n📋 CURRENT EMBEDDING CONFIGURATION")
    print("-" * 50)
    config = await get_embedding_config()
    print(f"  Model:                 {config['model']}")
    print(f"  Endpoint:              {config['endpoint'][:60]}..." if len(config['endpoint']) > 60 else f"  Endpoint:              {config['endpoint']}")
    print(f"  Configured Dimensions: {config['configured_dimensions']}")
    print(f"  Actual Dimensions:     {config['actual_dimensions']}")
    
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
                "status": "OK"
            })
            print(f"{table_name:<35} {stats['count']:>10} {stats['vector_type']:>15}")
        except Exception as e:
            table_results.append({
                "name": table_name,
                "count": 0,
                "vector_type": "N/A",
                "source_field": info["source_field"],
                "description": info["description"],
                "status": f"ERROR: {e}"
            })
            print(f"{table_name:<35} {'ERROR':>10} {'N/A':>15}")
    
    print("-" * 70)
    print(f"{'TOTAL':.<35} {total_records:>10}")
    
    # Summary
    print("\n📝 MIGRATION SUMMARY")
    print("-" * 50)
    print(f"  Total tables to migrate:  {len(VECTOR_TABLES)}")
    print(f"  Total records to re-embed: {total_records:,}")
    print(f"  Source field mapping:")
    for table_name, info in VECTOR_TABLES.items():
        print(f"    • {table_name}: payload->>'{info['source_field']}'")
    
    # Dimension change warning
    current_dim = None
    for t in table_results:
        if "1536" in t["vector_type"]:
            current_dim = 1536
            break
        elif "3072" in t["vector_type"]:
            current_dim = 3072
            break
    
    print("\n⚠️  DIMENSION STATUS")
    print("-" * 50)
    if current_dim == 1536:
        print("  Current vector columns: vector(1536)")
        print("  Target vector columns:  vector(3072)")
        print("  Status: ALTER TABLE required before re-embedding")
    elif current_dim == 3072:
        print("  Current vector columns: vector(3072)")
        print("  Status: Schema already migrated. Ready for re-embedding.")
    else:
        print("  Unable to determine current dimension. Please check manually.")
    
    print("\n" + "=" * 70)
    print("Next steps: Run 02_reembed.py to start batch re-embedding")
    print("=" * 70)
    
    return table_results


if __name__ == "__main__":
    asyncio.run(main())
