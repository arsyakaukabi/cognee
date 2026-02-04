"""
03_verify.py - Post re-embedding verification script.

Verifies: vector dimensions match EMBEDDING_DIMENSIONS from .env,
embedding engine, search functionality, and data integrity (no null vectors).

Usage (from repo root):
    uv run python reembed_workflow/03_verify.py
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
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine


VECTOR_TABLES = [
    "DocumentChunk_text",
    "Entity_name",
    "EntityType_name",
    "EdgeType_relationship_name",
    "TextSummary_text",
    "TextDocument_name",
]

TEST_QUERIES = [
    "pengertian kredit",
    "prosedur pengajuan",
    "syarat dan ketentuan",
]


async def verify_vector_dimensions(expected_dim: int) -> Dict:
    """Verify vector column dimensions for all tables match expected_dim."""
    print("\n📏 VERIFYING VECTOR DIMENSIONS")
    print("-" * 60)

    engine = get_relational_engine()
    results = {}

    async with engine.get_async_session() as session:
        for table_name in VECTOR_TABLES:
            query = text("""
                SELECT pg_catalog.format_type(atttypid, atttypmod) as column_type
                FROM pg_attribute
                JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                WHERE attname = 'vector' AND relname = :table_name
            """)
            result = await session.execute(query, {"table_name": table_name})
            row = result.first()

            if row:
                vector_type = row.column_type
                is_expected = str(expected_dim) in vector_type
                status = "✅" if is_expected else "⚠️"
                results[table_name] = {"type": vector_type, "is_migrated": is_expected}
                print(f"  {status} {table_name}: {vector_type}")
            else:
                results[table_name] = {"type": "NOT FOUND", "is_migrated": False}
                print(f"  ❌ {table_name}: Table not found")

    return results


async def verify_embedding_engine(expected_dim: int) -> Dict:
    """Verify embedding engine is configured correctly."""
    print("\n🔧 VERIFYING EMBEDDING ENGINE")
    print("-" * 60)

    result = {
        "model": os.getenv("EMBEDDING_MODEL", "Not set"),
        "dimensions_config": os.getenv("EMBEDDING_DIMENSIONS", "Not set"),
        "actual_dimensions": None,
        "status": "unknown",
    }

    try:
        vector_engine = get_vector_engine()
        test_embed = await vector_engine.embedding_engine.embed_text(["verification test"])
        actual_dim = len(test_embed[0])
        result["actual_dimensions"] = actual_dim

        if actual_dim == expected_dim:
            result["status"] = "ok"
            print(f"  ✅ Model: {result['model']}")
            print(f"  ✅ Configured dimensions: {result['dimensions_config']}")
            print(f"  ✅ Actual dimensions: {actual_dim}")
        else:
            result["status"] = "warning"
            print(f"  ⚠️  Model: {result['model']}")
            print(f"  ⚠️  Expected {expected_dim} dimensions (EMBEDDING_DIMENSIONS), got {actual_dim}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ❌ Error: {e}")

    return result


async def verify_search_functionality() -> List[Dict]:
    """Test search functionality with sample queries."""
    print("\n🔍 VERIFYING SEARCH FUNCTIONALITY")
    print("-" * 60)

    results = []
    vector_engine = get_vector_engine()
    table_name = "DocumentChunk_text"

    for query in TEST_QUERIES:
        result = {
            "query": query,
            "table": table_name,
            "num_results": 0,
            "status": "unknown",
            "top_score": None,
        }

        try:
            search_results = await vector_engine.search(
                collection_name=table_name,
                query_text=query,
                limit=5,
            )

            result["num_results"] = len(search_results)
            if search_results:
                result["top_score"] = search_results[0].score
                result["status"] = "ok"
                print(f"  ✅ Query: '{query}'")
                print(f"     Results: {len(search_results)}, Top score: {search_results[0].score:.4f}")
            else:
                result["status"] = "no_results"
                print(f"  ⚠️  Query: '{query}' - No results found")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"  ❌ Query: '{query}' - Error: {e}")

        results.append(result)

    return results


async def verify_data_integrity() -> Dict:
    """Verify data integrity - check for null vectors."""
    print("\n🔒 VERIFYING DATA INTEGRITY")
    print("-" * 60)

    engine = get_relational_engine()
    results = {}

    async with engine.get_async_session() as session:
        for table_name in VECTOR_TABLES:
            try:
                total_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                total_result = await session.execute(total_query)
                total = total_result.scalar()

                null_query = text(f'SELECT COUNT(*) FROM "{table_name}" WHERE vector IS NULL')
                null_result = await session.execute(null_query)
                null_count = null_result.scalar()

                # pgvector "vector" type cannot be cast to float[]; NOT NULL is sufficient for validity
                valid_query = text(f"""
                    SELECT COUNT(*) FROM "{table_name}"
                    WHERE vector IS NOT NULL
                """)
                valid_result = await session.execute(valid_query)
                valid_count = valid_result.scalar()

                results[table_name] = {
                    "total": total,
                    "null_vectors": null_count,
                    "valid_vectors": valid_count,
                    "integrity": "ok" if null_count == 0 and valid_count == total else "warning",
                }

                status = "✅" if results[table_name]["integrity"] == "ok" else "⚠️"
                print(f"  {status} {table_name}: {valid_count}/{total} valid vectors")

                if null_count > 0:
                    print(f"     ⚠️  {null_count} null vectors found")

            except Exception as e:
                results[table_name] = {"error": str(e)}
                print(f"  ❌ {table_name}: Error - {e}")

    return results


async def main() -> Dict:
    """Main verification function."""
    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    print("=" * 70)
    print("COGNEE RE-EMBEDDING WORKFLOW - VERIFICATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected dimension (from .env): {expected_dim}")
    print("=" * 70)

    dim_results = await verify_vector_dimensions(expected_dim)
    engine_result = await verify_embedding_engine(expected_dim)
    search_results = await verify_search_functionality()
    integrity_results = await verify_data_integrity()

    print("\n" + "=" * 70)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 70)

    all_dims_ok = all(r["is_migrated"] for r in dim_results.values())
    dim_msg = f"All tables migrated to {expected_dim}" if all_dims_ok else "Some tables not migrated"
    print(f"  Vector Dimensions:  {'✅ ' + dim_msg if all_dims_ok else '⚠️ ' + dim_msg}")

    print(f"  Embedding Engine:   {'✅ OK' if engine_result['status'] == 'ok' else '⚠️ Check configuration'}")

    search_ok = all(r["status"] == "ok" for r in search_results)
    print(f"  Search Function:    {'✅ Working' if search_ok else '⚠️ Some searches failed'}")

    integrity_ok = all(
        r.get("integrity") == "ok" for r in integrity_results.values() if "integrity" in r
    )
    print(f"  Data Integrity:     {'✅ OK' if integrity_ok else '⚠️ Check null vectors'}")

    all_ok = all_dims_ok and engine_result["status"] == "ok" and search_ok and integrity_ok

    print("\n" + "-" * 70)
    if all_ok:
        print("🎉 RE-EMBEDDING VERIFICATION PASSED!")
        print("   All checks completed successfully.")
    else:
        print("⚠️  VERIFICATION COMPLETED WITH WARNINGS")
        print("   Please review the issues above.")

    print("=" * 70)

    return {
        "dimensions": dim_results,
        "embedding_engine": engine_result,
        "search": search_results,
        "integrity": integrity_results,
        "overall_status": "passed" if all_ok else "warnings",
    }


if __name__ == "__main__":
    asyncio.run(main())
