"""
02_reembed.py - Batch re-embedding script with resume capability.

Re-embeds vector data in batches. Expected dimension is read from
EMBEDDING_DIMENSIONS in .env.

Usage (from repo root):
    uv run python reembed_workflow/02_reembed.py [--resume] [--table TABLE_NAME]

Options:
    --resume       Resume from last progress (progress.json in this folder)
    --table NAME   Process only one table
    --batch-size N Batch size (default: 50)
    --dry-run      Simulate without updating the database
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Resolve project root and load .env from there
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import text
from cognee.infrastructure.databases.relational import get_relational_engine, get_relational_config
from cognee.infrastructure.databases.vector import get_vector_engine

# Pool with prepared-statement cache disabled to avoid stale plan after ALTER (Postgres only)
_reembed_asyncpg_pool: Optional[Any] = None


# Vector tables: Cognee stores as IndexSchema with field 'text' in payload
VECTOR_TABLES = {
    "DocumentChunk_text": "text",
    "Entity_name": "text",
    "EntityType_name": "text",
    "EdgeType_relationship_name": "text",
    "TextSummary_text": "text",
    "TextDocument_name": "text",
}

PROGRESS_FILE = SCRIPT_DIR / "progress.json"
LOG_FILE = SCRIPT_DIR / f"reembed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Schema for vector tables (must match where alter_dimension_4096.sql was run; default public)
VECTOR_TABLE_SCHEMA = os.getenv("REEMBED_VECTOR_SCHEMA", "public")


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("reembed")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_progress() -> Dict:
    """Load progress from JSON file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "tables": {},
        "started_at": None,
        "last_updated": None,
        "completed": False,
    }


def save_progress(progress: Dict) -> None:
    """Save progress to JSON file."""
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _qualified_table(table_name: str) -> str:
    """Return schema-qualified table name so we hit the same table as the ALTER script."""
    return f'{VECTOR_TABLE_SCHEMA}."{table_name}"'


async def get_table_count(table_name: str) -> int:
    """Get total record count for a table."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = text(f'SELECT COUNT(*) FROM {_qualified_table(table_name)}')
        result = await session.execute(query)
        return result.scalar()


async def get_vector_column_type(table_name: str) -> Optional[str]:
    """Return the vector column type as seen by this process's DB connection (diagnostic)."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        # Same connection we use for UPDATE; nspname = schema, relname = table
        query = text("""
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS vector_type
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname = :table_name AND a.attname = 'vector'
        """)
        result = await session.execute(
            query, {"schema": VECTOR_TABLE_SCHEMA, "table_name": table_name}
        )
        row = result.first()
        return row.vector_type if row else None


async def _get_asyncpg_pool_no_cache():
    """Return an asyncpg pool with statement_cache_size=0 to avoid stale plans after ALTER."""
    global _reembed_asyncpg_pool
    if _reembed_asyncpg_pool is not None:
        return _reembed_asyncpg_pool
    try:
        import asyncpg
    except ImportError:
        return None
    r = get_relational_config()
    if (r.db_provider or "").lower() != "postgres":
        return None
    _reembed_asyncpg_pool = await asyncpg.create_pool(
        host=r.db_host,
        port=int(r.db_port or 5432),
        user=r.db_username,
        password=r.db_password,
        database=r.db_name,
        min_size=1,
        max_size=5,
        statement_cache_size=0,
    )
    return _reembed_asyncpg_pool


async def get_vector_indexes_asyncpg(table_name: str) -> List[Dict[str, str]]:
    """On the same asyncpg path we use for UPDATE: list indexes on the vector column. For root-cause diagnosis."""
    pool = await _get_asyncpg_pool_no_cache()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT i.relname AS index_name, pg_get_indexdef(ix.indexrelid) AS index_def
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class c ON c.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(ix.indkey)
                AND a.attname = 'vector' AND NOT a.attisdropped
            WHERE n.nspname = $1 AND c.relname = $2
            """,
            VECTOR_TABLE_SCHEMA,
            table_name,
        )
        return [{"index_name": r["index_name"], "index_def": r["index_def"]} for r in rows]


async def get_relation_kind(table_name: str) -> Optional[str]:
    """Return 'table' or 'view' for the relation (diagnostic: view may redirect UPDATE)."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = text("""
            SELECT c.relkind
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname = :table_name
        """)
        result = await session.execute(
            query, {"schema": VECTOR_TABLE_SCHEMA, "table_name": table_name}
        )
        row = result.first()
        if not row:
            return None
        # r=ordinary table, v=view, m=materialized view
        kind = {"r": "table", "v": "view", "m": "materialized view"}.get(row.relkind, row.relkind)
        return kind


async def fetch_batch(
    table_name: str,
    source_field: str,
    offset: int,
    batch_size: int,
    last_id: Optional[str] = None,
) -> List[Dict]:
    """Fetch a batch of records for re-embedding."""
    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        if last_id:
            query = text(f"""
                SELECT CAST(id AS text) as id, payload->>'{source_field}' as source_text
                FROM {_qualified_table(table_name)}
                WHERE id > CAST(:last_id AS uuid)
                ORDER BY id
                LIMIT :batch_size
            """)
            result = await session.execute(query, {"last_id": last_id, "batch_size": batch_size})
        else:
            query = text(f"""
                SELECT CAST(id AS text) as id, payload->>'{source_field}' as source_text
                FROM {_qualified_table(table_name)}
                ORDER BY id
                LIMIT :batch_size OFFSET :offset
            """)
            result = await session.execute(query, {"batch_size": batch_size, "offset": offset})

        return [{"id": row.id, "text": row.source_text} for row in result.all()]


async def update_vectors_batch(
    table_name: str,
    id_vector_pairs: List[tuple],
    dry_run: bool = False,
    vector_dimension: Optional[int] = None,
) -> int:
    """Update vectors for a batch of records. Uses asyncpg with statement_cache_size=0 for Postgres to avoid stale plan after ALTER."""
    if dry_run or not id_vector_pairs:
        return len(id_vector_pairs)

    dim = vector_dimension or int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
    qualified = _qualified_table(table_name)

    pool = await _get_asyncpg_pool_no_cache()
    if pool is not None:
        updated = 0
        async with pool.acquire() as conn:
            # List ALL relations with this name (any schema) to catch schema mismatch
            all_rows = await conn.fetch(
                """
                SELECT n.nspname AS schema_name, c.relname AS table_name,
                       pg_catalog.format_type(a.atttypid, a.atttypmod) AS vector_type
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = $1 AND a.attname = 'vector' AND a.attnum > 0
                """,
                table_name,
            )
            # Target relation we will UPDATE
            row = await conn.fetchrow(
                """
                SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS t
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = 'vector'
                """,
                VECTOR_TABLE_SCHEMA,
                table_name,
            )
            conn_dim = row["t"] if row else "unknown"
            if str(dim) not in str(conn_dim):
                msg = (
                    f"On UPDATE connection, column type is {conn_dim} but EMBEDDING_DIMENSIONS={dim}. "
                    "ALTER may have run on a different host (e.g. primary vs replica)."
                )
                if all_rows:
                    msg += f" All relations named {table_name}: " + ", ".join(
                        f"{r['schema_name']}.{r['table_name']}={r['vector_type']}" for r in all_rows
                    )
                raise RuntimeError(msg)
            # Phase 1: probe UPDATE to confirm column accepts dim (overwrites first row; we fix it in loop)
            probe_id = id_vector_pairs[0][0]
            probe_vector = [0.0] * dim
            probe_str = "[" + ",".join(map(str, probe_vector)) + "]"
            try:
                await conn.execute(
                    f"UPDATE {qualified} SET vector = '{probe_str}'::vector({dim}) WHERE id = $1::uuid",
                    probe_id,
                )
            except Exception as probe_err:
                err_str = str(probe_err)
                if "3072" in err_str or "dimensions" in err_str:
                    raise RuntimeError(
                        f"Column is still vector(3072); server said: {err_str}. "
                        "Run alter_dimension_4096.sql on this DB (same host/db as .env), then re-run reembed."
                    ) from probe_err
                raise
            for record_id, new_vector in id_vector_pairs:
                vector_str = "[" + ",".join(map(str, new_vector)) + "]"
                # Inline vector literal so server casts directly to vector(dim); param binding can use wrong type
                await conn.execute(
                    f"UPDATE {qualified} SET vector = '{vector_str}'::vector({dim}) WHERE id = $1::uuid",
                    record_id,
                )
                updated += 1
        return updated

    engine = get_relational_engine()
    updated = 0
    async with engine.get_async_session() as session:
        for record_id, new_vector in id_vector_pairs:
            vector_str = "[" + ",".join(map(str, new_vector)) + "]"
            query = text(f"""
                UPDATE {qualified}
                SET vector = CAST(:vector AS vector({dim}))
                WHERE id = CAST(:id AS uuid)
            """)
            await session.execute(query, {"id": record_id, "vector": vector_str})
            updated += 1
        await session.commit()
    return updated


async def reembed_table(
    table_name: str,
    source_field: str,
    logger: logging.Logger,
    progress: Dict,
    batch_size: int = 50,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Re-embed all records in a table with resume capability."""
    result = {
        "table": table_name,
        "total": 0,
        "processed": 0,
        "failed": 0,
        "failed_ids": [],
        "status": "pending",
    }

    if table_name not in progress["tables"]:
        progress["tables"][table_name] = {
            "last_processed_id": None,
            "processed_count": 0,
            "started_at": datetime.now().isoformat(),
            "completed": False,
        }

    table_progress = progress["tables"][table_name]

    if table_progress.get("completed", False):
        logger.info(f"⏭️  {table_name}: Already completed, skipping.")
        result["status"] = "skipped"
        return result

    logger.info(f"\n{'='*60}")
    logger.info(f"📦 Processing: {table_name}")
    logger.info(f"   Source field: payload->>'{source_field}'")
    logger.info(f"{'='*60}")

    # Diagnostic: what vector dimension does this connection see? (must match EMBEDDING_DIMENSIONS)
    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
    try:
        kind = await get_relation_kind(table_name)
        logger.info(f"   Relation kind: {kind}")
        col_type = await get_vector_column_type(table_name)
        logger.info(f"   DB column type (this connection): {col_type}")
        if kind == "view":
            logger.warning(
                "   ⚠️  This is a VIEW; UPDATE may go to underlying table with different dimension. "
                "ALTER and re-embed the base table, or use REEMBED_VECTOR_SCHEMA for the schema of the base table."
            )
        if col_type and str(expected_dim) not in col_type:
            logger.warning(
                f"   ⚠️  Column is {col_type}; EMBEDDING_DIMENSIONS={expected_dim}. "
                "Run alter_dimension_4096.sql or set REEMBED_VECTOR_SCHEMA."
            )
    except Exception as e:
        logger.warning(f"   Could not read column type: {e}")

    # Phase 1 evidence: indexes on vector column (same connection path as UPDATE)
    try:
        vector_indexes = await get_vector_indexes_asyncpg(table_name)
        if vector_indexes:
            logger.info(f"   Indexes on vector column: {len(vector_indexes)}")
            for idx in vector_indexes:
                logger.info(f"      - {idx['index_name']}: {idx['index_def'][:80]}...")
        else:
            logger.info("   Indexes on vector column: (none)")
    except Exception as e:
        logger.warning(f"   Could not list vector indexes: {e}")

    total_count = await get_table_count(table_name)
    result["total"] = total_count
    logger.info(f"   Total records: {total_count}")

    if total_count == 0:
        logger.info("   No records to process.")
        table_progress["completed"] = True
        save_progress(progress)
        result["status"] = "completed"
        return result

    vector_engine = get_vector_engine()
    last_id = table_progress.get("last_processed_id")
    processed_count = table_progress.get("processed_count", 0)

    if last_id:
        logger.info(f"   ⏩ Resuming from ID: {last_id}")
        logger.info(f"   Previously processed: {processed_count}")

    offset = 0
    batch_num = 0

    while True:
        try:
            records = await fetch_batch(table_name, source_field, offset, batch_size, last_id)

            if not records:
                logger.info("   ✅ No more records to process.")
                break

            batch_num += 1

            valid_records = [r for r in records if r["text"] and r["text"].strip()]
            skipped = len(records) - len(valid_records)

            if skipped > 0:
                logger.debug(f"   Batch {batch_num}: Skipped {skipped} empty records")

            if not valid_records:
                if records:
                    last_id = records[-1]["id"]
                    table_progress["last_processed_id"] = last_id
                    save_progress(progress)
                offset += batch_size
                continue

            texts = [r["text"] for r in valid_records]
            logger.debug(f"   Batch {batch_num}: Embedding {len(texts)} texts...")

            try:
                new_vectors = await vector_engine.embedding_engine.embed_text(texts)
            except Exception as e:
                logger.error(f"   ❌ Embedding error in batch {batch_num}: {e}")
                result["failed"] += len(valid_records)
                result["failed_ids"].extend([r["id"] for r in valid_records])
                save_progress(progress)
                offset += batch_size
                last_id = None
                await asyncio.sleep(2)
                continue

            id_vector_pairs = list(zip([r["id"] for r in valid_records], new_vectors))

            try:
                updated = await update_vectors_batch(table_name, id_vector_pairs, dry_run)
                processed_count += updated
                result["processed"] += updated

                last_id = valid_records[-1]["id"]
                table_progress["last_processed_id"] = last_id
                table_progress["processed_count"] = processed_count
                save_progress(progress)

                logger.info(f"   Batch {batch_num}: ✓ Updated {updated} records. Total: {processed_count}/{total_count}")

            except Exception as e:
                err_msg = str(e)
                logger.error(f"   ❌ Database update error in batch {batch_num}: {err_msg}")
                result["failed"] += len(valid_records)
                result["failed_ids"].extend([r["id"] for r in valid_records])
                save_progress(progress)
                # Fail fast on dimension mismatch — don't retry every batch
                if "expected" in err_msg and "dimensions" in err_msg:
                    logger.error(
                        "   🛑 Stopping (dimension mismatch). Fix DB column type and re-run."
                    )
                    result["status"] = "error"
                    return result

            await asyncio.sleep(0.3)

            last_id = valid_records[-1]["id"]
            offset = 0

        except Exception as e:
            logger.error(f"   ❌ Unexpected error in batch {batch_num}: {e}")
            logger.exception("   Full traceback:")
            save_progress(progress)
            result["status"] = "error"
            return result

    table_progress["completed"] = True
    table_progress["completed_at"] = datetime.now().isoformat()
    save_progress(progress)

    result["status"] = "completed" if result["failed"] == 0 else "completed_with_errors"
    logger.info(f"\n   📊 Table Summary: {result['processed']} processed, {result['failed']} failed")

    return result


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch Re-embedding Script (reembed_workflow)")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress")
    parser.add_argument("--table", type=str, help="Process specific table only")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without updating")
    args = parser.parse_args()

    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    logger = setup_logging(LOG_FILE)

    logger.info("=" * 70)
    logger.info("COGNEE RE-EMBEDDING WORKFLOW - BATCH RE-EMBEDDING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Vector table schema: {VECTOR_TABLE_SCHEMA} (set REEMBED_VECTOR_SCHEMA if different)")
    try:
        r = get_relational_config()
        logger.info(f"DB connection: host={r.db_host} db={r.db_name} (must match where you ran ALTER)")
    except Exception:
        pass
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("🔸 DRY RUN MODE - No database changes will be made")

    if args.resume:
        progress = load_progress()
        logger.info(f"📂 Resuming from progress file: {PROGRESS_FILE}")
        if progress.get("last_updated"):
            logger.info(f"   Last updated: {progress['last_updated']}")
    else:
        progress = {
            "tables": {},
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
            "completed": False,
        }
        save_progress(progress)
        logger.info("📂 Starting fresh re-embedding")

    logger.info("\n🔧 Verifying embedding configuration...")
    try:
        vector_engine = get_vector_engine()
        test_embed = await vector_engine.embedding_engine.embed_text(["test"])
        actual_dim = len(test_embed[0])
        logger.info(f"   ✓ Embedding engine OK. Dimensions: {actual_dim}")

        if actual_dim != expected_dim:
            logger.warning(f"   ⚠️  Expected {expected_dim} dimensions (EMBEDDING_DIMENSIONS), got {actual_dim}")
            logger.warning("   Please verify EMBEDDING_DIMENSIONS in .env")
    except Exception as e:
        logger.error(f"   ❌ Embedding engine error: {e}")
        logger.error("   Please check your embedding configuration in .env")
        return

    tables_to_process = {args.table: VECTOR_TABLES[args.table]} if args.table else VECTOR_TABLES

    results = []
    for table_name, source_field in tables_to_process.items():
        try:
            result = await reembed_table(
                table_name=table_name,
                source_field=source_field,
                logger=logger,
                progress=progress,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            results.append(result)
            if result.get("status") == "error":
                logger.error("🛑 Stopping entire run (dimension mismatch). Fix DB and re-run.")
                break
        except Exception as e:
            logger.error(f"❌ Fatal error processing {table_name}: {e}")
            logger.exception("Full traceback:")
            results.append({
                "table": table_name,
                "status": "fatal_error",
                "error": str(e),
            })

    logger.info("\n" + "=" * 70)
    logger.info("📊 RE-EMBEDDING SUMMARY")
    logger.info("=" * 70)

    total_processed = 0
    total_failed = 0

    for r in results:
        status_icon = "✅" if r["status"] == "completed" else "⚠️" if "error" in r["status"] else "⏭️"
        logger.info(f"  {status_icon} {r['table']}: {r.get('processed', 0)} processed, {r.get('failed', 0)} failed")
        total_processed += r.get("processed", 0)
        total_failed += r.get("failed", 0)

    logger.info("-" * 70)
    logger.info(f"  TOTAL: {total_processed} processed, {total_failed} failed")

    if total_failed == 0 and all(r["status"] in ["completed", "skipped"] for r in results):
        progress["completed"] = True
        logger.info("\n✅ RE-EMBEDDING COMPLETED SUCCESSFULLY!")
    else:
        logger.info(f"\n⚠️  Re-embedding completed with {total_failed} failures.")
        logger.info("   Run with --resume to retry failed records.")

    save_progress(progress)
    logger.info(f"\n📄 Log saved to: {LOG_FILE}")
    logger.info(f"📄 Progress saved to: {PROGRESS_FILE}")

    global _reembed_asyncpg_pool
    if _reembed_asyncpg_pool is not None:
        await _reembed_asyncpg_pool.close()
        _reembed_asyncpg_pool = None


if __name__ == "__main__":
    asyncio.run(main())
