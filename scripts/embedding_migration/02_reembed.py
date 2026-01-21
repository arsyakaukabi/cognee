"""
02_reembed.py - Batch Re-embedding Script with Resume Capability

Melakukan re-embedding vector data secara batch dengan fitur:
- Resume dari record terakhir jika gagal
- Logging ke file .log
- Progress tracking

Model: text-embedding-ada-002 (1536) → text-embedding-3-large (3072)

Usage:
    python scripts/embedding_migration/02_reembed.py [--resume] [--table TABLE_NAME]

Options:
    --resume       Resume dari progress terakhir (baca dari progress.json)
    --table NAME   Hanya proses tabel tertentu
    --batch-size N Ukuran batch (default: 50)
    --dry-run      Simulasi tanpa update database
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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine


# Konfigurasi tabel vector
VECTOR_TABLES = {
    "DocumentChunk_text": "text",
    "Entity_name": "name",
    "EntityType_name": "name",
    "EdgeType_relationship_name": "relationship_name",
    "TextSummary_text": "text",
    "TextDocument_name": "name",
}

# File paths
SCRIPT_DIR = Path(__file__).parent
PROGRESS_FILE = SCRIPT_DIR / "progress.json"
LOG_FILE = SCRIPT_DIR / f"reembed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("reembed")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_progress() -> Dict:
    """Load progress from JSON file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "tables": {},
        "started_at": None,
        "last_updated": None,
        "completed": False
    }


def save_progress(progress: Dict):
    """Save progress to JSON file."""
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


async def get_table_count(table_name: str) -> int:
    """Get total record count for a table."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = text(f'SELECT COUNT(*) FROM "{table_name}"')
        result = await session.execute(query)
        return result.scalar()


async def fetch_batch(
    table_name: str, 
    source_field: str, 
    offset: int, 
    batch_size: int,
    last_id: Optional[str] = None
) -> List[Dict]:
    """Fetch a batch of records for re-embedding."""
    engine = get_relational_engine()
    
    async with engine.get_async_session() as session:
        if last_id:
            # Resume from last processed ID
            query = text(f"""
                SELECT id::text, payload->>'{source_field}' as source_text
                FROM "{table_name}"
                WHERE id > :last_id::uuid
                ORDER BY id
                LIMIT :batch_size
            """)
            result = await session.execute(query, {"last_id": last_id, "batch_size": batch_size})
        else:
            query = text(f"""
                SELECT id::text, payload->>'{source_field}' as source_text
                FROM "{table_name}"
                ORDER BY id
                LIMIT :batch_size OFFSET :offset
            """)
            result = await session.execute(query, {"batch_size": batch_size, "offset": offset})
        
        return [{"id": row.id, "text": row.source_text} for row in result.all()]


async def update_vectors_batch(
    table_name: str, 
    id_vector_pairs: List[tuple],
    dry_run: bool = False
) -> int:
    """Update vectors for a batch of records."""
    if dry_run or not id_vector_pairs:
        return len(id_vector_pairs)
    
    engine = get_relational_engine()
    updated = 0
    
    async with engine.get_async_session() as session:
        for record_id, new_vector in id_vector_pairs:
            vector_str = "[" + ",".join(map(str, new_vector)) + "]"
            query = text(f"""
                UPDATE "{table_name}"
                SET vector = :vector::vector
                WHERE id = :id::uuid
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
    dry_run: bool = False
) -> Dict[str, Any]:
    """Re-embed all records in a table with resume capability."""
    
    result = {
        "table": table_name,
        "total": 0,
        "processed": 0,
        "failed": 0,
        "failed_ids": [],
        "status": "pending"
    }
    
    # Initialize table progress if not exists
    if table_name not in progress["tables"]:
        progress["tables"][table_name] = {
            "last_processed_id": None,
            "processed_count": 0,
            "started_at": datetime.now().isoformat(),
            "completed": False
        }
    
    table_progress = progress["tables"][table_name]
    
    # Skip if already completed
    if table_progress.get("completed", False):
        logger.info(f"⏭️  {table_name}: Already completed, skipping.")
        result["status"] = "skipped"
        return result
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📦 Processing: {table_name}")
    logger.info(f"   Source field: payload->>'{source_field}'")
    logger.info(f"{'='*60}")
    
    # Get total count
    total_count = await get_table_count(table_name)
    result["total"] = total_count
    logger.info(f"   Total records: {total_count}")
    
    if total_count == 0:
        logger.info(f"   No records to process.")
        table_progress["completed"] = True
        save_progress(progress)
        result["status"] = "completed"
        return result
    
    # Get vector engine
    vector_engine = get_vector_engine()
    
    # Resume from last processed ID if available
    last_id = table_progress.get("last_processed_id")
    processed_count = table_progress.get("processed_count", 0)
    
    if last_id:
        logger.info(f"   ⏩ Resuming from ID: {last_id}")
        logger.info(f"   Previously processed: {processed_count}")
    
    offset = 0
    batch_num = 0
    
    while True:
        try:
            # Fetch batch
            records = await fetch_batch(table_name, source_field, offset, batch_size, last_id)
            
            if not records:
                logger.info(f"   ✅ No more records to process.")
                break
            
            batch_num += 1
            
            # Filter out empty texts
            valid_records = [r for r in records if r["text"] and r["text"].strip()]
            skipped = len(records) - len(valid_records)
            
            if skipped > 0:
                logger.debug(f"   Batch {batch_num}: Skipped {skipped} empty records")
            
            if not valid_records:
                # Update last_id even if all empty
                if records:
                    last_id = records[-1]["id"]
                    table_progress["last_processed_id"] = last_id
                    save_progress(progress)
                offset += batch_size
                continue
            
            # Generate embeddings
            texts = [r["text"] for r in valid_records]
            logger.debug(f"   Batch {batch_num}: Embedding {len(texts)} texts...")
            
            try:
                new_vectors = await vector_engine.embedding_engine.embed_text(texts)
            except Exception as e:
                logger.error(f"   ❌ Embedding error in batch {batch_num}: {e}")
                result["failed"] += len(valid_records)
                result["failed_ids"].extend([r["id"] for r in valid_records])
                # Save progress and continue
                save_progress(progress)
                offset += batch_size
                last_id = None  # Reset to use offset
                await asyncio.sleep(2)  # Wait before retry
                continue
            
            # Update database
            id_vector_pairs = list(zip([r["id"] for r in valid_records], new_vectors))
            
            try:
                updated = await update_vectors_batch(table_name, id_vector_pairs, dry_run)
                processed_count += updated
                result["processed"] += updated
                
                # Update progress
                last_id = valid_records[-1]["id"]
                table_progress["last_processed_id"] = last_id
                table_progress["processed_count"] = processed_count
                save_progress(progress)
                
                logger.info(f"   Batch {batch_num}: ✓ Updated {updated} records. Total: {processed_count}/{total_count}")
                
            except Exception as e:
                logger.error(f"   ❌ Database update error in batch {batch_num}: {e}")
                result["failed"] += len(valid_records)
                result["failed_ids"].extend([r["id"] for r in valid_records])
                save_progress(progress)
            
            # Rate limiting
            await asyncio.sleep(0.3)
            
            # Clear last_id for next iteration (use offset-based fetch)
            last_id = valid_records[-1]["id"]
            offset = 0  # We're using ID-based pagination now
            
        except Exception as e:
            logger.error(f"   ❌ Unexpected error in batch {batch_num}: {e}")
            logger.exception("   Full traceback:")
            save_progress(progress)
            result["status"] = "error"
            return result
    
    # Mark as completed
    table_progress["completed"] = True
    table_progress["completed_at"] = datetime.now().isoformat()
    save_progress(progress)
    
    result["status"] = "completed" if result["failed"] == 0 else "completed_with_errors"
    logger.info(f"\n   📊 Table Summary: {result['processed']} processed, {result['failed']} failed")
    
    return result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch Re-embedding Script")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress")
    parser.add_argument("--table", type=str, help="Process specific table only")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without updating")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(LOG_FILE)
    
    logger.info("=" * 70)
    logger.info("COGNEE EMBEDDING MIGRATION - BATCH RE-EMBEDDING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 70)
    
    if args.dry_run:
        logger.info("🔸 DRY RUN MODE - No database changes will be made")
    
    # Load or initialize progress
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
            "completed": False
        }
        save_progress(progress)
        logger.info("📂 Starting fresh migration")
    
    # Verify embedding engine
    logger.info("\n🔧 Verifying embedding configuration...")
    try:
        vector_engine = get_vector_engine()
        test_embed = await vector_engine.embedding_engine.embed_text(["test"])
        actual_dim = len(test_embed[0])
        logger.info(f"   ✓ Embedding engine OK. Dimensions: {actual_dim}")
        
        if actual_dim != 3072:
            logger.warning(f"   ⚠️  Expected 3072 dimensions, got {actual_dim}")
            logger.warning("   Please verify EMBEDDING_DIMENSIONS in .env")
    except Exception as e:
        logger.error(f"   ❌ Embedding engine error: {e}")
        logger.error("   Please check your embedding configuration in .env")
        return
    
    # Process tables
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
                dry_run=args.dry_run
            )
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Fatal error processing {table_name}: {e}")
            logger.exception("Full traceback:")
            results.append({
                "table": table_name,
                "status": "fatal_error",
                "error": str(e)
            })
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 MIGRATION SUMMARY")
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
    
    # Mark overall completion
    if total_failed == 0 and all(r["status"] in ["completed", "skipped"] for r in results):
        progress["completed"] = True
        logger.info("\n✅ MIGRATION COMPLETED SUCCESSFULLY!")
    else:
        logger.info(f"\n⚠️  Migration completed with {total_failed} failures.")
        logger.info(f"   Run with --resume to retry failed records.")
    
    save_progress(progress)
    logger.info(f"\n📄 Log saved to: {LOG_FILE}")
    logger.info(f"📄 Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
