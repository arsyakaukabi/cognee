#!/usr/bin/env python3
"""
Test if database works without WAL file.
Creates a copy without WAL and tests opening it.
"""
import os
import shutil
import tempfile
import kuzu
from kuzu.database import Database

def test_without_wal():
    original_db = "/home/usr_00345779_hq_bri_co_id/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu"
    original_wal = "/home/usr_00345779_hq_bri_co_id/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu.wal"
    
    # Create temp copy
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_db")
        
        print("=" * 60)
        print("Test 1: Copy DB without WAL")
        print("=" * 60)
        
        # Copy only main database file (no WAL)
        shutil.copy(original_db, test_db)
        
        print(f"Copied to: {test_db}")
        print(f"File size: {os.path.getsize(test_db) / 1024 / 1024:.2f} MB")
        
        try:
            db = Database(
                test_db,
                buffer_pool_size=2048 * 1024 * 1024,
                max_db_size=4096 * 1024 * 1024,
            )
            db.init_database()
            conn = kuzu.Connection(db)
            
            print("✅ Database opens without WAL!")
            
            # Count nodes
            result = conn.execute("MATCH (n:Node) RETURN COUNT(n)")
            if result.has_next():
                count = result.get_next()[0]
                print(f"Total nodes: {count}")
            
            return True
        except Exception as e:
            print(f"❌ Still corrupted: {e}")
            return False


if __name__ == "__main__":
    test_without_wal()
