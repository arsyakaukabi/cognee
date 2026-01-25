#!/usr/bin/env python3
"""
Test if backup database is also corrupted.
"""
import kuzu
from kuzu.database import Database

def test_backup():
    db_path = "/home/usr_00345779_hq_bri_co_id/cognee/cognee/.cognee_system/databases_backup_20260113_073315/cognee_graph_kuzu"
    
    print("Testing backup database...")
    print(f"Path: {db_path}")
    
    try:
        db = Database(
            db_path,
            buffer_pool_size=2048 * 1024 * 1024,
            max_db_size=4096 * 1024 * 1024,
        )
        db.init_database()
        print("✅ Backup database opens successfully!")
        return True
    except Exception as e:
        print(f"❌ Backup also corrupted: {e}")
        return False

if __name__ == "__main__":
    test_backup()
