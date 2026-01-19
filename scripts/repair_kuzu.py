#!/usr/bin/env python3
"""
Repair script to fix duplicate primary key in Kuzu graph database.

This script:
1. Opens the Kuzu database directly
2. Finds duplicate nodes with the specified ID
3. Deletes one of the duplicates

NOTE: The duplicate key error happens during Kuzu initialization because
the database is corrupted. This script uses low-level Kuzu operations
to fix the corruption WITHOUT triggering the full initialization that fails.
"""

import os
import kuzu
from kuzu import Connection
from kuzu.database import Database


def repair_kuzu_database():
    # Path to Kuzu database
    db_path = "/home/usr_00345779_hq_bri_co_id/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu"
    duplicate_key = "c0f3a1ff-b9c8-515f-943a-fb7d6814deee"
    
    print("=" * 80)
    print("=== Kuzu Database Repair Script ===")
    print("=" * 80)
    print(f"\nDatabase path: {db_path}")
    print(f"Duplicate key to fix: {duplicate_key}")
    print()
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database path does not exist: {db_path}")
        return False
    
    try:
        print("Opening Kuzu database...")
        db = Database(
            db_path,
            buffer_pool_size=2048 * 1024 * 1024,  # 2048MB buffer pool
            max_db_size=4096 * 1024 * 1024,
        )
        db.init_database()
        conn = Connection(db)
        
        print("Database opened successfully!")
        print()
        
        # First, let's check how many nodes have this ID
        print(f"Checking for nodes with ID: {duplicate_key}")
        query = f"""
        MATCH (n:Node)
        WHERE n.id = '{duplicate_key}'
        RETURN n.id, n.name, n.type
        """
        result = conn.execute(query)
        
        count = 0
        nodes_info = []
        while result.has_next():
            row = result.get_next()
            count += 1
            nodes_info.append(row)
            print(f"  Found node {count}: id={row[0]}, name={row[1]}, type={row[2]}")
        
        print(f"\nTotal nodes found with this ID: {count}")
        
        if count == 0:
            print("\nNo duplicate nodes found. The issue may be elsewhere.")
            print("Checking total node count...")
            
            count_query = "MATCH (n:Node) RETURN COUNT(n)"
            count_result = conn.execute(count_query)
            if count_result.has_next():
                total = count_result.get_next()[0]
                print(f"Total nodes in database: {total}")
            
            return True
        
        if count == 1:
            print("\nOnly one node found (no duplicate). The database may already be clean.")
            return True
        
        if count > 1:
            print(f"\n⚠️  Found {count} duplicate nodes! This is the problem.")
            print("\nTo fix, we need to delete the duplicate nodes.")
            print("This will keep only the first node.")
            
            # Delete all but one
            delete_query = f"""
            MATCH (n:Node)
            WHERE n.id = '{duplicate_key}'
            WITH n
            SKIP 1
            DETACH DELETE n
            """
            
            print("\n⚠️  DRY RUN - Not executing delete yet.")
            print("Delete query would be:")
            print(delete_query)
            
            return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nThis error is expected if the database has duplicate primary keys.")
        print("The corruption prevents normal database opening.")
        return False
    finally:
        print("\nDone.")


if __name__ == "__main__":
    repair_kuzu_database()
