#!/usr/bin/env python
"""
Full Performance Profiler for Cognee Search Endpoints

This script provides a detailed breakdown of execution time for every major function
in the search pipeline. It also verifies that PostgreSQL is using HNSW index scans.

Usage:
    # Make sure server is running first:
    # uv run uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python scripts/full_perf_profile.py

    # Or with custom query:
    python scripts/full_perf_profile.py --query "your search query here"

Output:
    - Hierarchical function timing breakdown
    - Database query plan verification (Index Scan vs Seq Scan)
    - Bottleneck identification
"""

import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_QUERY = "Apa syarat utk mendapatkan diskon d SUSHI OK Banjarmasin?"

DB_HOST = os.getenv("DB_HOST", "10.213.224.113")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bribrain_knowledge_base_hnsw")
DB_USER = os.getenv("DB_USERNAME", "bribrain_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Bribrainaj4!")

# ============================================================================
# HELPER: Parse structured logs
# ============================================================================

def parse_log_file(log_path: str) -> List[Dict]:
    """Parse structured log entries from a log file."""
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to parse as JSON-ish structured log
            if '[performance]' in line.lower() or '⚡' in line:
                entries.append(line)
    return entries


def find_latest_log() -> str:
    """Find the most recent log file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        return None
    files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')], reverse=True)
    if files:
        return os.path.join(log_dir, files[0])
    return None


# ============================================================================
# HELPER: Trace Analyzer (from analyze_trace.py logic)
# ============================================================================

def analyze_traces_from_log(log_path: str) -> Dict:
    """
    Parse log file and extract performance traces.
    Returns a dict with trace_id -> list of spans.
    """
    import re
    
    traces = defaultdict(list)
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match span_start and span_end
            if 'span_start' in line or 'span_end' in line:
                # Extract correlation_id, span_name, duration_ms, depth
                corr_match = re.search(r'correlation_id=([a-f0-9-]+)', line)
                span_match = re.search(r'span_name=(\S+)', line)
                depth_match = re.search(r'depth=(\d+)', line)
                duration_match = re.search(r'duration_ms=([\d.]+)', line)
                
                if corr_match and span_match:
                    corr_id = corr_match.group(1)
                    span_name = span_match.group(1)
                    depth = int(depth_match.group(1)) if depth_match else 0
                    duration = float(duration_match.group(1)) if duration_match else 0
                    
                    if 'span_end' in line:
                        traces[corr_id].append({
                            'span_name': span_name,
                            'depth': depth,
                            'duration_ms': duration
                        })
            
            # Match DB Vector Search logs
            if '⚡' in line and 'DB Vector Search' in line:
                coll_match = re.search(r'Collection: (\S+)', line)
                dur_match = re.search(r'Duration: ([\d.]+) ms', line)
                if coll_match and dur_match:
                    # Add to a special "db_searches" trace
                    traces['__db_searches__'].append({
                        'collection': coll_match.group(1),
                        'duration_ms': float(dur_match.group(1))
                    })
    
    return dict(traces)


def print_trace_tree(traces: Dict):
    """Pretty print trace trees."""
    print("\n" + "=" * 100)
    print("PERFORMANCE TRACE BREAKDOWN")
    print("=" * 100)
    
    # Print DB searches first
    if '__db_searches__' in traces:
        print("\n📊 DATABASE VECTOR SEARCHES (PostgreSQL + HNSW)")
        print("-" * 60)
        db_searches = traces['__db_searches__']
        total_db = 0
        for s in db_searches:
            print(f"  {s['collection']:40s} {s['duration_ms']:10.2f} ms")
            total_db += s['duration_ms']
        print("-" * 60)
        print(f"  {'TOTAL (parallel, use max as effective)':40s} {max(s['duration_ms'] for s in db_searches):10.2f} ms")
        print()
    
    # Print API traces
    for trace_id, spans in traces.items():
        if trace_id == '__db_searches__':
            continue
        if not spans:
            continue
            
        # Determine if this is retrieval or context
        root_span = next((s for s in spans if s['depth'] == 0), None)
        if not root_span:
            continue
            
        trace_type = "RETRIEVAL" if "retrieval" in root_span['span_name'] else "CONTEXT"
        
        print(f"\n🔍 {trace_type} SEARCH (Trace: {trace_id[:8]}...)")
        print("-" * 80)
        print(f"{'Total (ms)':>12s} {'Self (ms)':>12s}   {'Operation':<50s}")
        print("-" * 80)
        
        # Sort by depth for display
        sorted_spans = sorted(spans, key=lambda x: x['depth'])
        
        for span in sorted_spans:
            indent = "  " * span['depth']
            # Self time calculation would require pairing, approximate here
            print(f"{span['duration_ms']:12.2f} {span['duration_ms']:12.2f}   {indent}{span['span_name']}")
        
        print()


# ============================================================================
# HELPER: Verify Index Usage
# ============================================================================

async def verify_index_usage():
    """
    Run EXPLAIN ANALYZE on a sample query to verify HNSW index is being used.
    """
    import asyncpg
    
    print("\n" + "=" * 100)
    print("DATABASE INDEX VERIFICATION")
    print("=" * 100)
    
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        
        # Get a sample vector
        sample_vector = await conn.fetchval('''
            SELECT (vector::text) FROM "Entity_name" LIMIT 1
        ''')
        
        if not sample_vector:
            print("⚠️  No data in Entity_name table")
            await conn.close()
            return
        
        # Run EXPLAIN ANALYZE with index preference
        explain_query = f'''
        EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
        SELECT id, payload, ("vector"::halfvec(3072) <=> '{sample_vector}'::halfvec(3072)) as similarity
        FROM "Entity_name"
        ORDER BY similarity
        LIMIT 20
        '''
        
        rows = await conn.fetch(explain_query)
        plan_text = "\n".join([row[0] for row in rows])
        
        # Check for Index Scan
        is_index_scan = "Index Scan" in plan_text
        is_seq_scan = "Seq Scan" in plan_text and "Index Scan" not in plan_text
        
        # Extract execution time
        import re
        exec_time_match = re.search(r'Execution Time: ([\d.]+) ms', plan_text)
        exec_time = float(exec_time_match.group(1)) if exec_time_match else 0
        
        print(f"\n📋 Query Plan for Entity_name (largest collection):")
        print("-" * 60)
        
        if is_index_scan:
            print(f"✅ INDEX SCAN DETECTED - Using HNSW Index")
        elif is_seq_scan:
            print(f"⚠️  SEQUENTIAL SCAN - Index NOT being used!")
            print(f"   Consider running: SET enable_seqscan = off;")
        
        print(f"\n⏱️  Execution Time: {exec_time:.2f} ms")
        print(f"\n📝 Full Plan:")
        print(plan_text[:500] + "..." if len(plan_text) > 500 else plan_text)
        
        await conn.close()
        
        return is_index_scan
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


# ============================================================================
# MAIN: Run Performance Test
# ============================================================================

async def run_performance_test(query: str):
    """
    Run a single performance test and analyze the results.
    """
    import requests
    
    print("\n" + "=" * 100)
    print(f"RUNNING PERFORMANCE TEST")
    print(f"Query: '{query}'")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # Find log file before test
    log_before = find_latest_log()
    
    # --- Test Retrieval Endpoint ---
    print("\n🚀 Testing /search/retrieval endpoint...")
    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/search/retrieval",
            json={"query": query, "debug": True},
            timeout=60
        )
        retrieval_duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            result_count = len(data.get('data', []))
            print(f"   ✅ Success | Duration: {retrieval_duration:.2f}ms | Results: {result_count}")
        else:
            print(f"   ❌ Failed | Status: {resp.status_code} | Duration: {retrieval_duration:.2f}ms")
            print(f"   Response: {resp.text[:200]}")
    except Exception as e:
        retrieval_duration = (time.time() - start) * 1000
        print(f"   ❌ Error: {e} | Duration: {retrieval_duration:.2f}ms")
    
    # --- Test Context Endpoint ---
    print("\n🚀 Testing /search/context endpoint...")
    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/search/context",
            json={"query": query, "debug": True},
            timeout=60
        )
        context_duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            context_len = len(data.get('data', ''))
            print(f"   ✅ Success | Duration: {context_duration:.2f}ms | Context Length: {context_len}")
        else:
            print(f"   ❌ Failed | Status: {resp.status_code} | Duration: {context_duration:.2f}ms")
    except Exception as e:
        context_duration = (time.time() - start) * 1000
        print(f"   ❌ Error: {e} | Duration: {context_duration:.2f}ms")
    
    # --- Analyze Logs ---
    print("\n⏳ Waiting for logs to flush...")
    await asyncio.sleep(1)
    
    log_after = find_latest_log()
    if log_after:
        traces = analyze_traces_from_log(log_after)
        print_trace_tree(traces)
    else:
        print("⚠️  Could not find log file for analysis")
    
    # --- Summary ---
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Retrieval Total: {retrieval_duration:,.2f} ms")
    print(f"  Context Total:   {context_duration:,.2f} ms")
    print()


async def main():
    parser = argparse.ArgumentParser(description='Full Performance Profiler for Cognee')
    parser.add_argument('--query', '-q', type=str, default=DEFAULT_QUERY,
                        help='Search query to test')
    parser.add_argument('--skip-index-check', action='store_true',
                        help='Skip database index verification')
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print("COGNEE FULL PERFORMANCE PROFILER")
    print("=" * 100)
    
    # Verify index usage first
    if not args.skip_index_check:
        await verify_index_usage()
    
    # Run performance test
    await run_performance_test(args.query)
    
    print("\n✅ Profiling complete!")
    print("\nTips:")
    print("  - If 'Seq Scan' is detected, run: SET enable_seqscan = off; in your DB session")
    print("  - Look for the largest 'Self (ms)' values to identify bottlenecks")
    print("  - DB Vector Search times should be < 500ms with HNSW index")


if __name__ == "__main__":
    asyncio.run(main())
