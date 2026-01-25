#!/usr/bin/env python3
"""
Script to check cognify pipeline status.
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun
from sqlalchemy import select, desc


async def check_pipeline_status():
    """Check the status of all pipeline runs"""
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        # Get latest pipeline runs
        query = (
            select(PipelineRun)
            .order_by(desc(PipelineRun.created_at))
            .limit(20)
        )
        
        runs = (await session.execute(query)).scalars().all()
        
        print("=" * 80)
        print("Pipeline Status Report")
        print("=" * 80)
        
        if not runs:
            print("No pipeline runs found.")
            return
        
        # Group by dataset
        datasets = {}
        for run in runs:
            dataset_id = str(run.dataset_id)
            if dataset_id not in datasets:
                datasets[dataset_id] = []
            datasets[dataset_id].append(run)
        
        for dataset_id, dataset_runs in datasets.items():
            latest = dataset_runs[0]
            print(f"\nDataset: {dataset_id[:8]}...")
            print(f"  Pipeline:   {latest.pipeline_name or 'N/A'}")
            print(f"  Status:     {latest.status}")
            print(f"  Run ID:     {latest.pipeline_run_id}")
            print(f"  Created:    {latest.created_at}")
            if len(dataset_runs) > 1:
                print(f"  History:    {len(dataset_runs)} runs")
        
        print("\n" + "=" * 80)
        print("STATUS SUMMARY")
        print("=" * 80)
        
        statuses = {}
        for run in runs:
            status = str(run.status)
            statuses[status] = statuses.get(status, 0) + 1
        
        for status, count in statuses.items():
            print(f"  {status}: {count}")


async def check_documents_processed():
    """Check how many documents have been processed (have chunks)"""
    from cognee.infrastructure.databases.graph import get_graph_engine
    
    print("\n" + "=" * 80)
    print("Document Processing Status")
    print("=" * 80)
    
    try:
        graph_engine = await get_graph_engine()
        
        # Count DocumentChunk nodes
        chunk_query = """
        MATCH (c:DocumentChunk)
        RETURN count(c) as chunk_count
        """
        chunks_result = await graph_engine.query(chunk_query)
        chunk_count = chunks_result[0]["chunk_count"] if chunks_result else 0
        
        # Count unique documents (by chunk.document_id)
        doc_query = """
        MATCH (c:DocumentChunk)
        RETURN count(DISTINCT c.document_id) as doc_count
        """
        docs_result = await graph_engine.query(doc_query)
        doc_count = docs_result[0]["doc_count"] if docs_result else 0
        
        # Count Entity nodes
        entity_query = """
        MATCH (e:Entity)
        RETURN count(e) as entity_count
        """
        entity_result = await graph_engine.query(entity_query)
        entity_count = entity_result[0]["entity_count"] if entity_result else 0
        
        print(f"  Documents processed: {doc_count}")
        print(f"  Total chunks:        {chunk_count}")
        print(f"  Entities extracted:  {entity_count}")
        
        if doc_count > 0:
            print(f"  Avg chunks/doc:      {chunk_count / doc_count:.1f}")
        
    except Exception as e:
        print(f"  Error querying graph: {e}")


async def main():
    await check_pipeline_status()
    await check_documents_processed()


if __name__ == "__main__":
    asyncio.run(main())
