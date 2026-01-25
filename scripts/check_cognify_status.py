#!/usr/bin/env python3
"""
Script to:
1. Find which dataset contains the duplicate key c0f3a1ff-b9c8-515f-943a-fb7d6814deee
2. Check how many documents in target dataset have been cognified (completed)
"""

import asyncio
from uuid import UUID
from sqlalchemy import select, func
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, Dataset, DatasetData
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.users.methods import get_default_user


async def find_data_by_id(data_id: str):
    """Find a data item by its ID and show which datasets it belongs to."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Find the data item
        result = await session.execute(
            select(Data).filter(Data.id == UUID(data_id))
        )
        data = result.scalar()
        
        if not data:
            print(f"Data with ID {data_id} NOT FOUND in relational database")
            return None
            
        print(f"Data ID: {data.id}")
        print(f"  Name: {data.name}")
        print(f"  Created: {data.created_at}")
        print(f"  Owner ID: {data.owner_id}")
        print(f"  Pipeline Status: {data.pipeline_status}")
        
        # Find which datasets this data belongs to
        dd_result = await session.execute(
            select(DatasetData.dataset_id).filter(DatasetData.data_id == UUID(data_id))
        )
        dataset_ids = [row[0] for row in dd_result.all()]
        
        print(f"  Belongs to datasets: {len(dataset_ids)}")
        for ds_id in dataset_ids:
            ds_result = await session.execute(
                select(Dataset).filter(Dataset.id == ds_id)
            )
            ds = ds_result.scalar()
            if ds:
                print(f"    - {ds.name} ({ds.id})")
        
        return data


async def check_cognify_status(dataset_id: UUID):
    """Check how many documents have been cognified (completed) for a dataset."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Get all data items for this dataset
        result = await session.execute(
            select(Data)
            .join(Data.datasets)
            .filter(Dataset.id == dataset_id)
        )
        data_items = list(result.scalars().all())
        
        completed = 0
        pending = 0
        
        for item in data_items:
            pipeline_status = item.pipeline_status or {}
            cognify_status = pipeline_status.get("cognify_pipeline", {}).get(str(dataset_id))
            
            if cognify_status == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED:
                completed += 1
            else:
                pending += 1
        
        return {
            "total": len(data_items),
            "completed": completed,
            "pending": pending
        }


async def main():
    duplicate_key = "c0f3a1ff-b9c8-515f-943a-fb7d6814deee"
    target_dataset_id = UUID("9107cf1c-e8e8-5869-9c87-64e9922843df")
    other_dataset_id = UUID("c27cf65d-7bb3-5b16-b944-4a5db7519143")
    
    print("=" * 80)
    print("=== Finding Duplicate Key ===")
    print("=" * 80)
    print(f"\nSearching for: {duplicate_key}")
    print()
    
    await find_data_by_id(duplicate_key)
    
    print("\n" + "=" * 80)
    print("=== Cognify Status for TARGET Dataset ===")
    print("=" * 80)
    print(f"\nDataset: 9107cf1c-e8e8-5869-9c87-64e9922843df")
    
    status = await check_cognify_status(target_dataset_id)
    print(f"\n  Total documents: {status['total']}")
    print(f"  ✅ Completed:     {status['completed']}")
    print(f"  ⏳ Pending:       {status['pending']}")
    
    print("\n" + "=" * 80)
    print("=== Cognify Status for OTHER Dataset ===")
    print("=" * 80)
    print(f"\nDataset: c27cf65d-7bb3-5b16-b944-4a5db7519143")
    
    status2 = await check_cognify_status(other_dataset_id)
    print(f"\n  Total documents: {status2['total']}")
    print(f"  ✅ Completed:     {status2['completed']}")
    print(f"  ⏳ Pending:       {status2['pending']}")


if __name__ == "__main__":
    asyncio.run(main())
