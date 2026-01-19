#!/usr/bin/env python3
"""
Investigation script to count documents for specific dataset IDs and identify
potential duplicate key issues in the bri_knowledge_base dataset.

This script will:
1. Find the dataset by name (bri_knowledge_base)
2. Count documents for each dataset using that name
3. Check for potential duplicates across datasets
"""

import asyncio
from uuid import UUID
from sqlalchemy import select, func
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, Dataset, DatasetData
from cognee.modules.users.methods import get_default_user


async def get_all_datasets():
    """Get all datasets."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Dataset))
        return list(result.scalars().all())


async def get_datasets_with_name(name: str):
    """Get all datasets with a specific name."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Dataset).filter(Dataset.name == name)
        )
        return list(result.scalars().all())


async def count_data_for_dataset(dataset_id: UUID):
    """Count the number of data items for a dataset."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(func.count(DatasetData.data_id))
            .filter(DatasetData.dataset_id == dataset_id)
        )
        return result.scalar()


async def get_data_ids_for_dataset(dataset_id: UUID):
    """Get all data IDs for a dataset."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(DatasetData.data_id)
            .filter(DatasetData.dataset_id == dataset_id)
        )
        return [str(row[0]) for row in result.all()]


async def investigate():
    """Main investigation function."""
    user = await get_default_user()
    print(f"Default user ID: {user.id}")
    print("=" * 80)
    
    # Target dataset IDs from the error
    target_dataset_id = UUID("9107cf1c-e8e8-5869-9c87-64e9922843df")
    other_dataset_id = UUID("c27cf65d-7bb3-5b16-b944-4a5db7519143")
    duplicate_key = "c0f3a1ff-b9c8-515f-943a-fb7d6814deee"
    
    print(f"\n=== Investigation of bri_knowledge_base ===")
    print(f"Target dataset ID (intended): {target_dataset_id}")
    print(f"Other dataset ID (not intended): {other_dataset_id}")
    print(f"Duplicate key causing error: {duplicate_key}")
    print()
    
    # Get all datasets named bri_knowledge_base
    datasets = await get_datasets_with_name("bri_knowledge_base")
    print(f"Found {len(datasets)} dataset(s) named 'bri_knowledge_base':")
    print("-" * 80)
    
    all_data_ids = {}
    for ds in datasets:
        count = await count_data_for_dataset(ds.id)
        data_ids = await get_data_ids_for_dataset(ds.id)
        all_data_ids[str(ds.id)] = set(data_ids)
        
        is_target = " (TARGET - intended)" if str(ds.id) == str(target_dataset_id) else ""
        is_other = " (OTHER - not intended)" if str(ds.id) == str(other_dataset_id) else ""
        
        print(f"\nDataset ID: {ds.id}{is_target}{is_other}")
        print(f"  Owner ID: {ds.owner_id}")
        print(f"  Created: {ds.created_at}")
        print(f"  Document count: {count}")
        
        # Check if duplicate key is in this dataset
        if duplicate_key in data_ids:
            print(f"  ⚠️  Contains the duplicate key: {duplicate_key}")
    
    print("\n" + "=" * 80)
    
    # Find overlapping data items between datasets
    if len(all_data_ids) > 1:
        print("\n=== Overlap Analysis ===")
        dataset_ids = list(all_data_ids.keys())
        for i, ds1_id in enumerate(dataset_ids):
            for ds2_id in dataset_ids[i+1:]:
                overlap = all_data_ids[ds1_id] & all_data_ids[ds2_id]
                if overlap:
                    print(f"\nOverlap between {ds1_id} and {ds2_id}:")
                    print(f"  {len(overlap)} shared documents")
                    if len(overlap) <= 10:
                        for data_id in sorted(overlap):
                            print(f"    - {data_id}")
                    else:
                        for data_id in sorted(list(overlap))[:5]:
                            print(f"    - {data_id}")
                        print(f"    ... and {len(overlap) - 5} more")
    
    print("\n" + "=" * 80)
    print("\n=== Summary ===")
    
    # Document counts for specified datasets
    for ds_id, label in [(target_dataset_id, "TARGET"), (other_dataset_id, "OTHER")]:
        if str(ds_id) in all_data_ids:
            count = len(all_data_ids[str(ds_id)])
            print(f"{label} dataset ({ds_id}): {count} documents")
        else:
            # Check if dataset exists at all
            db_engine = get_relational_engine()
            async with db_engine.get_async_session() as session:
                result = await session.execute(
                    select(Dataset).filter(Dataset.id == ds_id)
                )
                ds = result.scalar()
                if ds:
                    count = await count_data_for_dataset(ds_id)
                    print(f"{label} dataset ({ds_id}): {count} documents (different name: {ds.name})")
                else:
                    print(f"{label} dataset ({ds_id}): NOT FOUND")
    
    print("\n" + "=" * 80)
    print("\n=== All Datasets Overview ===")
    all_datasets = await get_all_datasets()
    for ds in all_datasets:
        count = await count_data_for_dataset(ds.id)
        print(f"  {ds.name} ({ds.id}): {count} documents")


if __name__ == "__main__":
    asyncio.run(investigate())
