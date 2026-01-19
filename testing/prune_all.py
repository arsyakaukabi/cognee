import asyncio
from cognee.modules.data.deletion import prune_data, prune_system

async def main():
    await prune_data()  # clears data_root_directory
    await prune_system(graph=True, vector=True, metadata=True, cache=True)

asyncio.run(main())

# rm -rf .data_storage .cognee_system .cognee_cache
# psql -h 127.0.0.1 -U admin -d postgres -c "DROP DATABASE IF EXISTS bribrain_knowledge_base;"
# psql -h 127.0.0.1 -U admin -d postgres -c "CREATE DATABASE bribrain_knowledge_base;"
