import asyncio

import cognee
from cognee.modules.search.types import SearchType

QUERY_TEXT = "contoh query untuk cari chunk"
DATASETS = None  # contoh: ["nama_dataset"]
TOP_K = 5


async def main():
    results = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text=QUERY_TEXT,
        datasets=DATASETS,
        top_k=TOP_K,
    )

    print("=== CHUNKS RESULTS ===")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item}")


if __name__ == "__main__":
    asyncio.run(main())
