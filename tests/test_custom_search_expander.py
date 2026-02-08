from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.modules.search.custom.expander import (
    expand_chunk_search_to_n_ids,
    expand_graph_completion_to_n_ids,
)


@pytest.mark.asyncio
async def test_expand_graph_completion_keeps_node_name_filter():
    search_executor = SimpleNamespace(
        execute_graph_completion_search=AsyncMock(return_value=[])
    )

    with patch(
        "cognee.modules.search.custom.expander.map_triplet_results_to_knowledge_ids",
        new_callable=AsyncMock,
    ) as mock_map:
        mock_map.return_value = ["eng_01"]

        await expand_graph_completion_to_n_ids(
            triplets=[Mock()],
            target_n=2,
            graph_engine=Mock(),
            search_executor=search_executor,
            query="overview",
            initial_top_k=10,
            max_top_k=20,
            node_name=["eng"],
        )

        search_executor.execute_graph_completion_search.assert_awaited_once_with(
            "overview", top_k=15, node_name=["eng"]
        )


@pytest.mark.asyncio
async def test_expand_chunk_search_keeps_node_name_filter():
    search_executor = SimpleNamespace(execute_chunk_search=AsyncMock(return_value=[]))

    with patch(
        "cognee.modules.search.custom.expander.map_chunk_results_to_knowledge_ids",
        new_callable=AsyncMock,
    ) as mock_map:
        mock_map.return_value = ["finance_01"]

        await expand_chunk_search_to_n_ids(
            chunk_results=[Mock()],
            target_n=2,
            graph_engine=Mock(),
            search_executor=search_executor,
            query="overview",
            initial_top_k=10,
            max_top_k=20,
            node_name=["finance"],
        )

        search_executor.execute_chunk_search.assert_awaited_once_with(
            "overview", top_k=15, node_name=["finance"]
        )
