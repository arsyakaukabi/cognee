
import pytest
from unittest.mock import AsyncMock, patch
from cognee.api.v1.search.retrieval import retrieve
from cognee.modules.search.custom import search_knowledge_ids

@pytest.mark.asyncio
async def test_search_knowledge_ids_integration():
    """Test integration of search_knowledge_ids into retrieval endpoint"""
    
    # Mock the search_knowledge_ids function
    with patch("cognee.modules.search.custom.search_knowledge_ids", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = ["produk__id1", "wi__id2"]
        
        # Test retrieve with custom type
        results = await retrieve(
            query="test query", 
            top_k=5, 
            search_type="graph_completion_custom"
        )
        
        # Verify search was called correctly
        mock_search.assert_called_once_with("test query", method="graph", target_n=5)
        
        # Verify results parsing
        assert len(results) == 2
        assert results[0].id_knowledge == "id1"
        assert results[0].knowledge_type == "produk"
        assert results[1].id_knowledge == "id2"
        assert results[1].knowledge_type == "wi"

@pytest.mark.asyncio
async def test_search_router_payload():
    """Test retrieval payload validation"""
    from cognee.api.v1.search.routers.get_search_router import RetrievalPayloadDTO
    
    payload = RetrievalPayloadDTO(
        query="test",
        search_type="graph_completion_custom",
        top_k=5
    )
    assert payload.search_type == "graph_completion_custom"


@pytest.mark.asyncio
async def test_retrieve_with_summary():
    """Test that retrieval includes summary field"""
    with patch("cognee.modules.search.custom.search_knowledge_ids", new_callable=AsyncMock) as mock_search:
        # Mock batch summary retrieval
        with patch("cognee.api.v1.search.retrieval._get_summaries_batch", new_callable=AsyncMock) as mock_summary:
            with patch("cognee.api.v1.search.retrieval.get_graph_engine", new_callable=AsyncMock):
                
                mock_search.return_value = ["produk__id1"]
                mock_summary.return_value = {"produk__id1": "This is a test summary"}
                
                results = await retrieve(
                    query="test query", 
                    top_k=5, 
                    search_type="graph_completion_custom"
                )
                
                assert len(results) == 1
                assert results[0].id_knowledge == "id1"
                assert results[0].knowledge_type == "produk"
                assert results[0].summary == "This is a test summary"
