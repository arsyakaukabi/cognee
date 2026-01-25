
import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, patch

# Add current directory to path so imports work
sys.path.append("/home/jupyter/backup_usr_data/cognee")

from cognee.api.v1.search.retrieval import retrieve
from cognee.modules.search.custom import search_knowledge_ids

class TestCustomRetrieval(unittest.TestCase):
    def test_integration(self):
        async def run_test():
            # Mock the search_knowledge_ids function
            # Patch where it is used, not where it is defined
            with patch("cognee.api.v1.search.retrieval.search_knowledge_ids", new_callable=AsyncMock) as mock_search:
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
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0].id_knowledge, "id1")
                self.assertEqual(results[0].knowledge_type, "produk")
                self.assertEqual(results[1].id_knowledge, "id2")
                self.assertEqual(results[1].knowledge_type, "wi")
                
                print("Retrieval integration test passed!")

        asyncio.run(run_test())

    def test_router_payload(self):
        from cognee.api.v1.search.routers.get_search_router import RetrievalPayloadDTO
        
        payload = RetrievalPayloadDTO(
            query="test",
            search_type="graph_completion_custom",
            top_k=5
        )
        self.assertEqual(payload.search_type, "graph_completion_custom")
        print("Payload validation test passed!")

if __name__ == "__main__":
    unittest.main()
