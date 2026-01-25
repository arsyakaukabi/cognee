"""
Unit tests for RRF (Reciprocal Rank Fusion) implementation.
"""

import pytest
from cognee.modules.retrieval.utils.rrf_fusion import rrf_fusion, rrf_fusion_with_payloads


class TestRRFFusion:
    """Tests for the rrf_fusion function."""

    def test_empty_input(self):
        """Empty input should return empty results."""
        result = rrf_fusion([])
        assert result == []

    def test_empty_lists(self):
        """Empty result lists should return empty results."""
        result = rrf_fusion([[], []])
        assert result == []

    def test_single_list(self):
        """Single result list should preserve order with RRF scores."""
        results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        fused = rrf_fusion([results], k=60)
        
        # First document should have highest RRF score
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        assert fused[2][0] == "doc3"
        
        # Check RRF scores are computed correctly
        # rank 1: 1/(60+1) = 0.0163...
        assert abs(fused[0][1] - 1.0 / 61) < 0.0001

    def test_two_lists_overlap(self):
        """Documents appearing in both lists should have summed RRF scores."""
        list1 = [("doc1", 0.9), ("doc2", 0.7)]
        list2 = [("doc2", 2.5), ("doc3", 1.0)]
        
        fused = rrf_fusion([list1, list2], k=60)
        
        # doc2 appears in both lists (rank 2 in list1, rank 1 in list2)
        # Should have combined score: 1/(60+2) + 1/(60+1)
        doc2_score = 1.0 / 62 + 1.0 / 61
        
        doc2_result = next(x for x in fused if x[0] == "doc2")
        assert abs(doc2_result[1] - doc2_score) < 0.0001

    def test_two_lists_no_overlap(self):
        """Disjoint lists should have independent RRF scores."""
        list1 = [("doc1", 0.9), ("doc2", 0.7)]
        list2 = [("doc3", 2.5), ("doc4", 1.0)]
        
        fused = rrf_fusion([list1, list2], k=60)
        
        # All documents should appear with their individual RRF scores
        doc_ids = [doc_id for doc_id, _ in fused]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids
        assert "doc4" in doc_ids

    def test_final_top_k_limit(self):
        """Should return at most final_top_k results."""
        results = [(f"doc{i}", i) for i in range(100)]
        fused = rrf_fusion([results], k=60, final_top_k=10)
        
        assert len(fused) == 10

    def test_deterministic_tie_breaking(self):
        """Documents with same RRF score should be ordered by ID."""
        # Create lists where doc_a and doc_b appear at same rank in different lists
        list1 = [("doc_b", 1.0)]
        list2 = [("doc_a", 1.0)]
        
        fused = rrf_fusion([list1, list2], k=60)
        
        # Both have same score (1/61), should be sorted by ID
        assert fused[0][0] == "doc_a"
        assert fused[1][0] == "doc_b"

    def test_k_parameter_effect(self):
        """Higher k should give more weight to lower-ranked documents."""
        results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        
        fused_low_k = rrf_fusion([results], k=1)
        fused_high_k = rrf_fusion([results], k=100)
        
        # With low k, score difference between ranks is larger
        low_k_diff = fused_low_k[0][1] - fused_low_k[1][1]
        high_k_diff = fused_high_k[0][1] - fused_high_k[1][1]
        
        assert low_k_diff > high_k_diff

    def test_many_lists(self):
        """Should handle fusion of many result lists."""
        lists = [
            [("doc1", 1.0), ("doc2", 0.5)],
            [("doc2", 1.0), ("doc3", 0.5)],
            [("doc3", 1.0), ("doc1", 0.5)],
        ]
        
        fused = rrf_fusion(lists, k=60)
        
        # All docs appear in 2 lists; scores should reflect this
        assert len(fused) == 3


class TestRRFFusionWithPayloads:
    """Tests for rrf_fusion_with_payloads function."""

    def test_preserves_payloads(self):
        """Should preserve payload data in results."""
        results = [
            ("doc1", 0.9, {"text": "hello world"}),
            ("doc2", 0.7, {"text": "test document"}),
        ]
        
        fused = rrf_fusion_with_payloads([results], k=60)
        
        assert len(fused) == 2
        doc1_result = next(x for x in fused if x[0] == "doc1")
        assert doc1_result[2] == {"text": "hello world"}

    def test_first_payload_wins(self):
        """When document appears in multiple lists, first payload is kept."""
        list1 = [("doc1", 0.9, {"source": "vector"})]
        list2 = [("doc1", 2.5, {"source": "bm25"})]
        
        fused = rrf_fusion_with_payloads([list1, list2], k=60)
        
        assert fused[0][2] == {"source": "vector"}

    def test_empty_input(self):
        """Empty input should return empty results."""
        result = rrf_fusion_with_payloads([])
        assert result == []

    def test_mixed_payloads(self):
        """Should handle mixed documents from multiple lists."""
        list1 = [("doc1", 0.9, {"a": 1})]
        list2 = [("doc2", 0.8, {"b": 2})]
        
        fused = rrf_fusion_with_payloads([list1, list2], k=60)
        
        payloads = {doc_id: payload for doc_id, _, payload in fused}
        assert payloads["doc1"] == {"a": 1}
        assert payloads["doc2"] == {"b": 2}
