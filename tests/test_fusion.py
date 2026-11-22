"""Unit tests for Reciprocal Rank Fusion (RRF) and HybridRetriever (Config C)."""

import pytest
from sahcr.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_math():
    # List 1: doc_A at rank 1, doc_B at rank 2
    list_1 = [("doc_A", 10.0), ("doc_B", 5.0)]
    # List 2: doc_B at rank 1, doc_C at rank 2
    list_2 = [("doc_B", 0.99), ("doc_C", 0.85)]

    # Expected RRF scores (with k=60):
    # doc_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # doc_A: 1/(60+1) = 1/61 = 0.016393
    # doc_C: 1/(60+2) = 1/62 = 0.016129

    fused = reciprocal_rank_fusion([list_1, list_2], k=60, top_k=3)
    assert len(fused) == 3

    # doc_B appeared in both lists and should rank #1
    assert fused[0][0] == "doc_B"
    assert fused[1][0] == "doc_A"
    assert fused[2][0] == "doc_C"

    # Check approximate score
    expected_doc_b_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert pytest.approx(fused[0][1], abs=1e-5) == expected_doc_b_score
