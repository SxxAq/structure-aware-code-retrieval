"""Reciprocal Rank Fusion (RRF) for merging candidate rankings."""

from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict


def reciprocal_rank_fusion(
    rankings_list: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Combines multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).
    Formula:
        RRF_score(d) = sum_{m} 1 / (k + rank_m(d))
    where rank_m(d) is 1-based rank in retrieval method m.

    Args:
        rankings_list: List of ranked candidate lists, each containing (doc_id, original_score)
        k: Smoothing constant (default 60)
        top_k: Number of fused candidates to return

    Returns:
        List of (doc_id, rrf_score) tuples sorted in descending order of rrf_score.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranking in rankings_list:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort candidates by combined RRF score descending
    fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_candidates[:top_k]

