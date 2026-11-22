"""Unit tests for metrics, statistical significance, and dataloader."""

import pytest
from sahcr.evaluation.metrics import (
    reciprocal_rank,
    recall_at_k,
    ndcg_at_k,
    BenchmarkEvaluator,
)
from sahcr.evaluation.statistical_test import compare_paired_systems
from sahcr.evaluation.dataloader import categorize_query


def test_evaluation_metric_computations():
    ranked = ["doc_A", "doc_B", "doc_C", "doc_D"]

    # Target is doc_A (Rank 1)
    assert reciprocal_rank(ranked, "doc_A", k=10) == 1.0
    assert recall_at_k(ranked, "doc_A", k=1) == 1.0
    assert ndcg_at_k(ranked, "doc_A", k=10) == 1.0

    # Target is doc_B (Rank 2)
    assert reciprocal_rank(ranked, "doc_B", k=10) == 0.5
    assert recall_at_k(ranked, "doc_B", k=1) == 0.0
    assert recall_at_k(ranked, "doc_B", k=2) == 1.0

    # Target not in top 2
    assert reciprocal_rank(ranked, "doc_C", k=2) == 0.0
    assert recall_at_k(ranked, "doc_C", k=2) == 0.0


def test_statistical_significance_test():
    # System A (baseline) vs System B (proposed superior system)
    scores_a = [0.1, 0.2, 0.15, 0.3, 0.25, 0.1, 0.2, 0.15, 0.3, 0.2]
    scores_b = [0.8, 0.9, 0.85, 0.95, 0.9, 0.75, 0.85, 0.8, 0.95, 0.85]

    res = compare_paired_systems("SystemA", "SystemB", "MRR@10", scores_a, scores_b)
    assert res.mean_b > res.mean_a
    assert res.delta_percentage > 0.0
    assert res.wilcoxon_p_value < 0.05
    assert res.is_significant_05


def test_query_categorization():
    cat1 = categorize_query("takes list of strings and returns dictionary")
    assert cat1 == "signature_constrained"

    cat2 = categorize_query("authenticate user session")
    assert cat2 == "conceptual"
