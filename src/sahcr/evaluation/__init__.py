"""Evaluation package public interfaces."""

from sahcr.evaluation.metrics import (
    MetricReport,
    BenchmarkEvaluator,
    reciprocal_rank,
    recall_at_k,
    ndcg_at_k,
)
from sahcr.evaluation.statistical_test import (
    SignificanceResult,
    compare_paired_systems,
)
from sahcr.evaluation.dataloader import (
    BenchmarkQuery,
    BenchmarkDataLoader,
    categorize_query,
)
from sahcr.evaluation.benchmark_runner import BenchmarkRunner

__all__ = [
    "MetricReport",
    "BenchmarkEvaluator",
    "reciprocal_rank",
    "recall_at_k",
    "ndcg_at_k",
    "SignificanceResult",
    "compare_paired_systems",
    "BenchmarkQuery",
    "BenchmarkDataLoader",
    "categorize_query",
    "BenchmarkRunner",
]

