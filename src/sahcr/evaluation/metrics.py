"""Information retrieval evaluation metrics: MRR, Recall@K, NDCG@K, and Latency."""

import math
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel


class MetricReport(BaseModel):
    """Aggregate evaluation report for a retrieval configuration."""
    configuration: str
    num_queries: int
    mrr_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float


def reciprocal_rank(ranked_ids: List[str], target_id: str, k: int = 10) -> float:
    """Calculates Reciprocal Rank at cutoff k."""
    cutoff_ids = ranked_ids[:k]
    for rank, doc_id in enumerate(cutoff_ids, start=1):
        if doc_id == target_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: List[str], target_id: str, k: int = 10) -> float:
    """Calculates binary Recall at cutoff k (1.0 if target in top-k, else 0.0)."""
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def ndcg_at_k(ranked_ids: List[str], target_id: str, k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain at cutoff k with binary relevance."""
    cutoff_ids = ranked_ids[:k]
    for rank, doc_id in enumerate(cutoff_ids, start=1):
        if doc_id == target_id:
            dcg = 1.0 / math.log2(rank + 1)
            idcg = 1.0 / math.log2(1 + 1)  # Ideal position is 1
            return dcg / idcg
    return 0.0


class BenchmarkEvaluator:
    """Aggregates retrieval metrics across a set of query evaluation runs."""

    def __init__(self, configuration_name: str):
        self.configuration_name = configuration_name
        self.rr_list: List[float] = []
        self.recall_1_list: List[float] = []
        self.recall_5_list: List[float] = []
        self.recall_10_list: List[float] = []
        self.ndcg_list: List[float] = []
        self.latency_list: List[float] = []

    def record_query_result(self, ranked_ids: List[str], target_id: str, latency_ms: float):
        """Records metrics for a single query."""
        self.rr_list.append(reciprocal_rank(ranked_ids, target_id, k=10))
        self.recall_1_list.append(recall_at_k(ranked_ids, target_id, k=1))
        self.recall_5_list.append(recall_at_k(ranked_ids, target_id, k=5))
        self.recall_10_list.append(recall_at_k(ranked_ids, target_id, k=10))
        self.ndcg_list.append(ndcg_at_k(ranked_ids, target_id, k=10))
        self.latency_list.append(latency_ms)

    def compute_report(self) -> MetricReport:
        """Computes summary mean metrics and latency percentiles."""
        n = len(self.rr_list)
        if n == 0:
            return MetricReport(
                configuration=self.configuration_name,
                num_queries=0,
                mrr_at_10=0.0,
                recall_at_1=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                ndcg_at_10=0.0,
                latency_mean_ms=0.0,
                latency_p50_ms=0.0,
                latency_p90_ms=0.0,
                latency_p99_ms=0.0,
            )

        lat = np.array(self.latency_list)
        return MetricReport(
            configuration=self.configuration_name,
            num_queries=n,
            mrr_at_10=float(np.mean(self.rr_list)),
            recall_at_1=float(np.mean(self.recall_1_list)),
            recall_at_5=float(np.mean(self.recall_5_list)),
            recall_at_10=float(np.mean(self.recall_10_list)),
            ndcg_at_10=float(np.mean(self.ndcg_list)),
            latency_mean_ms=float(np.mean(lat)),
            latency_p50_ms=float(np.percentile(lat, 50)),
            latency_p90_ms=float(np.percentile(lat, 90)),
            latency_p99_ms=float(np.percentile(lat, 99)),
        )
