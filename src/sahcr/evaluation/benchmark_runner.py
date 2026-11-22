"""Benchmark runner orchestrating comparative evaluation across Configurations A, B, C, D."""

from typing import List, Dict, Any, Optional
from sahcr.evaluation.dataloader import BenchmarkQuery
from sahcr.evaluation.metrics import BenchmarkEvaluator, MetricReport
from sahcr.evaluation.statistical_test import compare_paired_systems, SignificanceResult
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.ast_reranker import ASTRerankedRetriever


class BenchmarkRunner:
    """Executes controlled comparisons across Configurations A, B, C, and D."""

    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        ast_retriever: ASTRerankedRetriever
    ):
        self.systems = {
            "Config A (BM25)": lexical_retriever,
            "Config B (CodeBERT)": dense_retriever,
            "Config C (Hybrid RRF)": hybrid_retriever,
            "Config D (Hybrid + AST)": ast_retriever,
        }

    def run_evaluation(
        self,
        queries: List[BenchmarkQuery],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Runs evaluation over all systems and computes comparative metrics and significance."""
        evaluators: Dict[str, BenchmarkEvaluator] = {
            name: BenchmarkEvaluator(name) for name in self.systems
        }

        # Store per-query reciprocal ranks for paired significance testing
        per_query_rr: Dict[str, List[float]] = {name: [] for name in self.systems}

        for q in queries:
            for name, retriever in self.systems.items():
                resp = retriever.retrieve(q.query_text, top_k=top_k)
                ranked_ids = [r.unit_id for r in resp.results]
                evaluators[name].record_query_result(ranked_ids, q.target_unit_id, resp.latency_ms)

        reports: Dict[str, MetricReport] = {
            name: ev.compute_report() for name, ev in evaluators.items()
        }

        # Paired Significance Tests (Config D vs Config C and Config D vs Config B)
        sig_d_vs_c = compare_paired_systems(
            system_a_name="Config C (Hybrid RRF)",
            system_b_name="Config D (Hybrid + AST)",
            metric_name="MRR@10",
            scores_a=evaluators["Config C (Hybrid RRF)"].rr_list,
            scores_b=evaluators["Config D (Hybrid + AST)"].rr_list
        )

        sig_d_vs_b = compare_paired_systems(
            system_a_name="Config B (CodeBERT)",
            system_b_name="Config D (Hybrid + AST)",
            metric_name="MRR@10",
            scores_a=evaluators["Config B (CodeBERT)"].rr_list,
            scores_b=evaluators["Config D (Hybrid + AST)"].rr_list
        )

        return {
            "reports": reports,
            "significance_d_vs_c": sig_d_vs_c,
            "significance_d_vs_b": sig_d_vs_b,
        }
