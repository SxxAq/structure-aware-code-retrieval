"""Two-stage AST-based structural reranker for hybrid code search (Config D)."""

import time
from typing import List, Tuple, Dict, Any, Optional

from sahcr.models import ExtractedCodeUnit
from sahcr.retrieval.retrievers import RetrievalResult, RetrievalResponse
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.structural_features import StructuralFeatureExtractor
from sahcr.indexing.metadata_store import MetadataStore


class ASTRerankedRetriever:
    """
    Configuration D: Proposed Structure-Aware Hybrid Retrieval System.
    Stage 1: Generates candidate pool using Reciprocal Rank Fusion (BM25 + CodeBERT).
    Stage 2: Re-ranks top candidates using explicit AST structural features.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        metadata_store: MetadataStore,
        feature_extractor: Optional[StructuralFeatureExtractor] = None,
        alpha: float = 0.60,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        self.hybrid_retriever = hybrid_retriever
        self.metadata_store = metadata_store
        self.feature_extractor = feature_extractor or StructuralFeatureExtractor()
        self.alpha = alpha  # Weight for RRF score vs AST score

        # Default weights for structural components
        self.weights = feature_weights or {
            "identifier_sim": 0.40,
            "param_match": 0.30,
            "return_match": 0.15,
            "ast_node_match": 0.15,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        stage1_top_k: int = 50,
        final_top_k: Optional[int] = None
    ) -> RetrievalResponse:
        t0 = time.perf_counter()
        k_out = final_top_k or top_k

        # Stage 1: Candidate Generation via Hybrid Fusion (BM25 + CodeBERT RRF)
        stage1_resp = self.hybrid_retriever.retrieve(
            query=query,
            fusion_top_k=stage1_top_k,
            fetch_metadata=True
        )

        candidates = stage1_resp.results
        if not candidates:
            return RetrievalResponse(
                query=query,
                method="ast_reranked",
                results=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                total_candidates=0
            )

        # Normalize RRF scores across candidate pool to [0, 1]
        rrf_scores = [c.score for c in candidates]
        max_rrf = max(rrf_scores) if rrf_scores else 1.0
        min_rrf = min(rrf_scores) if rrf_scores else 0.0
        rrf_range = (max_rrf - min_rrf) if (max_rrf - min_rrf) > 1e-9 else 1.0

        reranked_candidates = []

        # Stage 2: Structural AST Scoring
        for candidate in candidates:
            unit = candidate.unit
            if not unit:
                # If metadata wasn't attached, attempt lookup
                unit = self.metadata_store.get_unit(candidate.unit_id)

            if unit:
                features = self.feature_extractor.compute_features(query, unit)
                ast_score = (
                    self.weights["identifier_sim"] * features["phi_identifier"] +
                    self.weights["param_match"] * features["phi_parameters"] +
                    self.weights["return_match"] * features["phi_return"] +
                    self.weights["ast_node_match"] * features["phi_ast_nodes"]
                )
            else:
                ast_score = 0.0

            # Normalized RRF score
            norm_rrf = (candidate.score - min_rrf) / rrf_range

            # Combined Stage 2 Final Score
            final_score = self.alpha * norm_rrf + (1.0 - self.alpha) * ast_score

            reranked_candidates.append({
                "unit_id": candidate.unit_id,
                "final_score": final_score,
                "ast_score": ast_score,
                "rrf_score": candidate.score,
                "unit": unit
            })

        # Sort by final score descending
        reranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        results = []
        for rank, item in enumerate(reranked_candidates[:k_out], start=1):
            results.append(
                RetrievalResult(
                    unit_id=item["unit_id"],
                    score=float(item["final_score"]),
                    rank=rank,
                    retrieval_method="ast_reranked",
                    unit=item["unit"]
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResponse(
            query=query,
            method="ast_reranked",
            results=results,
            latency_ms=latency_ms,
            total_candidates=len(results)
        )
