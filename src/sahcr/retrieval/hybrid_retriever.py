"""Hybrid lexical-semantic retriever using Reciprocal Rank Fusion (Config C)."""

import time
from typing import List, Tuple, Dict, Any, Optional

from sahcr.models import ExtractedCodeUnit
from sahcr.retrieval.retrievers import RetrievalResult, RetrievalResponse, LexicalRetriever, DenseRetriever
from sahcr.retrieval.fusion import reciprocal_rank_fusion
from sahcr.indexing.metadata_store import MetadataStore


class HybridRetriever:
    """Configuration C: Standard Lexical-Semantic Hybrid Baseline (BM25 + CodeBERT + RRF)."""

    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        dense_retriever: DenseRetriever,
        metadata_store: Optional[MetadataStore] = None,
        rrf_k: int = 60
    ):
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.metadata_store = metadata_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        lexical_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None,
        fetch_metadata: bool = True
    ) -> RetrievalResponse:
        t0 = time.perf_counter()

        lex_k = lexical_top_k or top_k
        dense_k = dense_top_k or top_k
        fuse_k = fusion_top_k or top_k

        # Retrieve independently from BM25 and FAISS
        lex_resp = self.lexical_retriever.retrieve(query, top_k=lex_k, fetch_metadata=False)
        dense_resp = self.dense_retriever.retrieve(query, top_k=dense_k, fetch_metadata=False)

        lex_rankings = [(r.unit_id, r.score) for r in lex_resp.results]
        dense_rankings = [(r.unit_id, r.score) for r in dense_resp.results]

        # Apply Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([lex_rankings, dense_rankings], k=self.rrf_k, top_k=fuse_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Fetch unit metadata
        unit_ids = [doc_id for doc_id, _ in fused]
        metadata_map = {}
        if fetch_metadata and self.metadata_store and unit_ids:
            metadata_map = self.metadata_store.get_units_by_ids(unit_ids)

        results = []
        for rank, (uid, rrf_score) in enumerate(fused, start=1):
            results.append(
                RetrievalResult(
                    unit_id=uid,
                    score=rrf_score,
                    rank=rank,
                    retrieval_method="hybrid",
                    unit=metadata_map.get(uid)
                )
            )

        return RetrievalResponse(
            query=query,
            method="hybrid",
            results=results,
            latency_ms=latency_ms,
            total_candidates=len(results)
        )
