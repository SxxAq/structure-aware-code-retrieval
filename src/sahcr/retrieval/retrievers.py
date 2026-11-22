"""Candidate retrieval wrappers for Lexical (BM25) and Dense (CodeBERT) search."""

import time
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from sahcr.models import ExtractedCodeUnit
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer
from sahcr.indexing.metadata_store import MetadataStore


class RetrievalResult(BaseModel):
    """Represents a single retrieved candidate with score and unit metadata."""
    unit_id: str
    score: float
    rank: int
    retrieval_method: str  # "bm25", "dense", "hybrid", "ast_reranked"
    unit: Optional[ExtractedCodeUnit] = None


class RetrievalResponse(BaseModel):
    """Encapsulates retrieval response, scores, and execution diagnostics."""
    query: str
    method: str
    results: List[RetrievalResult] = Field(default_factory=list)
    latency_ms: float
    total_candidates: int


class LexicalRetriever:
    """Configuration A: BM25 Lexical Retrieval baseline."""

    def __init__(self, indexer: BM25Indexer, metadata_store: Optional[MetadataStore] = None):
        self.indexer = indexer
        self.metadata_store = metadata_store

    def retrieve(self, query: str, top_k: int = 50, fetch_metadata: bool = True) -> RetrievalResponse:
        t0 = time.perf_counter()
        scored_pairs = self.indexer.search(query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        unit_ids = [p[0] for p in scored_pairs]
        metadata_map = {}
        if fetch_metadata and self.metadata_store and unit_ids:
            metadata_map = self.metadata_store.get_units_by_ids(unit_ids)

        results = []
        for rank, (uid, score) in enumerate(scored_pairs, start=1):
            results.append(
                RetrievalResult(
                    unit_id=uid,
                    score=score,
                    rank=rank,
                    retrieval_method="bm25",
                    unit=metadata_map.get(uid)
                )
            )

        return RetrievalResponse(
            query=query,
            method="bm25",
            results=results,
            latency_ms=latency_ms,
            total_candidates=len(results)
        )


class DenseRetriever:
    """Configuration B: CodeBERT + FAISS Dense Semantic Retrieval baseline."""

    def __init__(
        self,
        encoder: CodeBERTEncoder,
        indexer: FAISSIndexer,
        metadata_store: Optional[MetadataStore] = None
    ):
        self.encoder = encoder
        self.indexer = indexer
        self.metadata_store = metadata_store

    def retrieve(self, query: str, top_k: int = 50, fetch_metadata: bool = True) -> RetrievalResponse:
        t0 = time.perf_counter()
        q_vec = self.encoder.encode_query(query)
        scored_pairs = self.indexer.search(q_vec, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        unit_ids = [p[0] for p in scored_pairs]
        metadata_map = {}
        if fetch_metadata and self.metadata_store and unit_ids:
            metadata_map = self.metadata_store.get_units_by_ids(unit_ids)

        results = []
        for rank, (uid, score) in enumerate(scored_pairs, start=1):
            results.append(
                RetrievalResult(
                    unit_id=uid,
                    score=score,
                    rank=rank,
                    retrieval_method="dense",
                    unit=metadata_map.get(uid)
                )
            )

        return RetrievalResponse(
            query=query,
            method="dense",
            results=results,
            latency_ms=latency_ms,
            total_candidates=len(results)
        )

