"""Lexical Indexer using BM25Okapi for source code retrieval."""

import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Plus, BM25Okapi

from sahcr.models import ExtractedCodeUnit
from sahcr.parsers.normalizer import tokenize_text


class BM25Indexer:
    """Constructs and queries a BM25 index over extracted code units."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, algorithm: str = "bm25plus"):
        self.k1 = k1
        self.b = b
        self.algorithm = algorithm.lower()
        self.doc_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[Any] = None

    def build_index(self, units: List[ExtractedCodeUnit]):
        """Builds the BM25 index from a list of ExtractedCodeUnit objects."""
        self.doc_ids = []
        self.corpus_tokens = []

        for unit in units:
            searchable_text = unit.get_searchable_lexical_text()
            tokens = tokenize_text(searchable_text, remove_stopwords=True)
            if not tokens:
                # Fallback to function name tokens if empty
                tokens = unit.normalized_name_tokens or [unit.name.lower()]
            self.doc_ids.append(unit.id)
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            if self.algorithm == "bm25plus":
                self.bm25 = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b)
            else:
                self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b, epsilon=0.25)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Searches the BM25 index for a natural language or keyword query.
        Returns a list of (unit_id, bm25_score) tuples sorted in descending order.
        """
        if not self.bm25 or not self.doc_ids:
            return []

        query_tokens = tokenize_text(query, remove_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize_text(query, remove_stopwords=False)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices with non-zero or best scores
        scored_pairs = [(self.doc_ids[idx], float(scores[idx])) for idx in range(len(self.doc_ids))]
        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]

    def save(self, file_path: str | Path):
        """Serializes the index to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "corpus_tokens": self.corpus_tokens,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25Indexer":
        """Loads a serialized index from disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        indexer = cls(k1=data["k1"], b=data["b"])
        indexer.doc_ids = data["doc_ids"]
        indexer.corpus_tokens = data["corpus_tokens"]
        indexer.bm25 = data["bm25"]
        return indexer

