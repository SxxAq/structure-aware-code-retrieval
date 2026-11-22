"""End-to-end index pipeline constructing BM25 and FAISS indices from code units."""

from pathlib import Path
from typing import List, Optional
import time
import logging

from sahcr.models import ExtractedCodeUnit
from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer

logger = logging.getLogger(__name__)


class IndexPipeline:
    """Orchestrates building, persisting, and loading lexical and dense indices."""

    def __init__(
        self,
        indices_dir: str | Path = "data/indices",
        model_name: str = "microsoft/codebert-base",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        pooling: str = "mean",
        faiss_index_type: str = "flat"
    ):
        self.indices_dir = Path(indices_dir)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path = self.indices_dir / "bm25_index.pkl"
        self.faiss_index_path = self.indices_dir / "codebert.faiss"
        self.faiss_id_map_path = self.indices_dir / "codebert_id_map.json"

        self.bm25_indexer = BM25Indexer(k1=bm25_k1, b=bm25_b)
        self.encoder = CodeBERTEncoder(model_name=model_name, pooling=pooling)
        self.faiss_indexer = FAISSIndexer(dimension=768, index_type=faiss_index_type)

    def build_from_units(self, units: List[ExtractedCodeUnit], batch_size: int = 16):
        """Constructs both BM25 and FAISS indices from extracted code units."""
        if not units:
            logger.warning("No code units provided for indexing.")
            return

        # 1. Build BM25 Index
        t0 = time.perf_counter()
        self.bm25_indexer.build_index(units)
        self.bm25_indexer.save(self.bm25_path)
        logger.info(f"Built BM25 index over {len(units)} units in {(time.perf_counter() - t0):.2f}s")

        # 2. Build Dense FAISS Index
        t1 = time.perf_counter()
        doc_ids = [u.id for u in units]
        # Feed the source code snippet for CodeBERT representation
        code_texts = [u.source_code for u in units]
        vectors = self.encoder.encode_texts(code_texts, batch_size=batch_size)
        self.faiss_indexer.add_vectors(vectors, doc_ids)
        self.faiss_indexer.save(self.faiss_index_path, self.faiss_id_map_path)
        logger.info(f"Built FAISS index over {len(units)} units in {(time.perf_counter() - t1):.2f}s")

    def build_from_store(self, metadata_store: MetadataStore, batch_size: int = 16):
        """Loads all code units from SQLite metadata store and builds indices."""
        units = list(metadata_store.iter_all_units())
        self.build_from_units(units, batch_size=batch_size)

    def load_existing(self) -> bool:
        """Loads serialized BM25 and FAISS indices if they exist."""
        if self.bm25_path.exists() and self.faiss_index_path.exists() and self.faiss_id_map_path.exists():
            self.bm25_indexer = BM25Indexer.load(self.bm25_path)
            self.faiss_indexer = FAISSIndexer.load(self.faiss_index_path, self.faiss_id_map_path)
            return True
        return False

