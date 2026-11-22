"""Dense vector indexing and query encoding using CodeBERT and FAISS-CPU."""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

from sahcr.models import ExtractedCodeUnit


class CodeBERTEncoder:
    """Encodes natural language queries and code snippets into dense vectors using CodeBERT."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        pooling: str = "cls",
        max_length: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.pooling = pooling.lower()
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Try loading from local cache first to avoid remote HTTP checks and timeouts
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encodes a list of texts into normalized L2 float32 vectors.
        Output shape: (len(texts), hidden_dim = 768).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

                if self.pooling == "mean":
                    # Mean pooling over valid attention tokens
                    sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    # CLS token pooling (first token)
                    embeddings = last_hidden[:, 0, :]

                # L2 normalize vectors for cosine similarity via inner product
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1D normalized vector (768,)."""
        vectors = self.encode_texts([query], batch_size=1)
        return vectors[0]


class FAISSIndexer:
    """Manages FAISS-CPU vector index and mapping from integer IDs to unit IDs."""

    def __init__(self, dimension: int = 768, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.doc_ids: List[str] = []

        if self.index_type == "flat":
            # Inner product index (equivalent to cosine similarity on L2-normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

    def add_vectors(self, vectors: np.ndarray, doc_ids: List[str]):
        """Adds normalized vectors and their corresponding document IDs."""
        if len(vectors) != len(doc_ids):
            raise ValueError("Vectors and doc_ids length mismatch")
        if len(vectors) == 0:
            return

        self.index.add(vectors.astype(np.float32))
        self.doc_ids.extend(doc_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Searches nearest neighbors for a single query vector.
        Returns list of (unit_id, cosine_similarity_score) sorted descending.
        """
        if self.index.ntotal == 0 or not self.doc_ids:
            return []

        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))

        return results

    def save(self, index_file_path: str | Path, id_map_file_path: str | Path):
        """Saves the FAISS binary index and JSON ID map to disk."""
        idx_p = Path(index_file_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(idx_p))

        id_p = Path(id_map_file_path)
        with open(id_p, "w", encoding="utf-8") as f:
            json.dump({"doc_ids": self.doc_ids, "dimension": self.dimension, "index_type": self.index_type}, f)

    @classmethod
    def load(cls, index_file_path: str | Path, id_map_file_path: str | Path) -> "FAISSIndexer":
        """Loads index and ID map from disk."""
        id_p = Path(id_map_file_path)
        with open(id_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        indexer = cls(dimension=meta["dimension"], index_type=meta["index_type"])
        indexer.index = faiss.read_index(str(index_file_path))
        indexer.doc_ids = meta["doc_ids"]
        return indexer

