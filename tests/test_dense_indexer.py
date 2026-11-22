"""Unit tests for CodeBERTEncoder and FAISSIndexer."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer


@pytest.fixture(scope="module")
def encoder():
    return CodeBERTEncoder(pooling="mean")


def test_codebert_encoder_dimensions_and_normalization(encoder):
    texts = [
        "def parse_token(header: str) -> str:\n    return header.split()[1]",
        "def sum_two_numbers(a: int, b: int) -> int:\n    return a + b"
    ]
    embeddings = encoder.encode_texts(texts, batch_size=2)

    # Shape should be (2, 768)
    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32

    # L2 norms should be approximately 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_faiss_indexer_roundtrip():
    dim = 768
    indexer = FAISSIndexer(dimension=dim, index_type="flat")

    # Generate 3 normalized mock vectors
    v1 = np.random.randn(dim).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(dim).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    vectors = np.vstack([v1, v2])
    doc_ids = ["doc_1", "doc_2"]

    indexer.add_vectors(vectors, doc_ids)

    # Search with v1, top hit must be doc_1 with similarity ~1.0
    results = indexer.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "doc_1"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        idx_file = Path(tmpdir) / "index.faiss"
        map_file = Path(tmpdir) / "map.json"

        indexer.save(idx_file, map_file)
        loaded = FAISSIndexer.load(idx_file, map_file)

        loaded_results = loaded.search(v1, top_k=2)
        assert loaded_results[0][0] == "doc_1"
        assert pytest.approx(loaded_results[0][1], abs=1e-4) == 1.0

