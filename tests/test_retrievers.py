"""Integration tests for LexicalRetriever (Config A) and DenseRetriever (Config B)."""

import tempfile
from pathlib import Path
import pytest

from sahcr.models import ExtractedCodeUnit
from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever


@pytest.fixture(scope="module")
def encoder():
    return CodeBERTEncoder(pooling="cls")


def test_lexical_and_dense_retrievers(encoder):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_meta.sqlite"
        store = MetadataStore(db_path)

        u1 = ExtractedCodeUnit(
            id="auth.py:extract_bearer_token:1",
            file_path="auth.py",
            language="python",
            name="extract_bearer_token",
            normalized_name_tokens=["extract", "bearer", "token"],
            parameter_tokens=["header", "str"],
            return_tokens=["str"],
            docstring="Extract bearer authorization token from HTTP request header.",
            docstring_tokens=["extract", "bearer", "authorization", "token", "http", "request", "header"],
            source_code="def extract_bearer_token(header: str) -> str:\n    return header.split(' ')[1]",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=60
        )

        u2 = ExtractedCodeUnit(
            id="db.py:connect_sqlite:10",
            file_path="db.py",
            language="python",
            name="connect_sqlite",
            normalized_name_tokens=["connect", "sqlite"],
            parameter_tokens=["db", "path", "str"],
            return_tokens=["connection"],
            docstring="Establish local connection to SQLite database.",
            docstring_tokens=["establish", "local", "connection", "sqlite", "database"],
            source_code="def connect_sqlite(db_path: str):\n    return sqlite3.connect(db_path)",
            start_line=10,
            end_line=11,
            start_byte=100,
            end_byte=160
        )

        u3 = ExtractedCodeUnit(
            id="math.py:compute_factorial:20",
            file_path="math.py",
            language="python",
            name="compute_factorial",
            normalized_name_tokens=["compute", "factorial"],
            parameter_tokens=["n", "int"],
            return_tokens=["int"],
            docstring="Compute the factorial of non-negative integer n.",
            docstring_tokens=["compute", "factorial", "non", "negative", "integer", "n"],
            source_code="def compute_factorial(n: int) -> int:\n    return 1 if n <= 1 else n * compute_factorial(n - 1)",
            start_line=20,
            end_line=21,
            start_byte=200,
            end_byte=280
        )

        store.save_units([u1, u2, u3])

        # 1. Test Lexical Retriever (Config A)
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_index([u1, u2, u3])
        lexical = LexicalRetriever(bm25_indexer, metadata_store=store)

        resp_lex = lexical.retrieve("extract authorization token from header", top_k=2)
        assert resp_lex.method == "bm25"
        assert resp_lex.latency_ms > 0
        assert len(resp_lex.results) > 0
        assert resp_lex.results[0].unit_id == "auth.py:extract_bearer_token:1"
        assert resp_lex.results[0].unit is not None
        assert resp_lex.results[0].unit.name == "extract_bearer_token"

        # 2. Test Dense Retriever (Config B)
        faiss_indexer = FAISSIndexer(dimension=768, index_type="flat")
        code_vectors = encoder.encode_texts([u1.source_code, u2.source_code, u3.source_code])
        faiss_indexer.add_vectors(code_vectors, [u1.id, u2.id, u3.id])
        dense = DenseRetriever(encoder, faiss_indexer, metadata_store=store)

        resp_dense = dense.retrieve("parse authorization header token", top_k=2)
        assert resp_dense.method == "dense"
        assert resp_dense.latency_ms > 0
        assert len(resp_dense.results) == 2
        # Semantic search should rank the auth token function as top candidate
        assert resp_dense.results[0].unit_id == "auth.py:extract_bearer_token:1"
        assert resp_dense.results[0].unit is not None
        assert resp_dense.results[0].score > 0.0

