"""End-to-end integration test comparing Configurations A, B, C, and D."""

import tempfile
from pathlib import Path
import pytest

from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.ast_reranker import ASTRerankedRetriever
from sahcr.evaluation.benchmark_runner import BenchmarkRunner
from sahcr.evaluation.dataloader import BenchmarkQuery


@pytest.fixture(scope="module")
def encoder():
    return CodeBERTEncoder(pooling="cls")


def test_four_configurations_e2e(encoder):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "e2e_meta.sqlite"
        store = MetadataStore(db_path)

        # 1. Target function: parse_auth_header
        u1 = ExtractedCodeUnit(
            id="auth.py:parse_auth_header:1",
            file_path="auth.py",
            language="python",
            name="parse_auth_header",
            normalized_name_tokens=["parse", "auth", "header"],
            parameters=[
                ParameterInfo(name="header_val", type_annotation="str", position=0),
                ParameterInfo(name="prefix", default_value="'Bearer'", position=1)
            ],
            parameter_tokens=["header", "val", "str", "prefix", "bearer"],
            return_annotation="str",
            return_tokens=["str"],
            docstring="Parse bearer authorization token from header value.",
            docstring_tokens=["parse", "bearer", "authorization", "token", "header", "value"],
            source_code="def parse_auth_header(header_val: str, prefix='Bearer') -> str:\n    return header_val.split()[1]",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=80,
            ast_node_types=["return_statement", "call"]
        )

        # 2. Peripheral function (similar vocabulary in comments/docstring but different signature)
        u2 = ExtractedCodeUnit(
            id="logs.py:log_auth_failure:10",
            file_path="logs.py",
            language="python",
            name="log_auth_failure",
            normalized_name_tokens=["log", "auth", "failure"],
            parameters=[ParameterInfo(name="event", type_annotation="dict", position=0)],
            parameter_tokens=["event", "dict"],
            return_annotation=None,
            return_tokens=[],
            docstring="Log failed authorization attempt with token details.",
            docstring_tokens=["log", "failed", "authorization", "attempt", "token", "details"],
            source_code="def log_auth_failure(event: dict):\n    logger.error('Auth failure', extra=event)",
            start_line=10,
            end_line=11,
            start_byte=100,
            end_byte=170,
            ast_node_types=["call"]
        )

        # 3. Completely unrelated function
        u3 = ExtractedCodeUnit(
            id="math.py:matrix_dot:20",
            file_path="math.py",
            language="python",
            name="matrix_dot",
            normalized_name_tokens=["matrix", "dot"],
            parameters=[
                ParameterInfo(name="a", type_annotation="list", position=0),
                ParameterInfo(name="b", type_annotation="list", position=1)
            ],
            parameter_tokens=["a", "list", "b", "list"],
            return_annotation="list",
            return_tokens=["list"],
            docstring="Calculate dot product between matrices.",
            docstring_tokens=["calculate", "dot", "product", "matrices"],
            source_code="def matrix_dot(a: list, b: list) -> list:\n    return np.dot(a, b)",
            start_line=20,
            end_line=21,
            start_byte=200,
            end_byte=260,
            ast_node_types=["return_statement", "call"]
        )

        units = [u1, u2, u3]
        store.save_units(units)

        # Build BM25
        bm25 = BM25Indexer()
        bm25.build_index(units)
        lexical_retriever = LexicalRetriever(bm25, metadata_store=store)

        # Build FAISS
        faiss_idx = FAISSIndexer(dimension=768, index_type="flat")
        dense_texts = [u.get_dense_representation_text() for u in units]
        vectors = encoder.encode_texts(dense_texts)
        faiss_idx.add_vectors(vectors, [u.id for u in units])
        dense_retriever = DenseRetriever(encoder, faiss_idx, metadata_store=store)

        # Build Hybrid (Config C)
        hybrid_retriever = HybridRetriever(lexical_retriever, dense_retriever, metadata_store=store, rrf_k=60)

        # Build AST Reranked (Config D)
        ast_retriever = ASTRerankedRetriever(hybrid_retriever, metadata_store=store, alpha=0.5)

        # Test Benchmark Runner
        runner = BenchmarkRunner(lexical_retriever, dense_retriever, hybrid_retriever, ast_retriever)
        queries = [
            BenchmarkQuery(
                query_id="q1",
                query_text="parse authorization token header with prefix parameter returning string",
                target_unit_id="auth.py:parse_auth_header:1",
                language="python",
                category="signature_constrained"
            )
        ]

        results = runner.run_evaluation(queries, top_k=3)
        reports = results["reports"]

        # Verify all 4 configs evaluated successfully
        assert "Config A (BM25)" in reports
        assert "Config B (CodeBERT)" in reports
        assert "Config C (Hybrid RRF)" in reports
        assert "Config D (Hybrid + AST)" in reports

        # Config D should successfully find target at rank 1 with Recall@1 = 1.0 and MRR@10 = 1.0
        assert reports["Config D (Hybrid + AST)"].recall_at_1 == 1.0
        assert reports["Config D (Hybrid + AST)"].mrr_at_10 == 1.0
