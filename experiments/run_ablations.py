"""Automated ablation experiment runner evaluating RQ1, RQ2, and RQ3."""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.index_pipeline import IndexPipeline
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.ast_reranker import ASTRerankedRetriever
from sahcr.evaluation.dataloader import BenchmarkQuery, BenchmarkDataLoader
from sahcr.evaluation.benchmark_runner import BenchmarkRunner


def create_sample_evaluation_corpus(data_dir: Path) -> Tuple[List[ExtractedCodeUnit], List[BenchmarkQuery]]:
    """Creates a controlled evaluation corpus of Python and JS functions with paired benchmark queries."""
    units = [
        # 1. Auth header parser (Python)
        ExtractedCodeUnit(
            id="auth.py:parse_auth_header:1",
            file_path="auth.py",
            language="python",
            name="parse_auth_header",
            normalized_name_tokens=["parse", "auth", "header"],
            parameters=[
                ParameterInfo(name="header", type_annotation="str", position=0),
                ParameterInfo(name="scheme", default_value="'Bearer'", position=1),
            ],
            parameter_tokens=["header", "str", "scheme", "bearer"],
            return_annotation="str",
            return_tokens=["str"],
            docstring="Parse bearer authorization token from request header string.",
            docstring_tokens=["parse", "bearer", "authorization", "token", "request", "header", "string"],
            source_code="def parse_auth_header(header: str, scheme: str = 'Bearer') -> str:\n    return header.split()[1]",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=80,
            ast_node_types=["return_statement", "call"]
        ),
        # 2. Peripheral log function (mentions auth and token, but logs instead of parsing)
        ExtractedCodeUnit(
            id="logger.py:log_token_event:10",
            file_path="logger.py",
            language="python",
            name="log_token_event",
            normalized_name_tokens=["log", "token", "event"],
            parameters=[ParameterInfo(name="token_id", type_annotation="str", position=0)],
            parameter_tokens=["token", "id", "str"],
            return_annotation=None,
            return_tokens=[],
            docstring="Log bearer token authorization event to disk.",
            docstring_tokens=["log", "bearer", "token", "authorization", "event", "disk"],
            source_code="def log_token_event(token_id: str):\n    sys.stdout.write(token_id)",
            start_line=10,
            end_line=11,
            start_byte=100,
            end_byte=170,
            ast_node_types=["call"]
        ),
        # 3. Database connection function (Python)
        ExtractedCodeUnit(
            id="db.py:connect_database:20",
            file_path="db.py",
            language="python",
            name="connect_database",
            normalized_name_tokens=["connect", "database"],
            parameters=[
                ParameterInfo(name="connection_url", type_annotation="str", position=0),
                ParameterInfo(name="timeout", type_annotation="int", default_value="30", position=1),
            ],
            parameter_tokens=["connection", "url", "str", "timeout", "int"],
            return_annotation="Connection",
            return_tokens=["connection"],
            docstring="Establish connection to relational database with timeout.",
            docstring_tokens=["establish", "connection", "relational", "database", "timeout"],
            source_code="def connect_database(connection_url: str, timeout: int = 30):\n    return create_engine(connection_url)",
            start_line=20,
            end_line=21,
            start_byte=200,
            end_byte=290,
            ast_node_types=["return_statement", "call"]
        ),
        # 4. JSON Response Formatter (JavaScript)
        ExtractedCodeUnit(
            id="server.js:formatJsonResponse:1",
            file_path="server.js",
            language="javascript",
            name="formatJsonResponse",
            normalized_name_tokens=["format", "json", "response"],
            parameters=[
                ParameterInfo(name="payload", position=0),
                ParameterInfo(name="statusCode", default_value="200", position=1),
            ],
            parameter_tokens=["payload", "status", "code"],
            return_annotation="object",
            return_tokens=["object"],
            docstring="Format API JSON response object with timestamp.",
            docstring_tokens=["format", "api", "json", "response", "object", "timestamp"],
            source_code="function formatJsonResponse(payload, statusCode = 200) {\n    return { data: payload, status: statusCode, ts: Date.now() };\n}",
            start_line=1,
            end_line=3,
            start_byte=0,
            end_byte=110,
            ast_node_types=["return_statement", "object"]
        ),
        # 5. Retry helper with loops and exceptions (Python)
        ExtractedCodeUnit(
            id="utils.py:retry_operation:30",
            file_path="utils.py",
            language="python",
            name="retry_operation",
            normalized_name_tokens=["retry", "operation"],
            parameters=[
                ParameterInfo(name="fn", position=0),
                ParameterInfo(name="max_attempts", type_annotation="int", default_value="3", position=1),
            ],
            parameter_tokens=["fn", "max", "attempts", "int"],
            return_annotation=None,
            return_tokens=[],
            docstring="Retry an operation loop upon catching exceptions.",
            docstring_tokens=["retry", "operation", "loop", "catching", "exceptions"],
            source_code="def retry_operation(fn, max_attempts: int = 3):\n    for i in range(max_attempts):\n        try:\n            return fn()\n        except Exception:\n            pass\n    raise RuntimeError()",
            start_line=30,
            end_line=36,
            start_byte=300,
            end_byte=450,
            ast_node_types=["for_statement", "try_statement", "return_statement", "raise_statement"]
        ),
    ]

    queries = [
        # Signature-constrained queries (testing RQ2)
        BenchmarkQuery(
            query_id="q1_sig",
            query_text="parse authorization token taking header string and returning string",
            target_unit_id="auth.py:parse_auth_header:1",
            language="python",
            category="signature_constrained"
        ),
        BenchmarkQuery(
            query_id="q2_sig",
            query_text="format API response taking payload and status code returning object",
            target_unit_id="server.js:formatJsonResponse:1",
            language="javascript",
            category="signature_constrained"
        ),
        BenchmarkQuery(
            query_id="q3_sig",
            query_text="retry operation loop catching exception and raising error",
            target_unit_id="utils.py:retry_operation:30",
            language="python",
            category="signature_constrained"
        ),
        # Conceptual task queries (testing RQ2)
        BenchmarkQuery(
            query_id="q4_concept",
            query_text="connect to relational sql database",
            target_unit_id="db.py:connect_database:20",
            language="python",
            category="conceptual"
        ),
        BenchmarkQuery(
            query_id="q5_concept",
            query_text="record authentication security event",
            target_unit_id="logger.py:log_token_event:10",
            language="python",
            category="conceptual"
        ),
    ]

    return units, queries


def run_ablations_experiment(output_dir: str = "experiments/results"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    db_path = out_path / "eval_metadata.sqlite"
    indices_dir = out_path / "indices"

    store = MetadataStore(db_path)
    units, queries = create_sample_evaluation_corpus(out_path)
    store.save_units(units)

    # Build Pipeline
    pipeline = IndexPipeline(indices_dir=indices_dir, pooling="cls")
    pipeline.build_from_units(units)

    # Set up 4 Configurations
    lexical = LexicalRetriever(pipeline.bm25_indexer, metadata_store=store)
    dense = DenseRetriever(pipeline.encoder, pipeline.faiss_indexer, metadata_store=store)
    hybrid = HybridRetriever(lexical, dense, metadata_store=store, rrf_k=60)
    ast_retriever = ASTRerankedRetriever(hybrid, metadata_store=store, alpha=0.6)

    runner = BenchmarkRunner(lexical, dense, hybrid, ast_retriever)

    # 1. Overall Evaluation (RQ1)
    overall_results = runner.run_evaluation(queries, top_k=5)

    # 2. Stratified Evaluation (RQ2)
    sig_queries = [q for q in queries if q.category == "signature_constrained"]
    concept_queries = [q for q in queries if q.category == "conceptual"]

    sig_results = runner.run_evaluation(sig_queries, top_k=5)
    concept_results = runner.run_evaluation(concept_queries, top_k=5)

    # Format report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_queries": len(queries),
        "overall_reports": {name: r.model_dump() for name, r in overall_results["reports"].items()},
        "significance_d_vs_c": overall_results["significance_d_vs_c"].model_dump(),
        "significance_d_vs_b": overall_results["significance_d_vs_b"].model_dump(),
        "stratified_rq2": {
            "signature_constrained": {name: r.model_dump() for name, r in sig_results["reports"].items()},
            "conceptual": {name: r.model_dump() for name, r in concept_results["reports"].items()},
        }
    }

    json_file = out_path / "ablation_metrics.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"✓ Ablation metrics saved to {json_file}")
    return report_dict


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    run_ablations_experiment()
