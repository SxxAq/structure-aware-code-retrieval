"""Unit tests for BM25Indexer."""

import tempfile
from pathlib import Path
from sahcr.models import ExtractedCodeUnit
from sahcr.indexing.bm25_indexer import BM25Indexer


def _create_sample_units():
    u1 = ExtractedCodeUnit(
        id="file.py:parse_jwt:1",
        file_path="file.py",
        language="python",
        name="parse_jwt",
        normalized_name_tokens=["parse", "jwt"],
        parameter_tokens=["token", "secret"],
        return_tokens=["dict"],
        docstring="Decode and parse JWT authorization token.",
        docstring_tokens=["decode", "parse", "jwt", "authorization", "token"],
        source_code="def parse_jwt(token: str, secret: str) -> dict:\n    pass",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=50
    )
    u2 = ExtractedCodeUnit(
        id="file.py:calculate_fib:10",
        file_path="file.py",
        language="python",
        name="calculate_fib",
        normalized_name_tokens=["calculate", "fib"],
        parameter_tokens=["n", "int"],
        return_tokens=["int"],
        docstring="Compute fibonacci number recursively.",
        docstring_tokens=["compute", "fibonacci", "number", "recursively"],
        source_code="def calculate_fib(n: int) -> int:\n    pass",
        start_line=10,
        end_line=11,
        start_byte=100,
        end_byte=140
    )
    u3 = ExtractedCodeUnit(
        id="file.py:save_record:20",
        file_path="file.py",
        language="python",
        name="save_record",
        normalized_name_tokens=["save", "record"],
        parameter_tokens=["db", "row"],
        return_tokens=["bool"],
        docstring="Insert user record into database table.",
        docstring_tokens=["insert", "user", "record", "into", "database", "table"],
        source_code="def save_record(db, row) -> bool:\n    pass",
        start_line=20,
        end_line=21,
        start_byte=200,
        end_byte=240
    )
    return [u1, u2, u3]


def test_bm25_indexing_and_search():
    units = _create_sample_units()
    indexer = BM25Indexer()
    indexer.build_index(units)

    # Search for JWT authorization token
    results = indexer.search("parse authorization token", top_k=5)
    assert len(results) > 0
    top_id, top_score = results[0]
    assert top_id == "file.py:parse_jwt:1"
    assert top_score > 0.0

    # Search for fibonacci computation
    fib_results = indexer.search("compute fibonacci", top_k=5)
    assert len(fib_results) > 0
    assert fib_results[0][0] == "file.py:calculate_fib:10"


def test_bm25_save_and_load():
    units = _create_sample_units()
    indexer = BM25Indexer()
    indexer.build_index(units)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "bm25.pkl"
        indexer.save(save_file)
        assert save_file.exists()

        loaded_indexer = BM25Indexer.load(save_file)
        results = loaded_indexer.search("parse authorization token", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "file.py:parse_jwt:1"

