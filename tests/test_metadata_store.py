"""Unit tests for MetadataStore SQLite storage."""

import tempfile
from pathlib import Path
from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.indexing.metadata_store import MetadataStore


def test_metadata_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_meta.sqlite"
        store = MetadataStore(db_path)

        unit1 = ExtractedCodeUnit(
            id="test.py:foo:10",
            file_path="test.py",
            language="python",
            name="foo",
            normalized_name_tokens=["foo"],
            parameters=[ParameterInfo(name="x", type_annotation="int", position=0)],
            parameter_tokens=["x", "int"],
            return_annotation="bool",
            return_tokens=["bool"],
            docstring="Does foo things.",
            docstring_tokens=["foo", "things"],
            source_code="def foo(x: int) -> bool:\n    return x > 0",
            start_line=10,
            end_line=11,
            start_byte=100,
            end_byte=140,
            ast_node_types=["return_statement", "binary_operator"],
            is_async=False,
            is_method=False
        )

        unit2 = ExtractedCodeUnit(
            id="test.js:bar:20",
            file_path="test.js",
            language="javascript",
            name="bar",
            normalized_name_tokens=["bar"],
            parameters=[ParameterInfo(name="y", position=0)],
            parameter_tokens=["y"],
            return_annotation=None,
            return_tokens=[],
            docstring=None,
            docstring_tokens=[],
            source_code="function bar(y) { return y * 2; }",
            start_line=20,
            end_line=20,
            start_byte=200,
            end_byte=233,
            ast_node_types=["return_statement"],
            is_async=False,
            is_method=False
        )

        store.save_units([unit1, unit2])

        assert store.count() == 2
        assert store.count("python") == 1
        assert store.count("javascript") == 1

        # Test single retrieval
        retrieved1 = store.get_unit("test.py:foo:10")
        assert retrieved1 is not None
        assert retrieved1.name == "foo"
        assert retrieved1.return_annotation == "bool"
        assert len(retrieved1.parameters) == 1
        assert retrieved1.parameters[0].name == "x"
        assert retrieved1.parameters[0].type_annotation == "int"
        assert "binary_operator" in retrieved1.ast_node_types

        # Test batch retrieval
        batch = store.get_units_by_ids(["test.py:foo:10", "test.js:bar:20", "nonexistent"])
        assert len(batch) == 2
        assert "test.py:foo:10" in batch
        assert "test.js:bar:20" in batch

        # Test iteration
        all_units = list(store.iter_all_units())
        assert len(all_units) == 2

