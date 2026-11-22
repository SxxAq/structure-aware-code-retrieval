"""Unit tests for RepositoryIngester."""

import tempfile
from pathlib import Path
from sahcr.parsers.ingester import RepositoryIngester
from sahcr.indexing.metadata_store import MetadataStore


def test_repository_ingester_directory_walk():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "metadata.sqlite"
        store = MetadataStore(db_path)
        ingester = RepositoryIngester(store)

        # Create sample Python file
        src_dir = root / "src"
        src_dir.mkdir()
        py_file = src_dir / "math_utils.py"
        py_file.write_text("""
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
""", encoding="utf-8")

        # Create sample JS file
        js_file = src_dir / "api.js"
        js_file.write_text("""
function fetchUser(userId) {
    return api.get(`/users/${userId}`);
}
""", encoding="utf-8")

        # Create ignored dir
        ignored_dir = root / "node_modules" / "some-lib"
        ignored_dir.mkdir(parents=True)
        (ignored_dir / "index.js").write_text("function ignored() {}", encoding="utf-8")

        units = ingester.ingest_path(root, save_to_store=True)

        # Should extract 2 Python functions + 1 JS function = 3 functions (ignored dir skipped)
        assert len(units) == 3
        func_names = {u.name for u in units}
        assert "add" in func_names
        assert "multiply" in func_names
        assert "fetchUser" in func_names
        assert "ignored" not in func_names

        # Verify persisted in store
        assert store.count() == 3
        assert store.count("python") == 2
        assert store.count("javascript") == 1
