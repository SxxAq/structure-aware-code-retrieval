"""Repository-level code ingestion and batch AST extraction."""

from pathlib import Path
from typing import List, Optional, Set
import logging

from sahcr.models import ExtractedCodeUnit
from sahcr.parsers import get_parser
from sahcr.indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
}

IGNORED_DIRECTORIES = {
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "dist", "build", ".next", ".cache", "site-packages"
}


class RepositoryIngester:
    """Recursively walks repositories, parses code units, and stores AST metadata."""

    def __init__(self, metadata_store: Optional[MetadataStore] = None):
        self.metadata_store = metadata_store

    def ingest_path(
        self,
        target_path: str | Path,
        save_to_store: bool = True
    ) -> List[ExtractedCodeUnit]:
        path = Path(target_path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {target_path}")

        extracted_units: List[ExtractedCodeUnit] = []

        if path.is_file():
            units = self._ingest_file(path)
            extracted_units.extend(units)
        else:
            for file_path in path.rglob("*"):
                # Skip blacklisted directories
                if any(part in IGNORED_DIRECTORIES for part in file_path.parts):
                    continue
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    units = self._ingest_file(file_path)
                    extracted_units.extend(units)

        if save_to_store and self.metadata_store and extracted_units:
            self.metadata_store.save_units(extracted_units)

        return extracted_units

    def _ingest_file(self, file_path: Path) -> List[ExtractedCodeUnit]:
        ext = file_path.suffix.lower()
        lang = SUPPORTED_EXTENSIONS.get(ext)
        if not lang:
            return []

        try:
            parser = get_parser(lang)
            return parser.parse_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return []
