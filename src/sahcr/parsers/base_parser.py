"""Base abstract class for Tree-sitter source code parsers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import tree_sitter

from sahcr.models import ExtractedCodeUnit


class BaseParser(ABC):
    """Abstract parser interface for extracting functions and AST metadata."""

    def __init__(self, language_name: str, language_grammar: tree_sitter.Language):
        self.language_name = language_name
        self.language = language_grammar
        self.parser = tree_sitter.Parser()
        self.parser.language = self.language

    def parse_code(self, source_code: str, file_path: str = "<memory>") -> List[ExtractedCodeUnit]:
        """Parses a string of source code and extracts all function units."""
        source_bytes = source_code.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        return self.extract_units_from_tree(tree, source_bytes, file_path)

    def parse_file(self, file_path: str | Path) -> List[ExtractedCodeUnit]:
        """Reads a file from disk and parses extracted code units."""
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self.parse_code(content, file_path=str(path))
        except Exception as e:
            # Silently skip unreadable files or log error
            return []

    @abstractmethod
    def extract_units_from_tree(
        self,
        tree: tree_sitter.Tree,
        source_bytes: bytes,
        file_path: str
    ) -> List[ExtractedCodeUnit]:
        """Extract functions, parameters, signatures, and AST node types from the parse tree."""
        pass
