"""Factory and entrypoint for language-specific AST parsers."""

from typing import Dict, Optional
from sahcr.parsers.base_parser import BaseParser
from sahcr.parsers.python_parser import PythonParser
from sahcr.parsers.javascript_parser import JavaScriptParser

_PARSER_REGISTRY: Dict[str, BaseParser] = {}


def get_parser(language: str) -> BaseParser:
    """Returns a singleton parser instance for the specified language."""
    lang = language.lower().strip()
    if lang in ("py", "python"):
        if "python" not in _PARSER_REGISTRY:
            _PARSER_REGISTRY["python"] = PythonParser()
        return _PARSER_REGISTRY["python"]
    elif lang in ("js", "javascript"):
        if "javascript" not in _PARSER_REGISTRY:
            _PARSER_REGISTRY["javascript"] = JavaScriptParser()
        return _PARSER_REGISTRY["javascript"]
    else:
        raise ValueError(f"Unsupported language: {language}. Supported: 'python', 'javascript'")

