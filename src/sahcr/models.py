"""Data models representing extracted code entities and metadata."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    """Information about a function parameter extracted from AST."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    position: int = 0


class ExtractedCodeUnit(BaseModel):
    """Normalized representation of an indexed code unit (function/method)."""
    id: str = Field(description="Unique identifier (e.g., filepath:func_name:start_line)")
    file_path: str
    language: str  # "python" or "javascript"
    name: str
    normalized_name_tokens: List[str] = Field(default_factory=list)
    parameters: List[ParameterInfo] = Field(default_factory=list)
    parameter_tokens: List[str] = Field(default_factory=list)
    return_annotation: Optional[str] = None
    return_tokens: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    docstring_tokens: List[str] = Field(default_factory=list)
    source_code: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    ast_node_types: List[str] = Field(
        default_factory=list,
        description="Key AST node types present in function body (e.g. for_statement, try_statement)"
    )
    is_async: bool = False
    is_method: bool = False
    parent_class: Optional[str] = None

    def get_searchable_lexical_text(self) -> str:
        """Constructs representation for lexical index (BM25)."""
        tokens = []
        # Weighted identifier tokens
        tokens.extend(self.normalized_name_tokens * 3)
        # Parameter tokens
        tokens.extend(self.parameter_tokens * 2)
        # Return tokens
        tokens.extend(self.return_tokens * 2)
        # Docstring tokens
        if self.docstring_tokens:
            tokens.extend(self.docstring_tokens * 2)
        # AST structural cues
        tokens.extend([t.replace("_", " ") for t in self.ast_node_types])
        # Source code
        return " ".join(tokens)

