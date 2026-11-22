"""Benchmark data loader and query categorization for evaluation."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """Represents a benchmark query with ground-truth target unit and category."""
    query_id: str
    query_text: str
    target_unit_id: str
    language: str  # "python" or "javascript"
    category: str = Field(
        default="conceptual",
        description="Query category: 'signature_constrained' or 'conceptual'"
    )


# Keywords indicating structural / signature constraints in natural language
SIGNATURE_INDICATORS = {
    "return", "returns", "returning", "param", "params", "parameter",
    "argument", "arguments", "type", "dict", "dictionary", "list",
    "string", "int", "integer", "boolean", "bool", "float", "array",
    "tuple", "takes", "given", "yields", "accepts"
}


def categorize_query(query_text: str) -> str:
    """
    Categorizes a natural language query into:
    1. 'signature_constrained': Query mentions arguments, parameter names, or return types.
    2. 'conceptual': High-level task intent without explicit syntactic or signature constraints.
    """
    tokens = set(query_text.lower().split())
    if tokens.intersection(SIGNATURE_INDICATORS):
        return "signature_constrained"
    return "conceptual"


class BenchmarkDataLoader:
    """Loads and prepares benchmark evaluation datasets."""

    @staticmethod
    def load_jsonl(file_path: str | Path) -> List[BenchmarkQuery]:
        """Loads benchmark queries from a JSONL file."""
        path = Path(file_path)
        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                category = data.get("category") or categorize_query(data["query_text"])
                queries.append(BenchmarkQuery(
                    query_id=data["query_id"],
                    query_text=data["query_text"],
                    target_unit_id=data["target_unit_id"],
                    language=data.get("language", "python"),
                    category=category
                ))
        return queries

    @staticmethod
    def save_jsonl(queries: List[BenchmarkQuery], file_path: str | Path):
        """Saves queries to JSONL format."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps(q.model_dump()) + "\n")
