"""AST structural feature extraction for code units."""

from typing import List, Set, Dict, Any, Optional
from sahcr.models import ExtractedCodeUnit
from sahcr.parsers.normalizer import tokenize_text, split_identifier

# Mapping between natural language intent keywords and AST syntax node types
AST_INTENT_KEYWORD_MAP: Dict[str, Set[str]] = {
    # Iteration & Loops
    "loop": {"for_statement", "while_statement", "for_of_statement", "for_in_statement"},
    "iterate": {"for_statement", "while_statement", "for_of_statement", "for_in_statement"},
    "traverse": {"for_statement", "while_statement", "for_of_statement", "for_in_statement"},
    "each": {"for_statement", "for_of_statement", "for_in_statement"},
    # Exception handling
    "catch": {"try_statement"},
    "error": {"try_statement", "raise_statement", "throw_statement"},
    "exception": {"try_statement", "raise_statement", "throw_statement"},
    "handle": {"try_statement"},
    "raise": {"raise_statement", "throw_statement"},
    "throw": {"raise_statement", "throw_statement"},
    # Async / Concurrency
    "async": {"await_expression", "call"},
    "await": {"await_expression"},
    "promise": {"await_expression", "call_expression"},
    # Data structures & Comprehensions
    "comprehension": {"list_comprehension", "dictionary_comprehension", "set_comprehension"},
    "dict": {"dictionary", "object", "dictionary_comprehension"},
    "dictionary": {"dictionary", "object", "dictionary_comprehension"},
    "list": {"list", "array", "list_comprehension"},
    "array": {"list", "array", "list_comprehension"},
    "json": {"dictionary", "object"},
    # Returns / Generator
    "yield": {"yield", "generator_expression"},
    "generate": {"yield", "generator_expression"},
}

# Type synonyms for return type matching
TYPE_SYNONYMS: Dict[str, Set[str]] = {
    "dict": {"dict", "dictionary", "map", "object", "json"},
    "dictionary": {"dict", "dictionary", "map", "object", "json"},
    "bool": {"bool", "boolean"},
    "boolean": {"bool", "boolean"},
    "int": {"int", "integer", "number"},
    "integer": {"int", "integer", "number"},
    "float": {"float", "double", "number"},
    "str": {"str", "string", "text"},
    "string": {"str", "string", "text"},
    "list": {"list", "array", "sequence"},
    "array": {"list", "array", "sequence"},
    "void": {"none", "void", "null"},
}


class StructuralFeatureExtractor:
    """Extracts explicit AST structural features between query and code unit."""

    def compute_features(self, query: str, unit: ExtractedCodeUnit) -> Dict[str, float]:
        """Computes four structural match scores in [0.0, 1.0]."""
        query_tokens = set(tokenize_text(query, remove_stopwords=True))
        if not query_tokens:
            query_tokens = set(tokenize_text(query, remove_stopwords=False))

        return {
            "phi_identifier": self._compute_identifier_match(query_tokens, unit),
            "phi_parameters": self._compute_parameter_match(query_tokens, unit),
            "phi_return": self._compute_return_match(query_tokens, unit),
            "phi_ast_nodes": self._compute_ast_node_match(query_tokens, unit),
        }

    def _compute_identifier_match(self, query_tokens: Set[str], unit: ExtractedCodeUnit) -> float:
        """Measures lexical and sub-token overlap with function name."""
        if not query_tokens or not unit.normalized_name_tokens:
            return 0.0

        func_tokens = set(unit.normalized_name_tokens)
        intersection = query_tokens.intersection(func_tokens)

        # Jaccard + token coverage bonus
        jaccard = len(intersection) / len(query_tokens.union(func_tokens))
        coverage = len(intersection) / len(func_tokens)

        return min(1.0, 0.5 * jaccard + 0.5 * coverage)

    def _compute_parameter_match(self, query_tokens: Set[str], unit: ExtractedCodeUnit) -> float:
        """Measures overlap between query tokens and formal parameters / annotations."""
        if not query_tokens or not unit.parameters:
            return 0.0

        all_param_tokens: Set[str] = set()
        for p in unit.parameters:
            all_param_tokens.update(split_identifier(p.name))
            if p.type_annotation:
                all_param_tokens.update(tokenize_text(p.type_annotation))

        intersection = query_tokens.intersection(all_param_tokens)
        if not intersection:
            return 0.0

        # Overlap fraction of matching query words
        return min(1.0, len(intersection) / max(1, len(query_tokens)))

    def _compute_return_match(self, query_tokens: Set[str], unit: ExtractedCodeUnit) -> float:
        """Measures whether query indicates expected return type matching unit's return signature."""
        if not query_tokens or not unit.return_tokens:
            return 0.0

        ret_tokens = set(unit.return_tokens)

        # Check exact overlap
        exact_match = query_tokens.intersection(ret_tokens)
        if exact_match:
            return 1.0

        # Check synonym overlap
        for q_token in query_tokens:
            synonyms = TYPE_SYNONYMS.get(q_token, set())
            if synonyms.intersection(ret_tokens):
                return 0.8

        return 0.0

    def _compute_ast_node_match(self, query_tokens: Set[str], unit: ExtractedCodeUnit) -> float:
        """Measures alignment between query intent and structural AST node types in function body."""
        if not query_tokens or not unit.ast_node_types:
            return 0.0

        unit_node_types = set(unit.ast_node_types)
        expected_nodes: Set[str] = set()

        for q_token in query_tokens:
            if q_token in AST_INTENT_KEYWORD_MAP:
                expected_nodes.update(AST_INTENT_KEYWORD_MAP[q_token])

        if not expected_nodes:
            # Query has no explicit structural constraints
            return 0.0

        matched = expected_nodes.intersection(unit_node_types)
        return min(1.0, len(matched) / len(expected_nodes))

