"""Token normalization utilities for identifiers, code, and queries."""

import re
from typing import List

# Common stop words in queries and code tokens
STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "is", "are", "be", "this", "that", "it", "as",
    "self", "cls", "def", "function", "return", "var", "let", "const"
}


def split_identifier(identifier: str) -> List[str]:
    """
    Splits an identifier (camelCase, PascalCase, snake_case, kebab-case)
    into constituent lowercase sub-words.
    Example:
        'parseAuthToken' -> ['parse', 'auth', 'token']
        'get_user_by_id' -> ['get', 'user', 'by', 'id']
        'AST2JSONConverter' -> ['ast', '2', 'json', 'converter']
    """
    if not identifier:
        return []

    # Separate transitions between letters and digits
    s = re.sub(r'([a-zA-Z])([0-9])', r'\1_\2', identifier)
    s = re.sub(r'([0-9])([a-zA-Z])', r'\1_\2', s)
    # Handle consecutive uppercase (e.g. ASTParser -> AST, Parser)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Handle camelCase transitions (e.g. parseAuth -> parse_Auth)
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Split on underscores, hyphens, and non-alphanumerics
    tokens = re.split(r'[^a-zA-Z0-9]+', s)
    
    normalized = []
    for t in tokens:
        t_clean = t.strip().lower()
        if t_clean:
            normalized.append(t_clean)
            
    return normalized


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes natural language or code text into clean normalized tokens.
    """
    if not text:
        return []
    
    # Split words and code identifiers
    raw_tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    result = []
    for token in raw_tokens:
        sub_tokens = split_identifier(token)
        for st in sub_tokens:
            if remove_stopwords and st in STOP_WORDS:
                continue
            if len(st) > 1 or st.isdigit():
                result.append(st)
    return result
