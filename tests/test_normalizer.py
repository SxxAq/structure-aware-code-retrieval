"""Unit tests for token normalization and identifier splitting."""

from sahcr.parsers.normalizer import split_identifier, tokenize_text


def test_split_identifier_snake_case():
    assert split_identifier("parse_auth_token") == ["parse", "auth", "token"]
    assert split_identifier("get_user_by_id_2") == ["get", "user", "by", "id", "2"]


def test_split_identifier_camel_case():
    assert split_identifier("parseAuthToken") == ["parse", "auth", "token"]
    assert split_identifier("getUserById") == ["get", "user", "by", "id"]


def test_split_identifier_pascal_case():
    assert split_identifier("AuthTokenParser") == ["auth", "token", "parser"]
    assert split_identifier("AST2JSONConverter") == ["ast", "2", "json", "converter"]


def test_split_identifier_edge_cases():
    assert split_identifier("") == []
    assert split_identifier("___") == []
    assert split_identifier("test_") == ["test"]


def test_tokenize_text():
    text = "Extract authorization token from HTTP request header."
    tokens = tokenize_text(text)
    # Stop words like 'from' removed, key terms preserved
    assert "extract" in tokens
    assert "authorization" in tokens
    assert "token" in tokens
    assert "http" in tokens
    assert "request" in tokens
    assert "header" in tokens
    assert "from" not in tokens

