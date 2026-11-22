"""Unit tests for AST structural feature extraction and reranking (Config D)."""

import pytest
from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.reranking.structural_features import StructuralFeatureExtractor


def test_structural_feature_extractor():
    extractor = StructuralFeatureExtractor()

    unit = ExtractedCodeUnit(
        id="crypto.py:sign_data:10",
        file_path="crypto.py",
        language="python",
        name="sign_data",
        normalized_name_tokens=["sign", "data"],
        parameters=[
            ParameterInfo(name="private_key", type_annotation="bytes", position=0),
            ParameterInfo(name="payload", type_annotation="dict", position=1),
        ],
        parameter_tokens=["private", "key", "bytes", "payload", "dict"],
        return_annotation="bytes",
        return_tokens=["bytes"],
        docstring="Digitally sign dictionary payload using RSA private key.",
        docstring_tokens=["digitally", "sign", "dictionary", "payload", "rsa", "private", "key"],
        source_code="def sign_data(private_key: bytes, payload: dict) -> bytes:\n    pass",
        start_line=10,
        end_line=11,
        start_byte=100,
        end_byte=170,
        ast_node_types=["return_statement", "try_statement"],
        is_async=False,
        is_method=False
    )

    # 1. Test identifier match: "sign payload data"
    feat1 = extractor.compute_features("sign payload data", unit)
    assert feat1["phi_identifier"] > 0.4
    assert feat1["phi_parameters"] > 0.0

    # 2. Test parameter match: query specifying parameters "private key and payload"
    feat2 = extractor.compute_features("private key and payload", unit)
    assert feat2["phi_parameters"] > 0.3

    # 3. Test return match: query asking to return bytes
    feat3 = extractor.compute_features("function returning bytes", unit)
    assert feat3["phi_return"] == 1.0

    # 4. Test AST node pattern match: query mentioning error handling (try_statement)
    feat4 = extractor.compute_features("catch and handle sign errors", unit)
    assert feat4["phi_ast_nodes"] > 0.0

