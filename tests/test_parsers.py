"""Unit tests for Python and JavaScript Tree-sitter parsers."""

import pytest
from sahcr.parsers.python_parser import PythonParser
from sahcr.parsers.javascript_parser import JavaScriptParser
from sahcr.parsers import get_parser


def test_python_parser_function_extraction():
    parser = PythonParser()
    py_code = '''
def calculate_metrics(y_true: list[float], y_pred: list[float], threshold: float = 0.5) -> dict:
    """Calculate evaluation metrics given ground truth and predictions."""
    results = {}
    for i in range(len(y_true)):
        if y_pred[i] >= threshold:
            results[i] = True
    return results

class Evaluator:
    def evaluate(self, model, dataset):
        """Run evaluation loop."""
        try:
            return model.run(dataset)
        except Exception as e:
            raise e
'''
    units = parser.parse_code(py_code, file_path="test_eval.py")
    assert len(units) == 2

    # Check function 1: calculate_metrics
    u1 = units[0]
    assert u1.name == "calculate_metrics"
    assert u1.normalized_name_tokens == ["calculate", "metrics"]
    assert len(u1.parameters) == 3
    assert u1.parameters[0].name == "y_true"
    assert u1.parameters[0].type_annotation == "list[float]"
    assert u1.parameters[2].name == "threshold"
    assert u1.parameters[2].default_value == "0.5"
    assert u1.return_annotation == "dict"
    assert "dict" in u1.return_tokens
    assert u1.docstring == "Calculate evaluation metrics given ground truth and predictions."
    assert "calculate" in u1.docstring_tokens
    assert "for_statement" in u1.ast_node_types
    assert "if_statement" in u1.ast_node_types
    assert "return_statement" in u1.ast_node_types
    assert not u1.is_method
    assert u1.parent_class is None

    # Check function 2: Evaluator.evaluate
    u2 = units[1]
    assert u2.name == "evaluate"
    assert u2.is_method
    assert u2.parent_class == "Evaluator"
    assert "try_statement" in u2.ast_node_types
    assert "raise_statement" in u2.ast_node_types


def test_python_parser_async():
    parser = PythonParser()
    py_code = '''
async def fetch_remote_data(url: str) -> bytes:
    """Fetch data asynchronously."""
    async with aiohttp.ClientSession() as session:
        return await session.get(url)
'''
    units = parser.parse_code(py_code, file_path="client.py")
    assert len(units) == 1
    assert units[0].name == "fetch_remote_data"
    assert units[0].is_async


def test_javascript_parser_functions():
    parser = JavaScriptParser()
    js_code = '''
/**
 * Parse bearer authorization token from request headers
 */
function parseAuthToken(req, defaultToken = null) {
    if (!req.headers) {
        return defaultToken;
    }
    const authHeader = req.headers['authorization'];
    return authHeader ? authHeader.split(' ')[1] : defaultToken;
}

const formatResponse = (data, ...tags) => {
    return { data, tags, timestamp: Date.now() };
};

class AuthController {
    async handleLogin(req, res) {
        try {
            const token = await this.authService.authenticate(req.body);
            return res.json({ token });
        } catch (err) {
            throw err;
        }
    }
}
'''
    units = parser.parse_code(js_code, file_path="server.js")
    assert len(units) == 3

    # Function 1: parseAuthToken
    u1 = units[0]
    assert u1.name == "parseAuthToken"
    assert u1.normalized_name_tokens == ["parse", "auth", "token"]
    assert len(u1.parameters) == 2
    assert u1.parameters[0].name == "req"
    assert u1.parameters[1].name == "defaultToken"
    assert u1.docstring is not None
    assert "bearer" in u1.docstring.lower()
    assert "if_statement" in u1.ast_node_types
    assert "return_statement" in u1.ast_node_types

    # Function 2: formatResponse (arrow function)
    u2 = units[1]
    assert u2.name == "formatResponse"
    assert len(u2.parameters) == 2

    # Function 3: handleLogin (class method)
    u3 = units[2]
    assert u3.name == "handleLogin"
    assert u3.is_method
    assert u3.parent_class == "AuthController"
    assert u3.is_async
    assert "try_statement" in u3.ast_node_types
    assert "throw_statement" in u3.ast_node_types


def test_parser_factory():
    py_parser = get_parser("python")
    assert isinstance(py_parser, PythonParser)
    js_parser = get_parser("javascript")
    assert isinstance(js_parser, JavaScriptParser)

    with pytest.raises(ValueError):
        get_parser("ruby")

