"""Unit tests for the Typer CLI commands."""

from typer.testing import CliRunner
import pytest
from sahcr.interfaces.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Structure-Aware Hybrid Code Retrieval" in result.stdout
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "compare" in result.stdout
    assert "evaluate" in result.stdout


def test_cli_search_without_index_exits():
    result = runner.invoke(app, ["search", "parse token", "--indices-dir", "nonexistent/indices"])
    assert result.exit_code == 1
    assert "Indices not found" in result.stdout
