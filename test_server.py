"""Smoke tests for explainability-report-mcp (static — no runtime deps required)."""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "server.py"


def test_server_parses():
    """server.py must be syntactically valid (importable proxy)."""
    ast.parse(SRC.read_text(encoding="utf-8"))


def test_declares_tools():
    """server.py must register at least one MCP tool."""
    text = SRC.read_text(encoding="utf-8")
    assert "@mcp.tool" in text or ".tool(" in text or "list_tools" in text
