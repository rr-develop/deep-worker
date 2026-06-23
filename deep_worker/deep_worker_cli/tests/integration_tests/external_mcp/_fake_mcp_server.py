"""Minimal stdio MCP server used by the integration test.

Exposes exactly one tool, ``add(a, b) -> str``, so the integration test
can verify the full round-trip (list_tools → call_tool → result text).

Run as a child process by ExternalMCPClientPool; speaks stdio MCP via
FastMCP.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-mcp")


@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the result as a string."""
    return str(a + b)


if __name__ == "__main__":
    mcp.run()
