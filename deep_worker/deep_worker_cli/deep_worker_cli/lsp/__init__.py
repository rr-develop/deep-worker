"""LSP navigation for agents via a single MCP (Serena).

Optional capability — by analogy with the semantic index: if it is not
enabled, nothing in the agent's behavior changes.

The package intentionally does NOT contain its own LSP client and does not
touch External MCP: Serena connects as a regular external stdio MCP server
through the existing ``ExternalMCPClientPool``.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md``.

No side effects on import (the module is imported from the CLI as well).
"""
from __future__ import annotations
