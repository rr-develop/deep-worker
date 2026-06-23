"""Thin helper: adds external MCP tools to interrupt_on according to the
design's §3.6 rules.

USER RULE (2026-06-23): with ``-sp`` (skip permissions /
auto_approve=True) NO HITL is needed, including for external
MCP tools. With ``auto_approve=False`` external tools are interrupted BY
DEFAULT, except those marked ``auto_approve: true`` in mcp.json.
"""
from __future__ import annotations

from typing import Any, cast

from deep_worker_cli.agent import _add_interrupt_on


def build_interrupt_on_with_external(
    *,
    auto_approve: bool,
    auto_approve_send_message: bool,
    external_tool_names: set[str],
    external_auto_approved_names: set[str],
) -> dict[str, Any]:
    """Final interrupt_on taking external MCP tools into account.

    create_cli_agent contract:

      * ``auto_approve=True`` → return ``{}``  (NO HITL).
      * otherwise → take the existing ``_add_interrupt_on(...)`` for
        internal tools and add each name from ``external_tool_names``,
        EXCEPT those in ``external_auto_approved_names``.
    """
    if auto_approve:
        return {}

    base = cast(
        "dict[str, Any]",
        dict(_add_interrupt_on(auto_approve_send_message=auto_approve_send_message)),
    )
    # Config for each external tool — the same allowed_decisions as for the
    # other side-effect tools, plus a generic description (the adapter stores
    # the tool's real description in the BaseTool itself — the UI reads it
    # from there).
    generic_cfg: dict[str, Any] = {
        "allowed_decisions": ["approve", "reject"],
        "description": "External MCP tool — review arguments before approving.",
    }
    for name in external_tool_names:
        if name in external_auto_approved_names:
            continue
        base[name] = generic_cfg
    return base
