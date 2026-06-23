"""HITL policy for external MCP tools.

Rules (see §3.6 of the design; user rule 2026-06-23):

* ``auto_approve=True`` (caused by ``--skip-permissions / -sp``) →
  EXTERNAL MCP tools must NOT land in interrupt_on. No HITL at all.
* ``auto_approve=False`` → external tools go to HITL by default, EXCEPT
  those names the pool marked as ``auto_approve: true`` per-server.
"""
from __future__ import annotations

from deep_worker_cli.external_mcp.interrupt import (
    build_interrupt_on_with_external,
)


def test_skip_permissions_disables_all_hitl_including_external() -> None:
    # auto_approve=True corresponds to --skip-permissions.
    out = build_interrupt_on_with_external(
        auto_approve=True,
        auto_approve_send_message=False,
        external_tool_names={"mcp__github__create_issue", "mcp__slack__post"},
        external_auto_approved_names=set(),
    )
    # Contract of create_cli_agent: with auto_approve=True interrupt_on={}.
    assert out == {}


def test_no_skip_permissions_puts_external_tools_into_hitl() -> None:
    out = build_interrupt_on_with_external(
        auto_approve=False,
        auto_approve_send_message=False,
        external_tool_names={"mcp__github__create_issue"},
        external_auto_approved_names=set(),
    )
    # Internal tools may also be in out — we only check that the
    # external tool DID land in HITL.
    assert "mcp__github__create_issue" in out


def test_per_server_auto_approve_skips_hitl_even_without_sp() -> None:
    out = build_interrupt_on_with_external(
        auto_approve=False,
        auto_approve_send_message=False,
        external_tool_names={"mcp__t__read", "mcp__t__write"},
        external_auto_approved_names={"mcp__t__read"},
    )
    assert "mcp__t__write" in out
    assert "mcp__t__read" not in out
