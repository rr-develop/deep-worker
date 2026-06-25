"""HITL policy for external MCP tools.

Rules (see §3.6 of the design; the user rule of 2026-06-23):

* ``auto_approve=True`` (caused by ``--skip-permissions / -sp``) → EXTERNAL
  MCP tools must NOT get into interrupt_on. No HITL at all.
* ``auto_approve=False`` → external tools go into HITL by default, EXCEPT
  those names that the pool marked as ``auto_approve: true`` per-server.
"""
from __future__ import annotations

from deep_worker_cli.external_mcp.interrupt import (
    build_interrupt_on_with_external,
)


def test_skip_permissions_disables_all_hitl_including_external() -> None:
    # auto_approve=True corresponds to --skip-permissions.
    # Names come from ``pool.auto_approved_tool_names`` = provider-safe
    # (mangled). See docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md.
    out = build_interrupt_on_with_external(
        auto_approve=True,
        auto_approve_send_message=False,
        external_tool_names={"github__create_issue", "slack__post"},
        external_auto_approved_names=set(),
    )
    # The create_cli_agent contract: with auto_approve=True interrupt_on={}.
    assert out == {}


def test_no_skip_permissions_puts_external_tools_into_hitl() -> None:
    out = build_interrupt_on_with_external(
        auto_approve=False,
        auto_approve_send_message=False,
        external_tool_names={"github__create_issue"},
        external_auto_approved_names=set(),
    )
    # internal tools may also be in out — we only check that the
    # external tool DID get into HITL under the provider-safe name.
    assert "github__create_issue" in out


def test_per_server_auto_approve_skips_hitl_even_without_sp() -> None:
    out = build_interrupt_on_with_external(
        auto_approve=False,
        auto_approve_send_message=False,
        external_tool_names={"t__read", "t__write"},
        external_auto_approved_names={"t__read"},
    )
    assert "t__write" in out
    assert "t__read" not in out
