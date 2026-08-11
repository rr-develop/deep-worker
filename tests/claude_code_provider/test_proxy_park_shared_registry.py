"""Proxy-MCP parking must share one namespace with ``BoundedToolCallMiddleware``.

Fourth locus of the ``job_id: toolu_…`` defect (observed, 2026-07-31): three
provider proxy-MCP servers park work in their own ``self._jobs`` under the
``call_id`` key and advertise it to the model as ``job_id``. Neither
``wait_for_job`` nor ``kill_job`` resolves that registry — the model gets an
identifier nobody answers to. The defect class is the same one closed three
times before, but on a layer that was not covered then (see
``docs/DESIGN_FIX_SANDBOX_EXECUTE_JOB_ID.md``).

The contract pinned here:

* parking registers an entry in the shared ``ParkedJobRegistry`` — so that
  exactly the same ``wait_for_job``/``kill_job`` see it;
* ``job_id`` in the notice is the canonical ``job_N``, never ``toolu_…``;
* both notice branches (``job_handle`` and ``timeout_notice``) are built
  through the single source of truth ``build_in_progress_notice``, not a
  hand-rolled f-string;
* ``resolve_call(tool_use_id, …)`` still finishes the parked Future
  (dual-key: the local registry holds ``job_N``, entry comes in as ``toolu_…``) —
  critical, otherwise the tool result is lost (§10 of the design: blocking T3).
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Generator
from typing import Any

import pytest
from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)

from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata
from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer
from deepagents.middleware.parked_job_registry import get_parked_job_registry

pytestmark = pytest.mark.asyncio

_JOB_RE = re.compile(r"\bjob_\d+\b")


def _md(name: str, **extra: Any) -> ToolMetadata:
    md: dict[str, Any] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _clean_registry() -> Generator[None, None, None]:  # pyright: ignore[reportUnusedFunction]
    get_parked_job_registry().clear()
    yield
    get_parked_job_registry().clear()


async def _run_until_handle(proxy: ProxyMCPServer, tool: str, call_id: str):
    handler = proxy._create_tool_handler(tool)  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id(call_id)
    try:
        return await asyncio.wait_for(handler({"x": 1}), timeout=2.0)
    finally:
        reset_current_claude_tool_use_id(token)


# ── T1 — parking registers in the SHARED registry ─────────────────────────


async def test_default_job_handle_registers_in_shared_registry() -> None:
    """job-handle branch: the parked Future is visible externally by canonical
    id via ``get_parked_job_registry().resolve(job_id)``. This is exactly the
    point ``wait_for_job``/``kill_job`` read."""
    proxy = ProxyMCPServer([_md("edit_file")], long_running_wait_seconds=0.05)

    result = await _run_until_handle(proxy, "edit_file", "toolu_EDIT")
    text = result["content"][0]["text"]
    match = _JOB_RE.search(text)
    assert match, f"notice must advertise a canonical job_N, got: {text!r}"
    job_id = match.group(0)

    parked = get_parked_job_registry().resolve(job_id)
    assert parked is not None, (
        "parked job is invisible to wait_for_job/kill_job — the exact defect"
    )
    assert parked.tool_name == "edit_file"

    # Cleanup: cancel the Future so pytest doesn't complain about it.
    reg = proxy.get_job_registry()
    for _, entry in reg.items():
        entry["future"].cancel()


async def test_explicit_timeout_notice_registers_in_shared_registry() -> None:
    """The ``timeout_notice`` branch (explicit opt-out): the wording differs,
    but the work is genuinely parked too — it must land in the shared registry
    just like ``job_handle`` (§5 of the design: both branches → one invariant)."""
    proxy = ProxyMCPServer([_md("quick", long_running=False, timeout=0.05)])

    result = await _run_until_handle(proxy, "quick", "toolu_Q")
    text = result["content"][0]["text"]
    match = _JOB_RE.search(text)
    assert match, f"timeout_notice must advertise a canonical job_N: {text!r}"
    job_id = match.group(0)

    assert get_parked_job_registry().resolve(job_id) is not None

    for _, entry in proxy.get_job_registry().items():
        entry["future"].cancel()


# ── T2 — notice text is canonical (no toolu_ leaks; both branches) ────────


@pytest.mark.parametrize(
    "metadata, call_id",
    [
        ({"name": "edit_file"}, "toolu_JH"),                                # job_handle
        ({"name": "quick", "long_running": False, "timeout": 0.05}, "toolu_TN"),  # timeout_notice
    ],
)
async def test_notice_advertises_canonical_job_id_and_not_toolu(
    metadata: dict[str, Any], call_id: str
) -> None:
    """job_id must be job_N; any ``toolu_`` in the text is the defect this
    whole fix closes. Both branches are exercised through one parameter so
    they cannot drift apart (which is what happened historically)."""
    md = _md(**metadata)
    proxy = ProxyMCPServer([md], long_running_wait_seconds=0.05)

    result = await _run_until_handle(proxy, md["name"], call_id)
    text = result["content"][0]["text"]

    assert _JOB_RE.search(text), (
        f"notice must advertise a job_N id — got {text!r}"
    )
    assert call_id not in text, (
        f"notice leaked tool_use_id {call_id!r} — that is the defect this fix "
        f"closes. Full text: {text!r}"
    )

    for _, entry in proxy.get_job_registry().items():
        entry["future"].cancel()


# ── T3 — resolve_call by tool_use_id still finishes the parked Future ─────


async def test_resolve_call_by_tool_use_id_finishes_parked_future() -> None:
    """Reviewer §10, blocking test. After the local key changed to ``job_N``,
    ``resolve_call`` still receives ``toolu_…`` as its input (that is the
    LangChain protocol id). Resolution must find the parking via dual-key
    lookup (alias). Otherwise the parked tool's result is lost — a regression
    worse than the original defect."""
    proxy = ProxyMCPServer([_md("edit_file")], long_running_wait_seconds=0.05)
    call_id = "toolu_RESOLVE_ME"

    # Run the tool_handler and wait until it "goes into parking".
    handler = proxy._create_tool_handler("edit_file")  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id(call_id)
    try:
        task = asyncio.ensure_future(handler({"x": 1}))
        first_result = await asyncio.wait_for(task, timeout=2.0)
    finally:
        reset_current_claude_tool_use_id(token)

    # Confirm the job is genuinely parked.
    assert _JOB_RE.search(first_result["content"][0]["text"])
    reg = proxy.get_job_registry()
    assert reg, "job must be parked"
    # The Future is still alive (work is NOT cancelled — POLICIES §7.1).
    parked_future = next(iter(reg.values()))["future"]
    assert not parked_future.done()

    # Now the SDK delivers the real result via the ORIGINATING tool_use_id.
    from deep_worker.providers.claude_code_provider.custom_types import MCPResult

    payload: MCPResult = {"content": [{"type": "text", "text": "the-real-result"}]}
    proxy.resolve_call(call_id, payload)

    assert parked_future.done(), (
        "resolve_call(tool_use_id) must finish the parked Future via alias — "
        "without this the tool result is silently lost"
    )
    assert parked_future.result() == payload


# ── T4 — AST invariant: exactly ONE producer of the in_progress notice ────


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_no_provider_hand_rolls_the_in_progress_notice() -> None:
    """The single source of truth for the ``job_id:`` string is the shared
    ``build_in_progress_notice``. Any hand-rolled f-string in production
    creates a fifth locus of the same defect. The check is text-based,
    because text divergence was exactly the root cause (§4 of the design).

    Test scope — production code of the providers and middleware. Tests and
    documentation may freely mention the pattern.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    targets = [
        root / "claude_code_provider" / "proxy_mcp_server.py",
        root / "codex_provider" / "proxy_mcp_server.py",
        root / "grok_provider" / "proxy_mcp_server.py",
    ]
    offenders: list[str] = []
    for p in targets:
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        # Locate hand-rolled notice-text production: an f-string
        # containing ``job_id:`` is exactly the root pattern.
        for m in re.finditer(r'f"[^"]*job_id:[^"]*"', src):
            offenders.append(f"{p}: {m.group(0)!r}")
        for m in re.finditer(r"f'[^']*job_id:[^']*'", src):
            offenders.append(f"{p}: {m.group(0)!r}")

    assert not offenders, (
        "in_progress notice must be produced ONLY through "
        "build_in_progress_notice — hand-rolled f-strings found:\n  "
        + "\n  ".join(offenders)
    )


# ── T5 — race: BoundedToolCallMiddleware already minted an id, avoid dup ──


async def test_parking_reuses_existing_alias_when_middleware_already_parked() -> None:
    """Design §3.1: middleware and proxy share the same 120s bond. If the
    middleware already minted a ``job_N`` under the ``toolu_…`` alias, the
    proxy must reuse that same ``job_N`` (not mint a second one)."""
    registry = get_parked_job_registry()
    # Emulate "middleware already parked under the alias":
    prior_id = registry.allocate()
    prior_future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
    registry.register(
        prior_id,
        kind="tool",
        handle=prior_future,
        alias="toolu_RACE",
        tool_name="edit_file",
    )

    proxy = ProxyMCPServer([_md("edit_file")], long_running_wait_seconds=0.05)
    result = await _run_until_handle(proxy, "edit_file", "toolu_RACE")

    text = result["content"][0]["text"]
    match = _JOB_RE.search(text)
    assert match, text
    assert match.group(0) == prior_id, (
        f"proxy must reuse the id middleware already minted ({prior_id!r}), "
        f"got {match.group(0)!r} — duplication reproduces the ambiguity"
    )

    prior_future.cancel()
    for _, entry in proxy.get_job_registry().items():
        entry["future"].cancel()
