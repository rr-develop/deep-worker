"""Unit tests for SerenaBackend — the contract without a real serena.

See ``docs/DESIGN_LSP_SERENA_BACKEND_ATTACH.md`` §3–§8 and §10.

The MCP session here is fully mocked: the tests check the FACADE CONTRACT
(fail-loud matrix, a single lock, composition with server-side single-flight,
JSON-null skipping, argv, shutdown ordering, identification of the
refresh tool), not serena-agent itself. The real backend is exercised in
``tests/integration_tests/lsp/test_serena_backend_e2e.py``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb
from deep_worker_cli.lsp.service.serena_backend import (
    SerenaBackend,
    build_serena_spawn_argv,
    pick_refresh_tool,
)

pytestmark = pytest.mark.asyncio


# ── MCP test doubles ─────────────────────────────────────────────


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResult:
    def __init__(self, *, text: str = "", is_error: bool = False) -> None:
        self.content = [_FakeContent(text)]
        self.isError = is_error


class _FakeSession:
    """A minimal stand-in for ``mcp.ClientSession``. Fully asynchronous.

    Behavior is driven by attributes:
    - ``responses`` — FIFO queue of call_tool responses.
    - ``raise_on_call`` — if set, call_tool raises this exception.
    - ``sleep_before_reply`` — delay before replying (for serialization
      tests).
    - ``tools_list`` — what list_tools returns.
    """

    def __init__(
        self,
        *,
        tools_list: list[str] | None = None,
        responses: list[_FakeResult] | None = None,
        raise_on_call: BaseException | None = None,
        sleep_before_reply: float = 0.0,
    ) -> None:
        self.initialized = False
        self.list_tools_calls = 0
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.tools_list = tools_list or ["find_symbol"]
        self.responses = list(responses or [_FakeResult(text="{}")])
        self.raise_on_call = raise_on_call
        self.sleep_before_reply = sleep_before_reply
        self.concurrent_calls = 0
        self.peak_concurrent = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> Any:
        self.list_tools_calls += 1
        return SimpleNamespace(
            tools=[SimpleNamespace(name=n) for n in self.tools_list]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeResult:
        self.concurrent_calls += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent_calls)
        try:
            self.call_log.append((name, dict(arguments)))
            if self.raise_on_call is not None:
                raise self.raise_on_call
            if self.sleep_before_reply:
                await asyncio.sleep(self.sleep_before_reply)
            if not self.responses:
                return _FakeResult(text="{}")
            return self.responses.pop(0)
        finally:
            self.concurrent_calls -= 1


def _install_fake(monkeypatch: pytest.MonkeyPatch, session: _FakeSession):
    """Replace the MCP session that SerenaBackend.start() brings up.

    The real ``mcp.client.stdio.stdio_client`` is an asynchronous
    context manager yielding a pair of streams behind which a real
    subprocess lives. Here all of that is replaced by a single fake; the
    SerenaBackend itself does not know — it receives a ready-made session
    through the factory function ``_open_mcp_session``, which we patch.
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _AsyncCM:
        def __init__(self, sess: _FakeSession) -> None:
            self._sess = sess

        async def __aenter__(self) -> _FakeSession:
            self._sess.initialized = True
            return self._sess

        async def __aexit__(self, *exc: object) -> None:
            return None

    async def _fake_open(_argv: list[str]) -> _AsyncCM:
        return _AsyncCM(session)

    monkeypatch.setattr(sb, "_open_mcp_session", _fake_open, raising=True)


# ── argv (A4) ────────────────────────────────────────────────────────


def test_spawn_argv_mirrors_serena_config_contract(tmp_path: Path) -> None:
    """A4: child process arguments — the same contract as
    ``build_serena_mcp_entry``. There is no second CLI dialect.
    """
    from deep_worker_cli.lsp.serena_config import build_serena_mcp_entry

    root = tmp_path / "proj"; root.mkdir()
    argv = build_serena_spawn_argv(root)

    entry = build_serena_mcp_entry(workspace_root=root)
    expected_tail = entry["args"]

    # command goes into argv[0], then the same args in the same order.
    assert argv[0] == entry["command"], argv
    assert argv[1:] == expected_tail, argv
    assert "--project" in argv and str(root.resolve()) in argv


# ── A7: pick_refresh_tool ────────────────────────────────────────────


def test_pick_refresh_tool_prefers_activate_project() -> None:
    """Look up the refresh tool by name rather than hardcoding it — otherwise
    a Serena version change would silently break auto-refresh."""
    assert pick_refresh_tool(["find_symbol", "activate_project"]) == "activate_project"


def test_pick_refresh_tool_falls_back_to_refresh_project() -> None:
    assert pick_refresh_tool(["find_symbol", "refresh_project"]) == "refresh_project"


def test_pick_refresh_tool_returns_none_when_absent() -> None:
    """A7: no refresh tool present → return None instead of guessing.
    ``SerenaBackend.refresh`` must return is_error when None."""
    assert pick_refresh_tool(["find_symbol"]) is None


# ── call_tool contract ───────────────────────────────────────────────


async def test_call_tool_returns_content_from_serena(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _FakeSession(responses=[_FakeResult(text='{"ok": true}')])
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", json.dumps({"name": "X"}))
        assert isinstance(result, pb.ToolResult)
        assert result.is_error is False
        assert result.content_json == '{"ok": true}'
        assert session.call_log == [("find_symbol", {"name": "X"})]
    finally:
        await backend.stop()


async def test_call_tool_before_start_is_fail_loud(
    tmp_path: Path,
) -> None:
    """§5: backend not started → is_error with a non-empty message."""
    backend = SerenaBackend(tmp_path)
    result = await backend.call_tool("FindSymbol", "{}")
    assert result.is_error is True
    assert "not running" in result.error_message.lower()
    assert result.content_json == ""


async def test_call_tool_wraps_transport_error_as_is_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP transport error (broken pipe etc.) — is_error, not an RPC crash."""
    session = _FakeSession(raise_on_call=BrokenPipeError("pipe gone"))
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert "pipe" in result.error_message.lower() or "broken" in result.error_message.lower()
    finally:
        await backend.stop()


async def test_call_tool_propagates_serena_is_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Serena itself returned isError=True — propagate it as is_error."""
    session = _FakeSession(responses=[_FakeResult(text="tool blew up", is_error=True)])
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert "tool blew up" in result.error_message
    finally:
        await backend.stop()


async def test_call_tool_times_out_as_is_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§5: timeout — is_error, not a hung call."""
    session = _FakeSession(sleep_before_reply=2.0)
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path, call_timeout_s=0.1)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert "timed out" in result.error_message.lower() or "timeout" in result.error_message.lower()
    finally:
        await backend.stop()


async def test_call_tool_rejects_unknown_rpc_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§5: unknown RPC — a programming error, not a runtime one; is_error."""
    session = _FakeSession()
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FrobnicateAll", "{}")
        assert result.is_error is True
        assert "FrobnicateAll" in result.error_message
    finally:
        await backend.stop()


async def test_call_tool_drops_json_null_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A8: JSON null arguments are dropped so server-side strict
    optional schemas do not fail."""
    session = _FakeSession()
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        await backend.call_tool(
            "FindSymbol", json.dumps({"name": "X", "kind": None, "container": "y"})
        )
        assert session.call_log == [("find_symbol", {"name": "X", "container": "y"})]
    finally:
        await backend.stop()


# ── §4: single lock, composition with server single-flight ──────────────


async def test_call_tool_serializes_on_single_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§4: two parallel call_tool calls run strictly one after another."""
    session = _FakeSession(sleep_before_reply=0.05)
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        await asyncio.gather(
            backend.call_tool("FindSymbol", "{}"),
            backend.call_tool("FindSymbol", "{}"),
            backend.call_tool("FindSymbol", "{}"),
        )
        assert session.peak_concurrent == 1, (
            f"peak concurrent MCP calls was {session.peak_concurrent}; "
            f"backend lock is not serializing"
        )
    finally:
        await backend.stop()


async def test_refresh_shares_lock_with_call_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§4: refresh and call_tool do not run in parallel — the same lock."""
    session = _FakeSession(
        tools_list=["find_symbol", "activate_project"],
        sleep_before_reply=0.05,
    )
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        await asyncio.gather(
            backend.refresh("test"),
            backend.call_tool("FindSymbol", "{}"),
        )
        assert session.peak_concurrent == 1, (
            f"refresh ran in parallel with call_tool (peak={session.peak_concurrent})"
        )
    finally:
        await backend.stop()


async def test_refresh_reports_error_when_no_refresh_tool_advertised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A7: the tool list contains no refresh — refresh honestly reports
    an error instead of silently doing nothing."""
    session = _FakeSession(tools_list=["find_symbol"])
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await backend.refresh("test")
        assert "refresh" in str(excinfo.value).lower()
    finally:
        await backend.stop()


# ── stop is idempotent ────────────────────────────────────────────────


async def test_stop_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _FakeSession()
    _install_fake(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    await backend.stop()
    await backend.stop()  # must not raise
