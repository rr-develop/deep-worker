"""Retry after the language server dies (LS terminated).

See ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md``.

Why this exists: the Kotlin LSP hits an upstream Serena bug (the client
capability ``workspace/diagnostic/refresh`` is declared, but the handler
is not registered — see §1.2 of the design). The agent got a raw
``LanguageServerTerminatedException`` and drew two WRONG conclusions from
it: "calls cannot be parallelized" and "absolute paths / broad search
cannot be used". Both are disproved (§1.3).

Pinned here:

* read-only RPCs are retried once and the agent never sees the failure;
* mutating RPCs are NEVER retried (ambiguous commit, §4.2.0);
* a generation-based readiness-wait protocol without lost wakeups (§4.3);
* a hard dispatch budget around BOTH attempts (§4.4, variant A);
* three different agent-facing texts, none containing the raw
  exception.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.grpc_server import (
    LSPServiceServer,
    ServerConfig,
    _MUTATING_RPCS,
    _READ_ONLY_RPCS,
)
from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb
from deep_worker_cli.lsp.service.serena_backend import (
    _RPC_TO_TOOL,
    _DISPATCH_TIMEOUT_MSG,
    _LS_TERMINATED_MSG,
    _LS_TERMINATED_MUTATING_MSG,
    _is_backend_terminated,
)

#: Real text forms observed in live logs
#: under ~/.serena/logs/ (see §1.2 of the design).
_RAW_TERMINATED = (
    "LanguageServerTerminatedException: Language server stdout read "
    "process terminated unexpectedly"
)
_RAW_WRAPPED = (
    "Error processing request textDocument/documentSymbol with params:\n"
    "{'textDocument': {'uri': 'file:///project/app/src/test/java/"
    "BillingDTOTest.kt'}}\n(caused by LanguageServerTerminatedException: "
    "Language server stdout read process terminated unexpectedly)"
)
_RAW_TRANSPORT = "serena find_symbol transport error: [Errno 32] Broken pipe"


def _server(tmp_path: Path) -> LSPServiceServer:
    slug = tmp_path / "slug"
    slug.mkdir(mode=0o700, exist_ok=True)
    return LSPServiceServer(
        ServerConfig(
            socket_path=slug / "grpc.sock",
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            slug=slug.name,
        )
    )


async def _find_symbol(srv: LSPServiceServer) -> pb.ToolResult:
    reply = await srv._servicer.FindSymbol(
        pb.FindSymbolRequest(args_json="{}"), None
    )
    return reply.result


#: RPC → (servicer method, request type). Needed to parametrize over
#: all 11 tools, not only FindSymbol.
_RPC_CALL: dict[str, tuple[str, str]] = {
    "FindSymbol": ("FindSymbol", "FindSymbolRequest"),
    "FindReferencingSymbols": ("FindReferencingSymbols", "RefRequest"),
    "FindDeclaration": ("FindDeclaration", "DeclRequest"),
    "FindImplementations": ("FindImplementations", "ImplRequest"),
    "GetSymbolsOverview": ("GetSymbolsOverview", "OverviewRequest"),
    "GetDiagnosticsForFile": ("GetDiagnosticsForFile", "DiagRequest"),
    "RenameSymbol": ("RenameSymbol", "RenameRequest"),
    "ReplaceSymbolBody": ("ReplaceSymbolBody", "ReplaceRequest"),
    "InsertBeforeSymbol": ("InsertBeforeSymbol", "InsertRequest"),
    "InsertAfterSymbol": ("InsertAfterSymbol", "InsertRequest"),
    "SafeDeleteSymbol": ("SafeDeleteSymbol", "DeleteRequest"),
}


async def _call_rpc(srv: LSPServiceServer, rpc: str) -> pb.ToolResult:
    method_name, request_type = _RPC_CALL[rpc]
    method = getattr(srv._servicer, method_name)
    request = getattr(pb, request_type)(args_json="{}")
    reply = await method(request, None)
    return reply.result


# ── 1-3: basic read-only retry ────────────────────────────────────


@pytest.mark.asyncio
async def test_terminated_result_is_retried_and_succeeds(tmp_path: Path) -> None:
    """First attempt terminated, second OK → the agent sees SUCCESS."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("call")
        if len(calls) == 1:
            return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)
        return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert len(calls) == 2, "a terminated result must be retried"
    assert result.is_error is False
    assert result.content_json == '{"ok":true}'


@pytest.mark.asyncio
async def test_terminated_exception_is_retried(tmp_path: Path) -> None:
    """The handler THROWS a terminated exception → also a retry."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError(_RAW_TERMINATED)
        return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert len(calls) == 2
    assert result.is_error is False


@pytest.mark.asyncio
async def test_retry_uses_the_new_handler(tmp_path: Path) -> None:
    """The key one (MUT-T8): the retry must go to the NEW handler.

    This is exactly what the supervisor does: the old backend is dead, a
    new object is created and ``set_tool_handler`` is re-pointed at its
    ``call_tool``. If dispatch caches the handler before the loop, the
    retry will go to a corpse.
    """
    srv = _server(tmp_path)
    seen: list[str] = []

    async def new_handler(_rpc: str, _args: str) -> pb.ToolResult:
        seen.append("new")
        return pb.ToolResult(content_json='{"from":"new"}')

    async def dead_handler(_rpc: str, _args: str) -> pb.ToolResult:
        seen.append("dead")
        # Simulate the supervisor: swap the handler and drop readiness,
        # as _mark_lost -> mark_backend_not_ready does.
        srv.mark_backend_not_ready()
        srv.set_tool_handler(new_handler)
        srv.mark_backend_ready()
        return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)

    srv.set_tool_handler(dead_handler)
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert seen == ["dead", "new"], f"the retry went the wrong way: {seen}"
    assert result.content_json == '{"from":"new"}'


# ── 4, 5, 5a, 5b: generation-based wait protocol ───────────────────────


@pytest.mark.asyncio
async def test_waits_for_backend_ready_before_retry(tmp_path: Path) -> None:
    """Path (a): the retry does not start until the generation changes."""
    srv = _server(tmp_path)
    order: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        if not order:
            order.append("attempt1")
            srv.mark_backend_not_ready()
            return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)
        order.append("attempt2")
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    async def restore() -> None:
        await asyncio.sleep(0.05)
        order.append("ready")
        srv.mark_backend_ready()

    task = asyncio.create_task(restore())
    result = await _find_symbol(srv)
    await task

    assert order == ["attempt1", "ready", "attempt2"], (
        f"the retry did not wait for readiness: {order}"
    )
    assert result.is_error is False


@pytest.mark.asyncio
async def test_wait_survives_event_replacement(tmp_path: Path) -> None:
    """MUT-T7: replacing the event must not cause a lost wakeup.

    Reproduced experimentally: a waiter parked on E1 never wakes up when
    the attribute is swapped to E2. Therefore ``mark_backend_not_ready``
    must wake the old event before replacing it, and the waiter must
    re-read the current one.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            srv.mark_backend_not_ready()
            return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    async def churn() -> None:
        # One more event replacement AFTER the waiter has parked.
        await asyncio.sleep(0.03)
        srv.mark_backend_not_ready()
        await asyncio.sleep(0.03)
        srv.mark_backend_ready()

    task = asyncio.create_task(churn())
    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    await task

    assert len(calls) == 2
    assert result.is_error is False


@pytest.mark.asyncio
async def test_multiple_replacements_no_lost_wakeup(tmp_path: Path) -> None:
    """Several replacements in a row → the retry is still exactly one."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            srv.mark_backend_not_ready()
            return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    async def churn() -> None:
        for _ in range(3):
            await asyncio.sleep(0.02)
            srv.mark_backend_not_ready()
        await asyncio.sleep(0.02)
        srv.mark_backend_ready()

    task = asyncio.create_task(churn())
    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    await task

    assert len(calls) == 2, f"expected exactly one retry, calls: {len(calls)}"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_same_generation_is_not_treated_as_recovery(
    tmp_path: Path,
) -> None:
    """MUT-T14: a signal event WITHOUT a generation change ≠ recovery.

    This is path (b) of §3.1: Serena is alive, _mark_lost was not called.
    The retry must be immediate, not wait for a generation change that
    will never come.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            # Do NOT touch readiness — Serena is alive, the transport is intact.
            return pb.ToolResult(is_error=True, error_message=_RAW_WRAPPED)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert len(calls) == 2
    assert result.is_error is False


# ── 6, 7: exhaustion → explanatory text ─────────────────────────────


@pytest.mark.asyncio
async def test_exhausted_returns_explanatory_message(tmp_path: Path) -> None:
    """MUT-T3: both attempts terminated → an explanation, NOT raw text."""
    srv = _server(tmp_path)

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert result.is_error is True
    msg = result.error_message
    assert "LanguageServerTerminatedException" not in msg, (
        "the raw exception must not reach the agent"
    )
    # Both of the agent's false hypotheses are explicitly ruled out.
    assert "concurrent" in msg.lower()
    assert "absolute path" in msg.lower()


@pytest.mark.asyncio
async def test_ready_wait_timeout_returns_explanatory_message(
    tmp_path: Path,
) -> None:
    """The backend did not come up within the budget → the same explanatory text."""
    srv = _server(tmp_path)
    srv._ls_terminated_ready_wait_s = 0.1  # test knob

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        srv.mark_backend_not_ready()
        return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert result.is_error is True
    assert "LanguageServerTerminatedException" not in result.error_message
    assert "concurrent" in result.error_message.lower()


# ── 8, 9, 10, 11: strictness, counter, predicate ───────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_text",
    [
        "symbol not found",
        "cancelled (-32800)",
        "content modified (-32801)",
        "relative_path='/etc/passwd' points outside the project root",
    ],
)
async def test_non_terminated_error_is_not_retried(
    tmp_path: Path, error_text: str
) -> None:
    """MUT-T2: non-terminated errors must NOT be retried."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        return pb.ToolResult(is_error=True, error_message=error_text)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert len(calls) == 1, f"{error_text!r} must not be retried"
    assert result.error_message == error_text, (
        "foreign errors must reach the agent without substitution"
    )


@pytest.mark.asyncio
async def test_only_one_retry(tmp_path: Path) -> None:
    """MUT-T5: terminated always → exactly 2 calls, not infinity."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    await _find_symbol(srv)
    assert len(calls) == 2, f"expected exactly 2 attempts, got {len(calls)}"


@pytest.mark.asyncio
async def test_counter_incremented_once_per_retry(tmp_path: Path) -> None:
    """MUT-T6: the counter grows by EXACTLY 1 per fired retry."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 0

    await _find_symbol(srv)

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 1


def test_predicate_is_strict() -> None:
    """MUT-T2: the predicate catches all real forms and nothing extra."""
    assert _is_backend_terminated(_RAW_TERMINATED)
    assert _is_backend_terminated(_RAW_WRAPPED)
    assert _is_backend_terminated(_RAW_TRANSPORT)
    assert _is_backend_terminated("Language server process terminated")
    assert _is_backend_terminated("serena backend not running (call to X)")
    assert _is_backend_terminated("[Errno 32] Broken pipe")

    assert not _is_backend_terminated("")
    assert not _is_backend_terminated("symbol not found")
    assert not _is_backend_terminated("cancelled (-32800)")
    assert not _is_backend_terminated("content modified (-32801)")
    assert not _is_backend_terminated("points outside the project root")


# ── 12, 13: scenarios from live logs ──────────────────────────────────


@pytest.mark.asyncio
async def test_background_death_without_agent_call(tmp_path: Path) -> None:
    """Scenario from live logs: LS died BETWEEN calls.

    No RPC was in flight (Task-8 finished 0.34 s before the crash). So
    there must be neither a retry nor a counter increment, and the next
    call after recovery goes through normally.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    # Background death and recovery — without a single tool call.
    srv.mark_backend_not_ready()
    srv.mark_backend_ready()

    result = await _find_symbol(srv)
    assert len(calls) == 1, "background death must not produce retries"
    assert result.is_error is False

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 0


@pytest.mark.asyncio
async def test_in_flight_tool_at_moment_of_death(tmp_path: Path) -> None:
    """Scenario from live logs: a call IN FLIGHT at the moment of LS death.

    This is exactly the case the agent saw as a raw exception. Now it
    must get a normal result.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            await asyncio.sleep(0.02)  # the call is "in flight"
            srv.mark_backend_not_ready()
            srv.mark_backend_ready()
            return pb.ToolResult(is_error=True, error_message=_RAW_WRAPPED)
        return pb.ToolResult(content_json='{"symbols":[]}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert result.is_error is False
    assert result.content_json == '{"symbols":[]}'


# ── 14-17: mutating RPC (review R1, blocker) ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", sorted(_MUTATING_RPCS))
async def test_mutating_rpc_is_never_replayed_on_terminated_result(
    tmp_path: Path, rpc: str
) -> None:
    """MUT-T9: ambiguous commit → there must be NO replay.

    The edit is written to disk BEFORE the LS is asked for diagnostics
    (serena/tools/symbol_tools.py:638-641 + tools_base.py:474-493), and
    the transport can die after the edit has been applied. A replay would
    duplicate the insertion.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append(_rpc)
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _call_rpc(srv, rpc)
    assert len(calls) == 1, f"{rpc}: a mutating RPC must not be replayed"
    assert result.is_error is True


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", sorted(_MUTATING_RPCS))
async def test_mutating_rpc_is_never_replayed_on_exception(
    tmp_path: Path, rpc: str
) -> None:
    """MUT-T9: the same when the handler THROWS an exception."""
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append(_rpc)
        raise RuntimeError(_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _call_rpc(srv, rpc)
    assert len(calls) == 1, f"{rpc}: an exception is no reason to replay either"
    assert result.is_error is True


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", sorted(_MUTATING_RPCS))
async def test_mutating_rpc_returns_ambiguous_commit_message(
    tmp_path: Path, rpc: str
) -> None:
    """MUT-T10: mutating has its own text — "outcome unknown"."""
    srv = _server(tmp_path)

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await _call_rpc(srv, rpc)
    msg = result.error_message.lower()
    assert "unknown" in msg, "the outcome must be declared unknown"
    assert "not retried" in msg or "not been retried" in msg
    assert "LanguageServerTerminatedException" not in result.error_message
    # Both false hypotheses are ruled out here too.
    assert "concurrent" in msg
    assert "absolute path" in msg


def test_rpc_classification_is_total_and_disjoint() -> None:
    """MUT-T11/T17: the classification covers ALL RPCs and is disjoint.

    Review round 4 (blocker): previously ``_READ_ONLY_RPCS`` was derived
    as ``all - mutating``, so this assertion was a tautology — any new
    RPC automatically became read-only and gained the right to replay.
    Now both sets are explicit whitelists, and an unclassified RPC falls
    into neither.
    """
    all_rpcs = set(_RPC_TO_TOOL)
    assert _READ_ONLY_RPCS | _MUTATING_RPCS == all_rpcs, (
        "every registered RPC must be classified"
    )
    assert _READ_ONLY_RPCS & _MUTATING_RPCS == set(), (
        "an RPC cannot be both read-only and mutating"
    )
    assert _MUTATING_RPCS == {
        "RenameSymbol",
        "ReplaceSymbolBody",
        "InsertBeforeSymbol",
        "InsertAfterSymbol",
        "SafeDeleteSymbol",
    }
    assert _READ_ONLY_RPCS == {
        "FindSymbol",
        "FindReferencingSymbols",
        "FindDeclaration",
        "FindImplementations",
        "GetSymbolsOverview",
        "GetDiagnosticsForFile",
    }


def test_read_only_set_is_not_derived_from_mutating() -> None:
    """Blocker round 4: the whitelists must be INDEPENDENT.

    The check is STATIC, against the source, and that is essential: both
    variants — an explicit literal and a derivation
    ``frozenset(_RPC_TO_TOOL) - _MUTATING_RPCS`` — produce the SAME set
    today, so no runtime assert can tell them apart. The difference will
    show only in the future, when someone adds a tool to
    ``_RPC_TO_TOOL`` without classifying it: in the derived set it would
    silently become read-only and gain the right to replay. The only way
    to prevent this in advance is to require the definition to be a
    literal.
    """
    import ast
    import inspect

    import deep_worker_cli.lsp.service.grpc_server as gs

    tree = ast.parse(inspect.getsource(gs))
    assign = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            if node.target.id == "_READ_ONLY_RPCS":
                assign = node
                break
    assert assign is not None, "_READ_ONLY_RPCS not found in the source"

    # No subtraction/union operations in the definition.
    has_binop = any(
        isinstance(n, ast.BinOp) for n in ast.walk(assign.value)
    )
    assert not has_binop, (
        "_READ_ONLY_RPCS must be an explicit literal, not the result of "
        "a subtraction: a derived set fails open — a new unclassified "
        "RPC would automatically become read-only and "
        "get an auto-replay"
    )
    # And the names are listed directly in the definition.
    literals = {
        n.value for n in ast.walk(assign.value)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert literals == set(_READ_ONLY_RPCS), (
        "the read-only membership must be readable directly from the definition"
    )

    assert not (_READ_ONLY_RPCS & _MUTATING_RPCS)
    assert len(_READ_ONLY_RPCS) == 6


@pytest.mark.asyncio
async def test_unclassified_rpc_is_never_replayed(tmp_path: Path) -> None:
    """Fail closed: an unknown RPC gets no replay.

    We simulate "a tool was added but not classified": dispatch must
    behave as with a mutating one — a single call and an honest
    "outcome unknown", not a silent replay.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append(_rpc)
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    class _Ctx:
        async def abort(self, code, details):  # pragma: no cover
            raise AssertionError(f"unexpected abort {code}: {details}")

    result = await srv._dispatch_tool("BrandNewTool", "{}", _Ctx())
    assert len(calls) == 1, "an unclassified RPC must not be replayed"
    assert result.is_error is True
    assert "unknown" in result.error_message.lower()


# ── 18, 18a, 19, 19a, 19b: dispatch budget (variant A) ────────────────


@pytest.mark.asyncio
async def test_retry_fits_client_deadline(tmp_path: Path) -> None:
    """The total time of an RPC with a retry does not exceed the budget."""
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.5
    srv._ls_terminated_ready_wait_s = 0.1

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        await asyncio.sleep(10)  # deliberately longer than the budget
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    elapsed = loop.time() - start

    assert elapsed <= 2.0, f"RPC exceeded the budget: {elapsed:.2f}s"
    assert result.is_error is True


@pytest.mark.asyncio
async def test_first_attempt_is_capped_by_dispatch_budget(
    tmp_path: Path,
) -> None:
    """MUT-T15 — THE KEY ONE: the cap extends to the FIRST attempt.

    Review round 2: a budget not wrapped around the first attempt
    guarantees nothing — it can hang up to call_timeout_s=120s, by which
    time the bridge's client deadline (120s) has already expired.
    """
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.3
    cancelled = asyncio.Event()

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert cancelled.is_set(), (
        "the first attempt must be cancelled by the shared deadline"
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_no_retry_when_budget_exhausted(tmp_path: Path) -> None:
    """R2a: the budget is consumed by the first attempt → no second attempt."""
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.3
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        await asyncio.sleep(0.5)
        return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert len(calls) == 1, "with no budget left there must be no retry"
    assert result.is_error is True


@pytest.mark.asyncio
async def test_healthy_slow_call_gets_dispatch_timeout_not_terminated(
    tmp_path: Path,
) -> None:
    """MUT-T16: a slow HEALTHY call ≠ LS death.

    Termination was not observed, so issuing _LS_TERMINATED_MSG is
    forbidden — that would be a false diagnosis.
    """
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.3

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        await asyncio.sleep(10)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert result.is_error is True
    msg = result.error_message
    assert "budget" in msg.lower() or "exceeded" in msg.lower()
    assert "did not report a crash" in msg.lower() or "crash" in msg.lower()
    assert msg != _LS_TERMINATED_MSG.format(tool="find_symbol")

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 0, (
        "a budget timeout is not a terminated retry"
    )


@pytest.mark.asyncio
async def test_cancelled_attempt_leaves_backend_usable(tmp_path: Path) -> None:
    """R2b: a budget cancellation does not corrupt the backend.

    call_tool runs under an asyncio.Lock; when CancelledError unwinds,
    the lock must be released, otherwise the next call will
    hang forever.
    """
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.3
    lock = asyncio.Lock()
    fast = {"on": False}

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        async with lock:
            if not fast["on"]:
                await asyncio.sleep(10)
            return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    first = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert first.is_error is True
    assert not lock.locked(), "the lock must be released after cancellation"

    fast["on"] = True
    srv._dispatch_budget_s = 5.0
    second = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert second.is_error is False, "the backend must remain usable"


# ── round 4: defects found at CODE REVIEW ───────────────────────


@pytest.mark.asyncio
async def test_recovery_between_check_and_wait_is_not_missed(
    tmp_path: Path,
) -> None:
    """A race in capturing the generation (MAJOR round 4).

    The supervisor can finish recovery BETWEEN the moment dispatch saw
    ``_backend_ready is False`` and the moment the helper would have read
    the generation. With a local snapshot the helper would capture the
    NEW generation and wait for the next one — that is, a timeout on a
    healthy backend. The generation must be read BEFORE the attempt.
    """
    srv = _server(tmp_path)
    # Not enough: if the generation is read AFTER recovery, the helper will
    # wait for a SECOND raise that will never come and hit the ceiling —
    # the test will see that as a missing retry.
    srv._ls_terminated_ready_wait_s = 0.4
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            # The full death-and-recovery cycle happens INSIDE the first
            # attempt: the generation has already moved, and readiness is
            # off again by the time of the check (the backend "blinked" a
            # second time, but a new one already exists).
            srv.mark_backend_not_ready()
            srv.mark_backend_ready()      # the supervisor raised the replacement
            srv._backend_ready = False    # we have not "seen" this yet
            return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)
        return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    assert len(calls) == 2, (
        "the retry was lost: recovery ALREADY happened (the generation "
        "moved during the attempt), but the generation was read later "
        "and the helper waits for a non-existent second raise"
    )
    assert result.is_error is False


@pytest.mark.asyncio
async def test_stale_ready_flag_does_not_count_as_recovery(
    tmp_path: Path,
) -> None:
    """MUT-T14: compare the GENERATION, not just the readiness flag.

    Scenario: dispatch saw not-ready and started waiting. While it
    waits, readiness returns to True WITHOUT a new backend coming up (the
    flag "blinked" back, the generation did not move — for example, the
    supervisor cancelled the restart). If we wait only for the flag, we
    will decide we have recovered and send the retry to the SAME dead
    handler. We must wait for a generation change.
    """
    srv = _server(tmp_path)
    srv._ls_terminated_ready_wait_s = 0.3
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            srv.mark_backend_not_ready()

            async def flap() -> None:
                # The flag comes back, but there is NO mark_backend_ready
                # (i.e. there is no new backend and the generation does not grow).
                await asyncio.sleep(0.05)
                srv._backend_ready = True
                srv._backend_ready_event.set()

            asyncio.create_task(flap())
            return pb.ToolResult(is_error=True, error_message=_RAW_TRANSPORT)
        return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)

    # No new backend appeared -> the retry is not justified; the agent must
    # get an explanatory text, not a silent replay into a corpse.
    assert len(calls) == 1, (
        "a readiness flag without a generation change is not recovery"
    )
    assert result.is_error is True
    assert "LanguageServerTerminatedException" not in result.error_message


@pytest.mark.asyncio
async def test_retry_exception_aborts_internal(tmp_path: Path) -> None:
    """A non-terminated exception ON THE SECOND attempt ≠ LS death.

    Review round 4: previously such a case fell through to
    ``_LS_TERMINATED_MSG`` — a false diagnosis hiding a real bug. In this
    case the first attempt does abort(INTERNAL); the retry must behave
    the same.
    """
    srv = _server(tmp_path)
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)
        raise ValueError("totally unrelated internal bug")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    aborted: list[tuple] = []

    class _Ctx:
        async def abort(self, code, details):
            aborted.append((code, details))
            raise RuntimeError("aborted")

    with pytest.raises(RuntimeError):
        await srv._dispatch_tool("FindSymbol", "{}", _Ctx())

    assert aborted, "a real exception must go into abort(INTERNAL)"
    code, details = aborted[0]
    assert "INTERNAL" in str(code)
    assert "totally unrelated internal bug" in details
    assert "language server process died" not in details


@pytest.mark.asyncio
async def test_retry_timeout_still_reports_termination(
    tmp_path: Path,
) -> None:
    """A timeout on the second attempt still reports a terminated message.

    The boundary of the previous test: abort is only for REAL exceptions,
    while budget exhaustion remains an explanatory text.
    """
    srv = _server(tmp_path)
    srv._dispatch_budget_s = 0.6
    calls: list[str] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        calls.append("c")
        if len(calls) == 1:
            return pb.ToolResult(is_error=True, error_message=_RAW_TERMINATED)
        await asyncio.sleep(10)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    class _Ctx:
        async def abort(self, code, details):  # pragma: no cover
            raise AssertionError(f"unexpected abort: {details}")

    result = await asyncio.wait_for(
        srv._dispatch_tool("FindSymbol", "{}", _Ctx()), timeout=5.0
    )
    assert result.is_error is True
    assert "LanguageServerTerminatedException" not in result.error_message


@pytest.mark.asyncio
async def test_attach_session_not_released_by_not_ready(
    tmp_path: Path,
) -> None:
    """AttachSession must not declare readiness on a FALSE wakeup.

    Review round 4: ``mark_backend_not_ready`` now wakes the current event
    (so a parked retry picks up the new one). Previously AttachSession
    treated ANY wakeup as "backend ready" and told the bridge "tools can
    be called" — right at the moment when there is no backend.
    """
    srv = _server(tmp_path)
    srv._backend_ready_wait_s = 3.0

    events: list[pb.SessionEvent] = []

    async def consume() -> None:
        stream = srv._servicer.AttachSession(
            pb.AttachRequest(client_id="t"), None
        )
        async for ev in stream:
            events.append(ev)
            break  # we only care about the first event (ATTACHED)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    # A false wakeup: the backend is NOT ready.
    srv.mark_backend_not_ready()
    await asyncio.sleep(0.1)
    assert not events, (
        "ATTACHED must not be emitted on a wakeup from not-ready"
    )

    # And now real readiness.
    srv.mark_backend_ready()
    await asyncio.wait_for(task, timeout=3.0)

    assert events, "after real readiness ATTACHED must arrive"
    assert "backend-not-ready" not in events[0].message, (
        "a ready backend must not be marked as not-ready"
    )


# ── 20: path (b) — live Serena ──────────────────────────────────────


@pytest.mark.asyncio
async def test_live_serena_terminated_retries_same_handler(
    tmp_path: Path,
) -> None:
    """MUT-T13: path (b) — an immediate retry into the SAME handler.

    _mark_lost was not called, readiness never dropped. Nothing to wait
    for here — waiting would eat the whole budget.
    """
    srv = _server(tmp_path)
    srv._ls_terminated_ready_wait_s = 30.0  # deliberately large
    handlers_used: list[int] = []

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        handlers_used.append(id(handler))
        if len(handlers_used) == 1:
            return pb.ToolResult(is_error=True, error_message=_RAW_WRAPPED)
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()

    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await asyncio.wait_for(_find_symbol(srv), timeout=5.0)
    elapsed = loop.time() - start

    assert result.is_error is False
    assert len(handlers_used) == 2
    assert len(set(handlers_used)) == 1, "must be the same handler"
    assert elapsed < 1.0, (
        f"path (b) must not wait for anything, spent {elapsed:.2f}s"
    )
