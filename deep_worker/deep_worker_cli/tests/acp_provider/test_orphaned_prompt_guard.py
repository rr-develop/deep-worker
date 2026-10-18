"""An orphaned ``_active_prompts`` guard must not poison the session.

Defect (rg_linker, 2026-10-04 20:58:47Z and three more clusters that minute,
``~/.deep_worker/logs/rg_linker.err.log`` around 58065-59226):

``AcpConcurrentPromptError: session '01a10848-1023-7b62-a6c2-8ef847677680'
already has an active prompt`` — raised at ``chat_model.py`` step 12 on a
session created ONE SECOND earlier, with no ``session/cancel`` in between.

Mechanism, verified end to end:

1. The guard ``_active_prompts[session_id]`` is released ONLY by the
   ``release_prompt_owner`` done-callback (``acp_client.py``), i.e. when the
   prompt future is cancelled or resolved.
2. ``_consume_turn``'s mid-turn pause deliberately leaves that future IN
   FLIGHT and hands the tool_calls back to LangGraph — so the guard is held
   on purpose while a tool runs.
3. The ``--sum-query`` path runs two sequential turns
   (``agent_process.py``: ``_invoke_turn([])`` then ``_invoke_turn([query])``).
   Between them the history is wiped, and
   ``DanglingToolCallMiddleware.before_model`` drops the now-orphaned
   ``ToolMessage``.
4. Turn 2's mid-turn-resume gate keys off ``tool_resolved`` /
   ``saw_builtin_resume``, and those are set ONLY by finding a ``ToolMessage``
   in the tail. The middleware just removed it, so both are False and the
   branch that would have issued ``session_cancel`` is skipped.
5. Step 12 prompts the still-guarded session and raises.

The fix must be narrow: clear the guard when — and only when — the client
still holds it for this session while the thread state has no local future.
A turn that is legitimately waiting for a tool result HAS its local future,
so it must be left completely alone (reviewer condition).
"""

from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.acp.acp_client import (
    AcpClient,
    AcpConcurrentPromptError,
)


def _client() -> AcpClient:
    """An ``AcpClient`` with real guard bookkeeping and a captured transport."""
    client = object.__new__(AcpClient)
    client._active_prompts = {}
    client._sinks = {}
    client._pending_requests = {}
    client._session_ready = {}
    client._next_id = 0
    client._proc = None
    client._session_close_supported = True
    client.written = []
    client._write = client.written.append  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_guard_is_held_while_a_tool_runs() -> None:
    """Baseline: the mid-turn pause legitimately keeps the guard."""
    client = _client()
    fut = client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )
    assert client.has_active_prompt("s1")
    assert not fut.done()


@pytest.mark.asyncio
async def test_resolved_prompt_releases_the_guard() -> None:
    """A normal turn end releases it through the done-callback."""
    client = _client()
    fut = client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )
    fut.set_result({"result": {}})
    assert not client.has_active_prompt("s1")


@pytest.mark.asyncio
async def test_cancelled_prompt_releases_the_guard() -> None:
    """So does an explicit cancel."""
    client = _client()
    fut = client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )
    fut.cancel()
    assert not client.has_active_prompt("s1")


def test_has_active_prompt_is_false_for_unknown_session() -> None:
    client = _client()
    assert not client.has_active_prompt("nope")
    assert not client.has_active_prompt(None)


@pytest.mark.asyncio
async def test_orphaned_guard_blocks_the_next_prompt() -> None:
    """The defect itself, on the real client.

    "Local reference lost, future still alive" is the exact state the wipe
    leaves behind, and it is what makes step 12 raise.
    """
    client = _client()
    client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )
    # The wipe path drops chat_model's reference without cancelling.
    with pytest.raises(AcpConcurrentPromptError):
        client.session_prompt(
            session_id="s1", prompt=[], notification_sink=asyncio.Queue()
        )


@pytest.mark.asyncio
async def test_session_cancel_clears_an_orphaned_guard() -> None:
    """``session_cancel`` must make the session promptable again."""
    client = _client()
    client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )
    await client.session_cancel("s1")
    assert not client.has_active_prompt("s1")
    assert any(
        w.get("method") == "session/cancel" for w in client.written
    ), "the CLI must be told to abandon the turn, not just local state"
    # Promptable again — this is the line that used to raise.
    client.session_prompt(
        session_id="s1", prompt=[], notification_sink=asyncio.Queue()
    )


# --------------------------------------------------------------------------
# The step-12 preflight in chat_model: fires on the collision shape, and
# NEVER on a legitimate mid-turn resume.
# --------------------------------------------------------------------------


class _State:
    """Minimal stand-in for the fields the preflight reads."""

    def __init__(self, session_id, fut) -> None:
        self.session_id = session_id
        self.pending_session_prompt_future = fut


def _preflight(model, state):
    from deep_worker.providers.acp.chat_model import AcpChatModel

    return AcpChatModel._clear_orphaned_prompt_guard(model, state)


@pytest.mark.asyncio
async def test_preflight_cancels_the_orphaned_guard() -> None:
    """Collision shape: guard held, no local future -> cancel."""
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock()
    model._acp_client.has_active_prompt.return_value = True
    model._acp_client.session_cancel = AsyncMock()
    await _preflight(model, _State("s1", None))
    model._acp_client.session_cancel.assert_awaited_once_with("s1")


@pytest.mark.asyncio
async def test_preflight_leaves_a_legitimate_midturn_resume_alone() -> None:
    """Reviewer condition: a turn awaiting a tool result must NOT be cancelled.

    Here the tool result is still in flight, so the thread state HOLDS its
    future. Cancelling would abort a turn the CLI is legitimately serving.
    """
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock()
    model._acp_client.has_active_prompt.return_value = True
    model._acp_client.session_cancel = AsyncMock()
    loop = asyncio.get_running_loop()
    live = loop.create_future()
    await _preflight(model, _State("s1", live))
    model._acp_client.session_cancel.assert_not_awaited()
    assert not live.done()


@pytest.mark.asyncio
async def test_preflight_is_a_noop_when_no_guard_is_held() -> None:
    """The common path must not emit a spurious session/cancel."""
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock()
    model._acp_client.has_active_prompt.return_value = False
    model._acp_client.session_cancel = AsyncMock()
    await _preflight(model, _State("s1", None))
    model._acp_client.session_cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_is_a_noop_without_a_session() -> None:
    """A brand-new session (step 10 case (a)) has nothing to clear."""
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock()
    model._acp_client.has_active_prompt.return_value = False
    model._acp_client.session_cancel = AsyncMock()
    await _preflight(model, _State(None, None))
    model._acp_client.session_cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_survives_a_failing_cancel() -> None:
    """A wedged CLI must not break the turn we are about to start.

    ``session_cancel`` is best-effort by contract everywhere else; this
    preflight keeps that property, otherwise the recovery path would itself
    become a new way to lose a turn.
    """
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock()
    model._acp_client.has_active_prompt.return_value = True
    model._acp_client.session_cancel = AsyncMock(side_effect=RuntimeError("wedged"))
    await _preflight(model, _State("s1", None))  # must not raise


def test_preflight_is_wired_into_the_prompt_path() -> None:
    """The helper must actually be CALLED before ``session/prompt``.

    Every other test in this file exercises the helper directly, so they all
    keep passing if the single call site disappears — deleting it leaves the
    whole ACP suite green while ``AcpConcurrentPromptError`` comes straight
    back. This pins the other axis: the call exists inside the method that
    prompts, and it runs BEFORE the prompt is fired (clearing the guard after
    the raise would be useless).
    """
    import ast
    from pathlib import Path

    import deep_worker.providers.acp.chat_model as cm

    tree = ast.parse(Path(cm.__file__).read_text())

    def _call_lines(fn: ast.AST, attr: str) -> list[int]:
        return [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ]

    wired: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        guards = _call_lines(node, "_clear_orphaned_prompt_guard")
        prompts = _call_lines(node, "session_prompt")
        if guards and prompts:
            wired.append((min(guards), min(prompts)))

    assert wired, (
        "no method both clears the orphaned prompt guard and calls "
        "session_prompt — the preflight is not wired into the prompt path, "
        "so a leftover guard will raise AcpConcurrentPromptError again"
    )
    for guard_line, prompt_line in wired:
        assert guard_line < prompt_line, (
            f"_clear_orphaned_prompt_guard is called at line {guard_line}, "
            f"after session_prompt at line {prompt_line} — the guard must be "
            "cleared BEFORE the prompt is fired, not after it has raised"
        )
