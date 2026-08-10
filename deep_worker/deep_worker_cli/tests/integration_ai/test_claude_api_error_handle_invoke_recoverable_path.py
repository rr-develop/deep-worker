"""Layer-2 test: ``_handle_invoke`` taxonomy + notify on ClaudeApiResponseError.

This is NOT the signed §5.2 end-to-end test. The full §5.2 contract (real
child agent process, real SDK event capture, real provider->graph boundary,
real history/checkpointer assertions, mutation RED) is implemented in
``test_claude_api_error_e2e_subprocess.py``. This file deliberately covers a
narrower, cheaper slice and is kept only for fast feedback on the
``_handle_invoke`` classification branch.

Scope actually exercised here:

  real recorded API-error payload shape
    -> real ``api_error_detect.detect_api_error`` (§3.2)
    -> real ``ClaudeApiResponseError`` (§3.1)
    -> real ``AgentProcess._handle_invoke`` catch-all
    -> real ``is_provider_api_error`` registry lookup
    -> recoverable branch selected: notify type="error", status back to IDLE.

Explicitly NOT covered by this file (see the subprocess E2E instead):

  * ``ClaudeCodeChatModel.ainvoke`` — the exception is fault-injected at the
    ``graph.ainvoke`` boundary, so SDK event capture, ``returning_tool_calls``
    and detector wiring inside the model are all bypassed. That is precisely
    where the production defect lived.
  * Real history persistence — ``_append_recoverable_error_to_history`` is an
    ``AsyncMock`` here, so no SystemMessage is really written and no
    checkpointer is touched. Only the fact that the recoverable branch calls
    it is asserted.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_pkg_root = Path(__file__).resolve().parents[4]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus

# Deselected by default (pyproject addopts `-m 'not integration'`); run with
# `pytest -m integration` or by pointing at the file. Same convention as the
# other tests in this suite.
pytestmark = pytest.mark.integration


# VERBATIM payload from a recorded API-error transcript (session transcript
# jsonl line 523): single TextBlock beginning
# with "API Error:", stop_reason="end_turn", ALL-ZERO usage, NO
# isApiErrorMessage flag (absent in 8 of 9 recorded cases in that transcript).
# Reproducing it verbatim is required by POLICIES.md §10 — the earlier grok
# 402 fix passed a FLAT payload the real API never sends and shipped broken.
_REAL_400_TEXT = (
    "API Error: 400 messages.18.content.0.tool_result.content.1: "
    "Input tag 'image_url' found using 'type' does not match any of the "
    "expected tags: 'document', 'image', 'search_result', 'text', "
    "'tool_reference'"
)
_REAL_ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def _make_agent(name: str) -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_api_error_recoverable_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = name
    agent.data_dir = Path(tmp)
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    agent.thread_id = "default"
    agent._append_recoverable_error_to_history = AsyncMock()  # type: ignore[method-assign]
    return agent


@pytest.mark.asyncio
async def test_real_verbatim_payload_becomes_recoverable_error() -> None:
    """Layer 1: the REAL detector, on the REAL recorded API-error payload
    shape, produces a ``ClaudeApiResponseError`` — no hand-constructed
    exception."""
    from claude_code_provider.api_error_detect import detect_api_error
    from claude_code_provider.chat_model import ClaudeApiResponseError

    detected = detect_api_error(
        response_text=_REAL_400_TEXT,
        assistant_error=None,
        result_is_error=False,      # NOT set on this shape (§1.2b)
        result_subtype=None,
        result_errors=(),
        result_text=None,
        stop_reason="end_turn",     # verbatim
        usage=_REAL_ZERO_USAGE,     # KNOWN_ZERO (§3.2 rule 4 trigger)
    )
    assert detected is not None, (
        "verbatim recorded API-error payload must be detected via rule 4 "
        "(marker + KNOWN_ZERO)"
    )
    exc = ClaudeApiResponseError(
        detected.message,
        assistant_error=detected.assistant_error,
        result_subtype=detected.result_subtype,
        result_errors=detected.result_errors,
        stop_reason=detected.stop_reason,
    )
    assert str(exc) == _REAL_400_TEXT
    assert exc.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_real_api_error_flows_through_recoverable_path() -> None:
    """Layer 2: REAL ClaudeApiResponseError -> REAL _handle_invoke catch-all
    -> REAL is_provider_api_error registry lookup -> REAL recoverable branch
    (notify type='error', history append, IDLE, success=False)."""
    from claude_code_provider.api_error_detect import detect_api_error
    from claude_code_provider.chat_model import ClaudeApiResponseError

    detected = detect_api_error(
        response_text=_REAL_400_TEXT,
        assistant_error=None,
        result_is_error=False,
        result_subtype=None,
        result_errors=(),
        result_text=None,
        stop_reason="end_turn",
        usage=_REAL_ZERO_USAGE,
    )
    assert detected is not None
    real_exc = ClaudeApiResponseError(
        detected.message,
        assistant_error=detected.assistant_error,
        result_subtype=detected.result_subtype,
        result_errors=detected.result_errors,
        stop_reason=detected.stop_reason,
    )

    agent = _make_agent("test_claude_api_error_recoverable")
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=real_exc)
    agent.agent = fake_graph

    notifications: list[dict] = []

    async def _notify(payload: dict) -> None:
        notifications.append(payload)

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "quick user request", "stream": False},
        connection=None,
        notify_callback=_notify,
        approval_callback=None,
    )

    # G3: agent stays IDLE, invoke returns success=False.
    assert result["success"] is False
    assert agent.session_manager.status != AgentStatus.ERROR, (
        f"ClaudeApiResponseError is recoverable — status must NOT be ERROR; "
        f"got {agent.session_manager.status}"
    )

    # G2: persistent type="error" notification carrying real API text.
    error_notifs = [n for n in notifications if n.get("type") == "error"]
    assert error_notifs, (
        f"POLICIES §9: recoverable provider API error must surface as visible "
        f"persistent type='error' notification. Got: {notifications!r}"
    )
    n = error_notifs[-1]
    assert n.get("role") == "system"
    content = n.get("content") or ""
    assert _REAL_400_TEXT in content, (
        f"notification must carry the REAL upstream API text; got: {content!r}"
    )
    assert "ClaudeApiResponseError" in content, (
        "notification content must name the exception class (agent_process "
        "formats as 'Provider API error (<Class>): ...')"
    )

    # G4 (guaranteed): SystemMessage appended to history.
    assert agent._append_recoverable_error_to_history.await_count == 1
    call_kwargs = agent._append_recoverable_error_to_history.await_args.kwargs
    assert call_kwargs.get("exc_name") == "ClaudeApiResponseError"
    assert _REAL_400_TEXT in (call_kwargs.get("exc_repr") or "")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
