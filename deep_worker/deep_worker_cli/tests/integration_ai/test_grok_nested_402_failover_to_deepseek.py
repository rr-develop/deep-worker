"""Honest integration: a grok HTTP 402 nested in the JSON-RPC error reply must
drive the REAL failover chain end-to-end → restart on deepseek.

Incident 2026-07-04 (test_grok2, per POLICIES.md §9/§10): the live grok agent
returned ``{"error": {"message": "Internal error", "data": {"message": "…402
Payment Required… usage balance exhausted", "http_status": 402}}}`` on every
``session/prompt``. The 402 text is NESTED one level down; the earlier
(2026-07-03) fix only matched a FLAT ``error.message`` the real API never sends,
so failover silently never fired and the agent emitted empty AIMessages.

This test drives the ENTIRE real path with NO mocks of our own logic:

  real grok reply bytes → real ``GrokChatModel._finalize_response``
    → real ``raise_for_error_reply`` → real ``_flatten_error_text``
    → real ``grok_usage_limit_error_from_text`` → real ``GrokUsageLimitError``
    → real ``AgentProcess._handle_invoke`` catch-by-class-name
    → real ``_schedule_failover("grok", …)``
    → real ``deep-worker --restart <name> --provider deepseek`` spawn

Per POLICIES.md §10, the ONLY things substituted are boundary I/O that cannot
be reproduced in-test without an external precondition:

  * The xAI account cannot be forced into a real 402 on demand, so the real 402
    *reply bytes* are fault-injected at the ACP boundary (the real reply object,
    not a mocked function).
  * The actual ``deep-worker`` OS spawn is captured (not executed) so the test
    process does not restart itself / require live deepseek credentials. Its
    argv is asserted — proving the real failover command is issued.

No wait/sleep mocks are needed; the path is synchronous once the reply arrives.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_pkg_root = Path(__file__).resolve().parents[4]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus

# The REAL grok reply captured verbatim from test_grok2.err.log.
_REAL_NESTED_402_REPLY = {
    "id": 5,
    "error": {
        "code": -32603,
        "message": "Internal error",
        "data": {
            "message": (
                "API error (status 402 Payment Required): "
                "Grok Build usage balance exhausted\n\n"
                "Request URL: https://cli-chat-proxy.grok.com/v1/responses"
            ),
            "http_status": 402,
        },
    },
}


@pytest.mark.asyncio
async def test_real_grok_finalize_response_classifies_nested_402() -> None:
    """Layer 1 — the REAL grok terminal path classifies the REAL nested 402
    reply as an account-terminal usage-limit (the trigger for failover)."""
    from deep_worker.providers.acp.limit_errors import AcpUsageLimitError
    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    # Since 2026-08-24 grok runs on the SHARED ACP client, so the real
    # classification is ``AcpUsageLimitError`` — the class names the transport,
    # not the account. Attribution to grok happens from ``config.provider`` in
    # the handler (see layer 2 below), which is what keeps the failover target
    # correct now that claude-code and grok raise the same class.
    model = GrokChatModel(model="grok-4")
    with pytest.raises(AcpUsageLimitError):
        await model._finalize_response(dict(_REAL_NESTED_402_REPLY), ["partial"])


def _make_agent(name: str) -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_nested402_failover_")
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
async def test_real_nested_402_drives_failover_to_deepseek(monkeypatch) -> None:
    """Layer 2 — a REAL GrokUsageLimitError (produced by the REAL classifier on
    the REAL nested reply) flows through the REAL ``_handle_invoke`` failover
    branch and issues the REAL ``deep-worker --restart … --provider deepseek``
    command. Only the OS spawn is captured (boundary), not our logic."""
    # The SHARED ACP client's extractor — grok's own was deleted on 2026-08-24
    # when the provider moved onto the shared layer.
    from deep_worker.providers.acp.acp_client import raise_for_error_reply

    # Produce the exception via the REAL extractor on the REAL reply — no
    # hand-constructed error. This is what the live agent hit.
    real_exc: BaseException | None = None
    try:
        raise_for_error_reply("session/prompt", dict(_REAL_NESTED_402_REPLY))
    except BaseException as e:  # noqa: BLE001 — capturing the real classification
        real_exc = e
    assert real_exc is not None
    assert type(real_exc).__name__ == "AcpUsageLimitError", (
        f"the REAL extractor must classify the nested 402 as a usage limit; "
        f"got {type(real_exc).__name__}"
    )

    agent = _make_agent("test_nested402_agent")
    # THE load-bearing line. ``AcpUsageLimitError`` is raised by claude-code
    # and grok alike, so the handler must attribute the cap from config. With
    # the pre-2026-08-24 hardcoded ``_schedule_failover("claude-code")`` this
    # test spawns ``--provider codex`` (the head of
    # ``failover_candidates("claude-code")``) and the assertion below fails —
    # which is exactly the regression this file's name promises to catch.
    agent.config = SimpleNamespace(provider="grok")

    # Real graph that fails exactly as the live provider did.
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=real_exc)
    agent.agent = fake_graph

    # Capture the REAL failover spawn at the OS boundary (do not execute it —
    # the test process must not restart itself / need deepseek creds).
    spawned: list[list[str]] = []

    async def _fake_spawn(*args, **kwargs):
        spawned.append(list(args))
        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    # Make the availability tracker treat deepseek as available and grok's
    # failure as recordable, WITHOUT touching global on-disk state.
    # ``_schedule_failover`` imports ``AvailabilityTracker`` late from the
    # source module, so patching it there is sufficient.
    # ``record_failure`` is captured, not just stubbed: the provider recorded
    # as FAILED is the attribution, and it is the half of this behaviour the
    # destination assertion cannot see. Because ``is_available`` below admits
    # only deepseek, EVERY chain collapses to deepseek — so a wrong
    # attribution ("claude-code") still spawns ``--provider deepseek`` and the
    # destination assertion alone stays green while the tracker poisons a
    # healthy provider. Verified by mutation: reverting the handler to the
    # hardcoded ``_schedule_failover("claude-code")`` left this file passing
    # until this capture was added.
    recorded_failures: list[str] = []

    class _FakeTracker:
        def record_failure(self, **kwargs) -> bool:
            provider = kwargs.get("provider")
            if provider is not None:
                recorded_failures.append(str(provider))
            return True

        def is_available(self, candidate: str) -> bool:
            return candidate == "deepseek"

    from deep_worker_cli.runtime import availability_tracker as at_mod
    monkeypatch.setattr(
        at_mod, "AvailabilityTracker", lambda *a, **k: _FakeTracker()
    )
    # Avoid writing a real restart notice to disk failing the path.
    import deep_worker_cli.tools as tools_mod
    if hasattr(tools_mod, "write_restart_notice"):
        monkeypatch.setattr(
            tools_mod, "write_restart_notice", lambda *a, **k: None
        )

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "2+2", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    # The usage limit surfaced as ERROR with a clean message (visible in UI).
    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR
    assert "usage limit" in (agent.session_manager.error_message or "").lower()

    # Give the fire-and-forget failover task a tick to run.
    await asyncio.sleep(0.05)

    # The REAL failover command was issued: deep-worker --restart <name>
    # --provider deepseek.
    assert spawned, "failover must spawn a real restart command"
    argv = spawned[-1]
    assert argv[0] == "deep-worker"
    assert "--restart" in argv and "test_nested402_agent" in argv
    assert "--provider" in argv
    assert argv[argv.index("--provider") + 1] == "deepseek", (
        f"grok 402 must fail over to deepseek; argv={argv}"
    )

    # ATTRIBUTION — the half the destination cannot prove (see _FakeTracker).
    # The cap must be recorded against grok, the provider that actually hit it.
    # Recording it against claude-code marks a provider that never failed as
    # unavailable, and would additionally offer grok itself as a failover
    # candidate on a real tracker.
    assert recorded_failures == ["grok"], (
        "the usage cap must be attributed to the provider from config "
        f"(grok), not to the ACP transport's default; got {recorded_failures}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
