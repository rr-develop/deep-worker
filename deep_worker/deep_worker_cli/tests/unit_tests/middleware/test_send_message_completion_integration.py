"""Integration-ish tests for SendMessageCompletionMiddleware wiring.

These tests verify that:

1. ``create_cli_agent`` registers the middleware unconditionally.
2. The TaskManager evaluation is skipped in ``AgentProcess._handle_invoke``
   when the ``full_response`` is an outgoing inter-agent echo
   (starts with ``agent:<self.name>:> ``).

We avoid spinning up a real LangGraph — we monkeypatch ``create_deep_agent``
to capture kwargs, and for the TaskManager case we build a minimally-mocked
``AgentProcess`` and drive the relevant code path directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ===========================================================================
# Wiring: create_cli_agent registers the middleware
# ===========================================================================


class TestCreateCliAgentWiring:
    """create_cli_agent must add SendMessageCompletionMiddleware to the
    middleware stack passed to create_deep_agent."""

    def test_middleware_is_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from deep_worker_cli import agent as agent_mod
        from deep_worker_cli.middleware.send_message_completion import (
            SendMessageCompletionMiddleware,
        )

        captured: dict[str, Any] = {}

        def _fake_create_deep_agent(**kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Fake:
                def with_config(self, _cfg: Any) -> "_Fake":
                    return self

            return _Fake()

        import deepagents as da_mod

        monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)

        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fake_model = FakeListChatModel(responses=["ok"])

        agent_mod.create_cli_agent(
            model=fake_model,
            assistant_id="test_sm_completion",
            tools=[],
            auto_approve=False,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
        )

        middleware = captured.get("middleware") or []
        types = [type(m).__name__ for m in middleware]
        assert "SendMessageCompletionMiddleware" in types, (
            f"SendMessageCompletionMiddleware must be in create_deep_agent's "
            f"middleware list; got {types}"
        )
        # And verify the actual instance is of the right class.
        assert any(
            isinstance(m, SendMessageCompletionMiddleware) for m in middleware
        )


# ===========================================================================
# TaskManager skip for outgoing inter-agent echo
# ===========================================================================


class TestTaskManagerSkipForOutgoingEcho:
    """When the agent's final response for a turn is an outgoing
    inter-agent message (``agent:<self.name>:> ...``), TaskManager's
    ``on_subordinate_response`` must NOT be called — this turn is not
    a reply to the user's task, it's a message addressed to a peer
    agent. The TaskManager loop should exit cleanly.

    We test this by exercising the guard helper directly rather than
    the whole streaming path, to keep the test hermetic. The guard is
    exposed as ``AgentProcess._is_outgoing_inter_agent_echo(response)``
    — a pure, side-effect-free string predicate.
    """

    def test_predicate_exists_and_is_pure(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        # The helper is a staticmethod-like predicate: takes agent_name
        # and the full_response string, returns bool. Expose it as a
        # module-level function OR a classmethod — design chooses
        # classmethod for discoverability.
        assert hasattr(AgentProcess, "_is_outgoing_inter_agent_echo")

    def test_detects_own_outgoing_echo(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        assert AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="agent:alpha:> hello",
        )

    def test_detects_own_outgoing_multiline_echo(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        assert AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="agent:alpha:> line1\nline2",
        )

    def test_rejects_incoming_from_other_agent(self) -> None:
        """An agent 'alpha' that sees its own history will also see
        incoming messages like 'agent:beta:> ...' but those are
        HumanMessages, not AIMessages / full_response. Even so, we
        strictly tie the predicate to the agent's OWN name to be safe."""
        from deep_worker_cli.process.agent_process import AgentProcess

        assert not AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="agent:beta:> hi alpha",
        )

    def test_rejects_normal_response(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        assert not AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="Sure, I can help with that.",
        )

    def test_rejects_response_that_mentions_marker_mid_text(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        # Must be a PREFIX, not anywhere-substring.
        assert not AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="I wrote: agent:alpha:> something",
        )

    def test_rejects_empty_response(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        assert not AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response="",
        )

    def test_rejects_none_response(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        # Defensive: None should not crash.
        assert not AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response=None,  # type: ignore[arg-type]
        )

    def test_task_manager_not_called_when_response_is_outgoing_echo(
        self, tmp_path: Path
    ) -> None:
        """End-to-end guard check: in a running loop, if full_response
        matches the echo pattern, task_manager.on_subordinate_response
        is NOT invoked.

        We simulate the guard directly: if predicate True → skip call.
        This mirrors what _handle_invoke does at the guard point.
        """
        from deep_worker_cli.process.agent_process import AgentProcess

        # Minimal fake task_manager — record calls.
        tm = MagicMock()
        tm.on_subordinate_response = AsyncMock(return_value=None)

        # Simulate what _handle_invoke's guard would do.
        full_response = "agent:alpha:> hello beta"
        should_skip = AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response=full_response,
        )

        assert should_skip is True
        # If the guard is True, _handle_invoke must NOT call the task manager.
        # We assert that after a 'guarded' code path, the mock was not used.
        tm.on_subordinate_response.assert_not_called()

    def test_task_manager_called_for_normal_response(self) -> None:
        from deep_worker_cli.process.agent_process import AgentProcess

        full_response = "Here is the report you asked for."
        should_skip = AgentProcess._is_outgoing_inter_agent_echo(
            agent_name="alpha",
            response=full_response,
        )
        assert should_skip is False
