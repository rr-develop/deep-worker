"""A rate-limit from a config-driven provider must reach the tracker.

Incident 2026-08-17: two agents (both on ``minimax``)
stopped on a 429 and nothing recorded it. ``_schedule_failover`` was wired
for exactly four built-in providers, and minimax's cap arrives as a plain
``openai.RateLimitError`` — so it matched no branch, the tracker learned
nothing, and no failover was attempted.

The rule under test is deliberately provider-AGNOSTIC. Config-driven
providers are added by editing ``providers.json`` with no code change
(owner directive 2026-08-15), so a rule that only knew about "minimax"
would leave the next vendor in exactly the same hole. Hence the parametrized
provider names below: the fix must not contain the string "minimax"
anywhere in its control flow, and the provider identity must come from
``self.config.provider``, never from parsing the error text.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class FakeRateLimitError(Exception):
    """Shaped like ``openai.RateLimitError`` for classification purposes."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


#: The verbatim minimax cap message, from an agent log (2026-08-17).
MINIMAX_MARKER = (
    "Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', "
    "'message': 'Token Plan usage limit reached: Upgrade your Token Plan or "
    "purchase Credits for more usage. (2056)'}}"
)


def _make_agent(provider: str) -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_cfgdriven_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_config_driven_limit"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    agent.config = MagicMock()
    agent.config.provider = provider
    agent.launch_config = {"provider": provider, "preferred_provider": provider}
    return agent


async def _invoke_with_limit(agent: AgentProcess, error: Exception) -> dict:
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=error)
    agent.agent = fake_graph
    return await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )


class TestConfigDrivenRateLimitReachesFailover:

    @pytest.mark.parametrize(
        "provider",
        ["minimax", "some-new-vendor", "another-openai-compatible"],
        ids=["minimax", "unknown-vendor-1", "unknown-vendor-2"],
    )
    @pytest.mark.asyncio
    async def test_rate_limit_is_recorded_for_any_config_driven_provider(
        self, provider
    ):
        """The rule keys off the ACTIVE provider, not the error text."""
        agent = _make_agent(provider)
        agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

        await _invoke_with_limit(agent, FakeRateLimitError(MINIMAX_MARKER))

        agent._schedule_failover.assert_awaited_once()
        called_provider = agent._schedule_failover.await_args.args[0]
        assert called_provider == provider, (
            "the provider must come from self.config.provider; a rule that "
            "sniffs the message would mis-attribute every other vendor"
        )

    @pytest.mark.asyncio
    async def test_the_error_surfaces_as_a_limit_not_a_silent_idle(self):
        """A cap is an ERROR state — a lesson from an earlier incident."""
        agent = _make_agent("minimax")
        agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

        result = await _invoke_with_limit(agent, FakeRateLimitError(MINIMAX_MARKER))

        assert result["success"] is False
        assert agent.session_manager.status == AgentStatus.ERROR
        assert "limit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_builtin_providers_keep_their_own_branch(self):
        """No double-recording: built-ins already have typed branches."""
        agent = _make_agent("claude-code")
        agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

        # A generic 429 on a BUILT-IN provider must not be re-routed by the
        # new config-driven rule (its typed branch owns that decision).
        await _invoke_with_limit(agent, FakeRateLimitError("some 429"))

        if agent._schedule_failover.await_count:
            assert agent._schedule_failover.await_args.args[0] == "claude-code"

    @pytest.mark.asyncio
    async def test_a_non_limit_error_does_not_record_a_limit(self):
        """Only rate limits count — an ordinary failure must not fence a
        provider off for hours."""
        agent = _make_agent("minimax")
        agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

        await _invoke_with_limit(agent, ValueError("bad request, not a cap"))

        agent._schedule_failover.assert_not_awaited()


class TestNoProviderNameHardcoded:
    """MUST-HAVE #2/#3: the fix may not name a vendor in its control flow."""

    def test_agent_process_does_not_hardcode_minimax(self):
        import deep_worker_cli.process.agent_process as ap

        assert ap.__file__ is not None
        text = Path(ap.__file__).read_text()
        assert "minimax" not in text.lower(), (
            "the 3h fallback and the 429 rule must be provider-agnostic; "
            "the only place a vendor is named is "
            "_PROVIDER_FALLBACK_HOURS in availability_tracker.py"
        )
