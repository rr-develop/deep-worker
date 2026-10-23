"""The provider runtime preflight is PER-PROVIDER, and it runs at agent
startup rather than at the first ``ainvoke``.

Two separate claims, both regressions that were real:

1. **Scope.** The node >= 22 gate belongs to ``claude-code`` alone — it is
   the ``engines`` requirement of its ``@agentclientprotocol/claude-agent-acp``
   adapter, not a deep-worker-wide dependency. An agent on an API provider
   (``openai``, ``anthropic``, a ``providers.json`` entry, …) talks HTTP and
   must start on a host with no node installed at all. A blanket "check node
   before starting any agent" would invent a dependency those providers do
   not have, so the tests below assert the NEGATIVE arm too: the API-provider
   path must not probe node even when node is absent/broken.

2. **Timing.** ``preflight_async`` used to be reached only from
   ``ClaudeCodeAcpChatModel._ensure_acp_started``, i.e. at the first real
   turn. ``deep-worker <name>`` therefore printed a healthy start for an
   agent that could never answer, and the user discovered the broken adapter
   only after sending a prompt. It now runs inside ``AgentProcess.start()``'s
   init ``try``, so the failure becomes ``AgentStatus.ERROR`` + an "error"
   report on the readiness channel (``deep-worker <name>`` prints the cause
   and exits nonzero — see ``daemon._wait_for_readiness``).

Both arms are asserted because an oracle that only checks the claude-code
side would pass just as happily on a blanket implementation that probed node
for everyone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_DW_CLI_ROOT = Path(__file__).resolve().parents[2]
if str(_DW_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DW_CLI_ROOT))

from deep_worker_cli import config as cfg  # noqa: E402


#: Providers that reach an external runtime and therefore MUST be probed.
_RUNTIME = ["claude-code"]

#: Providers that are pure HTTP/API and therefore MUST NOT be probed. Covers
#: the built-in chain and the config-declared (``providers.json``) dialects,
#: including ``azure`` — adding a dialect must not quietly add a node
#: dependency to it.
_API_ONLY = [
    "openai",
    "anthropic",
    "google",
    "vertexai",
    "deepseek",
    "cfgtest-openai",
    "cfgtest-azure",
    None,
]


class TestPreflightScope:
    """Which providers the runtime probe applies to."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", _API_ONLY)
    async def test_api_providers_are_never_probed(self, provider) -> None:
        """An API provider must start on a host with NO node.

        The probe is patched to explode: if the dispatcher reaches it for one
        of these providers, the test fails with that explosion rather than
        silently passing. This is the negative control for claim 1 — without
        it, a blanket implementation would satisfy the claude-code test below
        and still break every node-less deployment.
        """
        boom = AsyncMock(
            side_effect=AssertionError(
                f"provider {provider!r} is API-only and must not be "
                "runtime-probed — it needs no node and no CLI binary"
            )
        )
        with patch(
            "deep_worker.providers.claude_code_provider.profile.preflight_async",
            boom,
        ):
            await cfg.preflight_provider_async(provider)
        boom.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", _RUNTIME)
    async def test_claude_code_is_probed(self, provider) -> None:
        """The positive arm: claude-code DOES reach its own preflight."""
        probe = AsyncMock(return_value=None)
        with patch(
            "deep_worker.providers.claude_code_provider.profile.preflight_async",
            probe,
        ):
            await cfg.preflight_provider_async(provider)
        probe.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claude_code_failure_propagates_verbatim(self) -> None:
        """The adapter's own cause must reach the caller unchanged.

        ``AgentProcess.start()`` persists ``str(exc)`` as the agent's error
        message and sends it over the readiness channel, so any wrapping or
        generic re-phrasing here would be what the user ends up reading in
        the terminal instead of the real remedy.
        """
        from deep_worker.providers.claude_code_provider.profile import (
            AcpAdapterError,
        )

        cause = AcpAdapterError(
            "claude-code requires node >= 22 (adapter engines) — "
            "found node major 18"
        )
        with patch(
            "deep_worker.providers.claude_code_provider.profile.preflight_async",
            AsyncMock(side_effect=cause),
        ):
            with pytest.raises(AcpAdapterError) as exc_info:
                await cfg.preflight_provider_async("claude-code")

        assert "node >= 22" in str(exc_info.value)
        assert "found node major 18" in str(exc_info.value)


class TestPreflightRunsAtStartup:
    """Claim 2: the probe is wired into the STARTUP path, not just the model.

    A static check, deliberately: driving a full ``AgentProcess.start()``
    needs a registry, an rpyc port, a checkpointer and a live event loop, and
    would prove far less about ordering than reading the one thing that
    matters — that ``start()`` awaits the dispatcher inside the init block
    whose ``except`` produces the ERROR state, and does so BEFORE the agent
    is registered as running.
    """

    def test_start_awaits_the_dispatcher_before_reporting_ready(self) -> None:
        from deep_worker_cli.process import agent_process as ap

        source = Path(ap.__file__).read_text(encoding="utf-8")

        assert "await preflight_provider_async(self.config.provider)" in source, (
            "AgentProcess.start() must await preflight_provider_async() so a "
            "broken provider runtime fails the START, not the first prompt. "
            "Without it, `deep-worker <name>` reports a healthy start for an "
            "agent that can never answer."
        )

        probe_at = source.index("await preflight_provider_async(")
        ready_at = source.index("self._report_ready()")
        assert probe_at < ready_at, (
            "the runtime probe must run BEFORE _report_ready() — otherwise "
            "the launching terminal is told the agent is up and only then "
            "does the adapter check fail."
        )
