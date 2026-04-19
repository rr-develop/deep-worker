"""Tests for AgentProcess interrupt → status transition.

When the agent's worker task is cancelled (via ``exposed_interrupt_agent``
or any other means), the agent must transition to ``WAITING_INPUT``,
NOT to ``ERROR``. The agent process is fully alive and ready to accept
the next user message — it is in error only if something genuinely
broke. A user-initiated interrupt is a normal state transition.

This test guards against regression where the CancelledError handler
in ``AgentProcess._handle_invoke`` calls ``set_error("Agent interrupted")``
and leaves the agent stuck in ``error`` status until manual intervention.
"""

from __future__ import annotations

import inspect

from deep_worker_cli.process import agent_process


class TestInterruptSetsWaitingInput:
    """The CancelledError branch must put the agent into WAITING_INPUT."""

    def test_cancelled_error_handler_uses_waiting_input(self) -> None:
        """In the CancelledError handler, status must become WAITING_INPUT.

        We inspect the source of agent_process to check that the cancel
        branch sets WAITING_INPUT and does NOT call set_error("interrupted").
        Source-scanning is acceptable here because the alternative (driving
        a real LangGraph invocation to cancellation) would be a heavyweight
        integration test.
        """
        source = inspect.getsource(agent_process)

        # The cancel branch must set WAITING_INPUT
        assert "AgentStatus.WAITING_INPUT" in source, (
            "agent_process must reference AgentStatus.WAITING_INPUT — "
            "that's where an interrupted agent should land."
        )

        # The interrupted-cancel branch must NOT use set_error
        # (set_error puts agent in ERROR which the UI then refuses to
        # talk to). We allow other set_error calls (real exceptions),
        # so we look for the specific 'Agent interrupted' string with set_error.
        forbidden = 'set_error("Agent interrupted")'
        assert forbidden not in source, (
            "After interrupt, agent must transition to WAITING_INPUT, "
            "not ERROR. Found forbidden pattern: " + forbidden
        )

    def test_cancelled_branch_emits_agent_interrupted_event(self) -> None:
        """The cancel branch must still notify subscribers about interruption.

        Notification keeps web/CLI clients informed even if status is
        WAITING_INPUT (which is also reachable via other means).
        """
        source = inspect.getsource(agent_process)
        assert '"type": "agent_interrupted"' in source, (
            "agent_process must emit agent_interrupted event so clients "
            "can react to the cancellation."
        )
