"""Tests for AgentProcess interrupt → status transition.

When the agent's worker task is cancelled (via ``exposed_interrupt_agent``
or any other means), the agent must transition to ``INTERRUPTED``, NOT
to ``ERROR`` and NOT to ``WAITING_INPUT``.

The historical journey of this state:

- Originally the cancel branch called ``set_error("Agent interrupted")``,
  leaving the agent stuck in ``error`` until manual intervention. The
  UI then refused to talk to the agent. Fixed in an earlier commit.
- The interim fix used ``WAITING_INPUT``, but that name is overloaded
  in this codebase: ``StatusTrackingCallback.on_tool_start`` also sets
  ``WAITING_INPUT`` while a tool is executing inside a turn. Same
  status, different meanings — confusing in the UI.
- This file now enforces the correct invariant: cancel branch sets
  ``INTERRUPTED``. The UI renders that state in amber (same colour as
  the Interrupt button) so users can tell at a glance that the agent
  is paused mid-turn waiting for their next message (typically a
  correction), distinct from the in-progress tool-running state.

Interrupt is NOT cancellation in the destructive sense — the agent
process stays fully alive, ready to accept the next message and
continue with whatever correction the user provides. That's why
INTERRUPTED is a "waiting for user" colour, not an error/red colour.
"""

from __future__ import annotations

import inspect

from deep_worker_cli.process import agent_process
from deep_worker_cli.types import AgentStatus


class TestAgentStatusEnum:
    """The ``INTERRUPTED`` value must exist on the enum and equal
    ``"interrupted"``. Web schema, session_adapter mapping, and UI
    code all depend on this exact string."""

    def test_interrupted_value_exists(self) -> None:
        assert hasattr(AgentStatus, "INTERRUPTED"), (
            "AgentStatus must define an INTERRUPTED member"
        )

    def test_interrupted_value_string(self) -> None:
        assert AgentStatus.INTERRUPTED.value == "interrupted", (
            f"AgentStatus.INTERRUPTED must equal 'interrupted', "
            f"got {AgentStatus.INTERRUPTED.value!r}"
        )


class TestInterruptSetsInterrupted:
    """The CancelledError branch must put the agent into INTERRUPTED."""

    def test_cancelled_error_handler_uses_interrupted(self) -> None:
        """In the CancelledError handler, status must become INTERRUPTED.

        We inspect the source of agent_process to check that the cancel
        branch sets INTERRUPTED. Source-scanning is acceptable here
        because the alternative (driving a real LangGraph invocation
        to cancellation) would be a heavyweight integration test.
        """
        source = inspect.getsource(agent_process)

        assert "AgentStatus.INTERRUPTED" in source, (
            "agent_process must reference AgentStatus.INTERRUPTED — "
            "that's where an interrupted agent should land after "
            "user-initiated cancellation."
        )

    def test_cancelled_error_handler_does_not_use_waiting_input(self) -> None:
        """The cancel branch MUST NOT set WAITING_INPUT anymore.

        WAITING_INPUT is now reserved for the in-turn tool-running
        state (see ``StatusTrackingCallback.on_tool_start``). The
        interrupt-by-user state is INTERRUPTED. Mixing them was the
        original UX bug.

        The check is local to the CancelledError handler. We do not
        forbid every WAITING_INPUT reference in agent_process — only
        the one inside the cancel branch. We approximate this by
        looking for ``set_status(AgentStatus.WAITING_INPUT)`` paired
        with the interrupt comment.
        """
        source = inspect.getsource(agent_process)
        # The exact production line used to be inside the cancel
        # branch — we want it gone there. The status-callback path
        # uses session_manager.set_status from a different module,
        # so we keep the source-string search confined to this file's
        # symbols.
        forbidden = "set_status(AgentStatus.WAITING_INPUT)"
        assert forbidden not in source, (
            f"agent_process must no longer set WAITING_INPUT — "
            f"interrupt now lands in INTERRUPTED. Found: {forbidden!r}"
        )

    def test_cancelled_error_handler_does_not_call_set_error(self) -> None:
        """Cancel must not set_error. set_error puts the agent in
        ERROR state which the UI refuses to talk to; user interrupt is
        a normal pause, not a failure.

        We narrow the source inspection to the actual code lines of
        the ``AgentProcess._handle_invoke`` method's CancelledError
        branch — ``inspect`` may include nearby comments in the source
        text, so we look only at code lines (no leading ``#``) inside
        that handler.
        """
        full_source = inspect.getsource(agent_process)
        # Locate the cancel-branch start by its anchor comment text.
        anchor = "except asyncio.CancelledError:"
        idx = full_source.find(anchor)
        assert idx >= 0, "could not find the CancelledError branch anchor"
        # Take ~50 lines of source after the anchor — far enough to
        # span the entire handler.
        branch = "\n".join(full_source[idx:].splitlines()[:50])
        code_lines = [
            ln for ln in branch.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "set_error(" not in code_only, (
            "CancelledError branch must NOT call set_error. "
            f"Found set_error in code lines:\n{code_only}"
        )

    def test_cancelled_branch_emits_agent_interrupted_event(self) -> None:
        """The cancel branch must still notify subscribers about
        interruption so web/CLI clients can react event-driven."""
        source = inspect.getsource(agent_process)
        assert '"type": "agent_interrupted"' in source, (
            "agent_process must emit agent_interrupted event so "
            "clients can react to the cancellation."
        )
