"""POLICIES.md §12 — a summarization failure reaches the operator.

The middleware fix (``SummarizationFailedError`` instead of a soft-returned
``"Error generating summary: …"`` string) is only half the contract. The other
half is the promise the policy makes about what the operator then sees:

    the runtime catch-all transitions the agent to ``AgentStatus.ERROR``
    with the message surfaced in the CLI and the web UI

That promise spans two packages. ``deepagents`` raises the exception;
``deep_worker_cli`` classifies it by class NAME and decides between the fatal
path (``set_error`` → ``AgentStatus.ERROR``, persistent ``type="error"``
notification) and the recoverable path (``SystemMessage``, agent stays
``IDLE``, benign ``type="system"`` note).

Nothing in the middleware suite can observe that decision, and nothing in the
runtime suite knows the class exists. A rename on either side — or dropping the
entry from ``FATAL_EXCEPTION_TYPES`` — silently routes the failure down the
recoverable path: the agent would report "please type continue" and stay IDLE
while its context was gone. That is the original incident wearing a different
mask, so it gets its own pin here, at the seam.
"""

from __future__ import annotations

import pytest

from deep_worker_cli.exceptions import is_fatal_exception, is_provider_api_error


def _middleware_error_classes() -> tuple[type[BaseException], type[BaseException]]:
    from deepagents.middleware.summarization import (
        HistoryOffloadError,
        SummarizationFailedError,
    )

    return SummarizationFailedError, HistoryOffloadError


class TestSummarizationFailureIsRoutedToTheFatalPath:
    """The runtime's own classifier must send these down the ERROR branch."""

    def test_both_summarization_failures_are_fatal(self) -> None:
        for cls in _middleware_error_classes():
            assert is_fatal_exception(cls("boom")), (
                f"{cls.__name__} must be fatal: the recoverable path leaves the "
                "agent IDLE with a 'type continue' note, which is exactly the "
                "silent failure POLICIES.md §12 forbids"
            )

    def test_not_misrouted_to_the_recoverable_api_error_path(self) -> None:
        """Fatal and recoverable-provider-error are mutually exclusive here.

        ``is_provider_api_error`` selects a *persistent* notification but keeps
        the agent IDLE. Losing context is not something the agent can continue
        through, so these classes must not appear in that set.
        """
        for cls in _middleware_error_classes():
            assert not is_provider_api_error(cls("boom"))

    def test_classification_survives_the_cross_package_boundary(self) -> None:
        """Matching is by NAME, and the two packages never import each other.

        This asserts the real classes the middleware raises are matched — not
        look-alikes built in the test — so a rename on the ``deepagents`` side
        fails here instead of silently downgrading the failure in production.
        """
        summarization_failed, offload_failed = _middleware_error_classes()
        assert summarization_failed.__name__ == "SummarizationFailedError"
        assert offload_failed.__name__ == "HistoryOffloadError"

    @pytest.mark.asyncio
    async def test_status_callback_flips_the_agent_to_error(self) -> None:
        """The seam that actually performs the transition.

        ``StatusTrackingCallback.on_tool_error`` is the shared implementation of
        the fatal-vs-recoverable rule (it mirrors the catch-all in
        ``agent_process._handle_invoke``). Driving it with the real exception
        proves the ERROR transition, rather than re-asserting the lookup table.
        """
        from unittest.mock import AsyncMock, MagicMock

        from deep_worker_cli.process.status_callback import StatusTrackingCallback

        summarization_failed, _ = _middleware_error_classes()

        session_manager = MagicMock()
        session_manager.set_error = AsyncMock()
        session_manager.set_status = AsyncMock()

        handler = StatusTrackingCallback(session_manager=session_manager)
        await handler.on_tool_error(
            summarization_failed("Failed to generate conversation summary: boom"),
            run_id="run-1",
        )

        session_manager.set_error.assert_awaited_once()
        assert "boom" in str(session_manager.set_error.await_args.args[0]), (
            "the operator must see WHY summarization failed"
        )
        session_manager.set_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ordinary_error_still_takes_the_recoverable_path(self) -> None:
        """Control: the fix must not make every failure fatal."""
        from unittest.mock import AsyncMock, MagicMock

        from deep_worker_cli.process.status_callback import StatusTrackingCallback

        session_manager = MagicMock()
        session_manager.set_error = AsyncMock()
        session_manager.set_status = AsyncMock()

        handler = StatusTrackingCallback(session_manager=session_manager)
        await handler.on_tool_error(ValueError("just a tool being sloppy"), run_id="run-2")

        session_manager.set_error.assert_not_awaited()
        session_manager.set_status.assert_awaited_once()
