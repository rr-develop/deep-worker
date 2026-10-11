"""sum_query wipe must reset ACP session state before invoking.

Regression harness for the alternating burn/ok pattern reported by
rg1_worker (2026-08-22) on the repo_graph extractor:

    ACP client state on disk still carries ``sent_count=k>1`` and the
    pre-wipe ``session_id`` from the previous document. sum_query
    ``aupdate_state``-wipes the LangGraph channel to length 1. The next
    ``ainvoke`` hits the ``chat_model.py:571`` fail-loud guard
    (``sent_count > len(state_messages)``), burning the first invoke.

The fix belongs on the wipe side: when ``_run_startup_or_send_query``
shrinks the LangGraph messages, it must also call the model's
``areset_after_summary(thread_id=...)`` — the same port
``SummarizerPort`` uses — so ACP state cannot outlive the wipe.

These tests pin the ORDER (offload → wipe → reset → invoke) and pin that
the fix is opt-in by capability (a model without the method is skipped
silently — API-only providers stay unaffected).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import RemoveMessage

from deep_worker_cli.process.agent_process import AgentProcess

# Reuse the harness helpers built for the sibling test file — same
# invariants ("real middleware, mocked graph"), same setup, no drift.
from tests.unit_tests.process.test_auto_run_startup_query import (
    MockBackend,
    _make_fake_self,
    make_conversation_messages,
)


def _install_model_with_reset(fake: MagicMock) -> AsyncMock:
    """Wire an ACP-shaped model into ``fake`` and return its ``areset`` spy.

    Production reads it as ``self._model`` (set in ``AgentProcess.start``,
    agent_process.py:1341). No graph descent — the model is already held
    on the AgentProcess.
    """
    # A real CLASS, not a bare MagicMock with attributes hung on it.
    # ``implements_summarizer_port`` inspects the TYPE (an instance probe
    # reads truthy for every MagicMock attribute, which would make every test
    # double look like a live session provider) and requires BOTH halves of
    # the port, so the double has to actually have both on its class.
    spy = AsyncMock(return_value=None)

    class _PortModel:
        async def asummarize(self, **_kw: object) -> None: ...

        async def areset_after_summary(self, **kwargs: object) -> None:
            await spy(**kwargs)

    fake._model = _PortModel()
    return spy


def _install_model_without_reset(fake: MagicMock) -> MagicMock:
    """Wire an API-only model (no ACP reset) so we can pin the skip path."""
    model = MagicMock(spec=["invoke", "ainvoke"])  # no areset_after_summary
    fake._model = model
    return model


@pytest.mark.asyncio
async def test_sum_query_wipe_calls_areset_after_summary_on_the_model() -> None:
    """Between the wipe and the first invoke, areset_after_summary must fire.

    Without this call the ACP client's ``sent_count`` (persisted from the
    previous document's turn) exceeds the freshly wiped
    ``len(state_messages)`` and the very next ``ainvoke`` burns on the
    ``chat_model.py:571`` fail-loud guard.
    """
    backend = MockBackend()
    current = make_conversation_messages(num_old=4)
    fake = _make_fake_self(backend=backend, current_messages=current)
    areset = _install_model_with_reset(fake)

    await AgentProcess._run_startup_or_send_query(fake, "next document", None)

    areset.assert_awaited_once_with(thread_id=fake.name)


@pytest.mark.asyncio
async def test_sum_query_reset_happens_after_wipe_and_before_invoke() -> None:
    """Order matters: reset must land between the wipe and the invoke.

    Reset BEFORE the wipe: ACP state re-populates from the pre-wipe
    session as the next turn re-enters ``ainvoke`` (the state seed is
    read lazily on demand, but for symmetry with the summarizer-port
    contract we still pin order strictly).

    Reset AFTER the invoke: too late — the invoke has already burned.
    """
    backend = MockBackend()
    current = make_conversation_messages(num_old=4)
    fake = _make_fake_self(backend=backend, current_messages=current)
    areset = _install_model_with_reset(fake)

    events: list[str] = []
    orig_update = fake.agent.aupdate_state

    async def _record_update(*args: Any, **kwargs: Any) -> Any:
        events.append("wipe")
        return await orig_update(*args, **kwargs)

    async def _record_reset(**kwargs: Any) -> None:
        events.append(f"reset:{kwargs.get('thread_id')}")

    async def _record_invoke(*args: Any, **kwargs: Any) -> Any:
        events.append("invoke")
        return {"success": True}

    fake.agent.aupdate_state = AsyncMock(side_effect=_record_update)
    areset.side_effect = _record_reset
    fake._handle_invoke_owned = AsyncMock(side_effect=_record_invoke)

    await AgentProcess._run_startup_or_send_query(fake, "next document", None)

    assert events == ["wipe", f"reset:{fake.name}", "invoke"], events


@pytest.mark.asyncio
async def test_sum_query_reset_skipped_for_models_without_the_port() -> None:
    """API-only models do not carry ACP state — the reset call must be a no-op.

    Capability-based skip (``getattr(model, 'areset_after_summary', None)``
    returns nothing callable). The wipe + invoke sequence still runs.
    """
    backend = MockBackend()
    current = make_conversation_messages(num_old=4)
    fake = _make_fake_self(backend=backend, current_messages=current)
    _install_model_without_reset(fake)

    await AgentProcess._run_startup_or_send_query(fake, "next document", None)

    fake.agent.aupdate_state.assert_awaited_once()
    fake._handle_invoke_owned.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_only_path_does_not_call_areset() -> None:
    """No wipe → no reset. The pure ``query`` path must not touch ACP state."""
    backend = MockBackend()
    fake = _make_fake_self(backend=backend)
    areset = _install_model_with_reset(fake)

    await AgentProcess._run_startup_or_send_query(fake, None, "do the thing")

    areset.assert_not_awaited()
    fake.agent.aupdate_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_offload_failure_skips_reset_too() -> None:
    """Atomicity: if offload fails, neither wipe nor reset may run."""
    backend = MockBackend(should_fail=True, error_message="disk full")
    current = make_conversation_messages(num_old=4)
    fake = _make_fake_self(backend=backend, current_messages=current)
    areset = _install_model_with_reset(fake)

    with pytest.raises(Exception):
        await AgentProcess._run_startup_or_send_query(
            fake, "next document", None
        )

    fake.agent.aupdate_state.assert_not_awaited()
    areset.assert_not_awaited()
    fake._handle_invoke_owned.assert_not_awaited()
